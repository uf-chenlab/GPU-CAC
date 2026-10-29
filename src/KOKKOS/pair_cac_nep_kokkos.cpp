/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#if __has_include("nep.h")

#include "pair_cac_nep_kokkos.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "kokkos.h"

#include <cstdlib>
#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   device functors (named: CUDA extended lambdas cannot live in protected members)
------------------------------------------------------------------------- */

namespace {

// x(s) = sum_r w(s,r) nodes(base(s)+r) for element sites, nodes(base(s)) for atoms
template <class DeviceType> struct CACNepSitePos {
  Kokkos::View<int *, DeviceType> base, isatom;
  Kokkos::View<double *[8], Kokkos::LayoutRight, DeviceType> w;
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> nodes;
  typename ArrayTypes<DeviceType>::t_kkfloat_1d_3_lr x;
  KOKKOS_INLINE_FUNCTION void operator()(const int s) const
  {
    const int b = base(s);
    if (isatom(s)) {
      for (int d = 0; d < 3; d++) x(s, d) = (KK_FLOAT) nodes(b, d);
    } else {
      double p[3] = {0.0, 0.0, 0.0};
      for (int r = 0; r < 8; r++) {
        const double wr = w(s, r);
        p[0] += nodes(b + r, 0) * wr;
        p[1] += nodes(b + r, 1) * wr;
        p[2] += nodes(b + r, 2) * wr;
      }
      for (int d = 0; d < 3; d++) x(s, d) = (KK_FLOAT) p[d];
    }
  }
};

// out(q) = (f, e) of the site of quadrature point q
template <class DeviceType> struct CACNepGather {
  Kokkos::View<int *, DeviceType> quad_site;
  typename ArrayTypes<DeviceType>::t_kkacc_1d_3 f;
  typename ArrayTypes<DeviceType>::t_kkacc_1d eatom;
  Kokkos::View<double *[4], Kokkos::LayoutRight, DeviceType> out;
  KOKKOS_INLINE_FUNCTION void operator()(const int q) const
  {
    const int s = quad_site(q);
    out(q, 0) = f(s, 0);
    out(q, 1) = f(s, 1);
    out(q, 2) = f(s, 2);
    out(q, 3) = eatom(s);
  }
};

// padded 2d neighbor array and counts from the flat CSR site lists
template <class DeviceType> struct CACNepExpand {
  Kokkos::View<const int *, DeviceType> flat, first;
  typename ArrayTypes<DeviceType>::t_neighbors_2d neighbors;
  typename ArrayTypes<DeviceType>::t_int_1d numneigh;
  KOKKOS_INLINE_FUNCTION void operator()(const int s) const
  {
    const int b = first(s);
    const int n = first(s + 1) - b;
    numneigh(s) = n;
    for (int k = 0; k < n; k++) neighbors(s, k) = flat(b + k);
  }
};

// neighbor lists from a cell grid over the sites.  MODE 0: quadrature sites (count, mark neighbors
// as centers); MODE 1: other centers (count); MODE 2: fill the padded list of every site with
// a non-zero count.
template <class DeviceType, int MODE> struct CACNepQuery {
  Kokkos::View<const double *, DeviceType> xs, xb;
  Kokkos::View<const int *, DeviceType> order, bstart, sbin, isquad;
  Kokkos::View<int *, DeviceType> has, cnt;
  typename ArrayTypes<DeviceType>::t_neighbors_2d neigh;
  int nbx, nby, nbz, rx, ry, rz;
  double cutsq;
  KOKKOS_INLINE_FUNCTION void operator()(const int s) const
  {
    if (MODE == 0) {
      if (!isquad(s)) return;
      has(s) = 1;
    } else if (MODE == 1) {
      if (isquad(s) || !has(s)) return;
    } else {
      if (cnt(s) == 0) return;
    }
    const double x0 = xs(3 * s), y0 = xs(3 * s + 1), z0 = xs(3 * s + 2);
    const int b = sbin(s);
    const int cx = b % nbx, cy = (b / nbx) % nby, cz = b / (nbx * nby);
    const int ix0 = cx - rx < 0 ? 0 : cx - rx, ix1 = cx + rx > nbx - 1 ? nbx - 1 : cx + rx;
    const int iy0 = cy - ry < 0 ? 0 : cy - ry, iy1 = cy + ry > nby - 1 ? nby - 1 : cy + ry;
    const int iz0 = cz - rz < 0 ? 0 : cz - rz, iz1 = cz + rz > nbz - 1 ? nbz - 1 : cz + rz;
    int n = 0;
    for (int iz = iz0; iz <= iz1; iz++)
      for (int iy = iy0; iy <= iy1; iy++) {
        const long row = ((long) iz * nby + iy) * nbx;
        const int kend = bstart(row + ix1 + 1);
        for (int k = bstart(row + ix0); k < kend; k++) {
          const double dx = x0 - xb(3 * k);
          const double dy = y0 - xb(3 * k + 1);
          const double dz = z0 - xb(3 * k + 2);
          if (dx * dx + dy * dy + dz * dz < cutsq) {
            const int j = order(k);
            if (j == s) continue;
            if (MODE == 2) neigh(s, n) = j;
            if (MODE == 0) has(j) = 1;
            n++;
          }
        }
      }
    if (MODE != 2) cnt(s) = n;
  }
};

template <class DeviceType> struct CACNepCompact {
  Kokkos::View<const int *, DeviceType> has;
  Kokkos::View<int *, DeviceType> ilist;
  KOKKOS_INLINE_FUNCTION void operator()(const int s, int &upd, const bool final) const
  {
    if (has(s)) {
      if (final) ilist(upd) = s;
      upd++;
    }
  }
};

template <class DeviceType> struct CACNepMaxCount {
  Kokkos::View<const int *, DeviceType> cnt;
  KOKKOS_INLINE_FUNCTION void operator()(const int s, int &m) const
  {
    if (cnt(s) > m) m = cnt(s);
  }
};

}    // namespace

/* ---------------------------------------------------------------------- */

template <class DeviceType>
PairCACNEPKokkos<DeviceType>::PairCACNEPKokkos(LAMMPS *lmp) : PairCACNEP(lmp)
{
  dev_table_ncalls = -1;
  site_host_positions = 0;
  dev_lists = 0;
  ncenters = 0;
  lists_pending = false;
  static_valid = false;
  neigh_cols = 0;
  nnode_rows = 0;
  max_site_neigh = 0;
}

template <class DeviceType> PairCACNEPKokkos<DeviceType>::~PairCACNEPKokkos() = default;

/* ----------------------------------------------------------------------
   pair_style cac/nep/kk [one] [fp64 | fp32 | df64]
------------------------------------------------------------------------- */

template <class DeviceType> void PairCACNEPKokkos<DeviceType>::settings(int narg, char **arg)
{
  engine.use_fp32 = engine.use_df64 = false;
  int nbase = 0;
  char *base_args[1];
  for (int i = 0; i < narg; i++) {
    if (strcmp(arg[i], "fp32") == 0 || strcmp(arg[i], "mixed") == 0) engine.use_fp32 = true;
    else if (strcmp(arg[i], "df64") == 0) engine.use_df64 = true;
    else if (strcmp(arg[i], "fp64") == 0 || strcmp(arg[i], "double") == 0) continue;
    else if (nbase == 0) base_args[nbase++] = arg[i];
    else error->all(FLERR, "Illegal pair_style cac/nep/kk command: {}", arg[i]);
  }
  PairCACNEP::settings(nbase, base_args);
}

/* ----------------------------------------------------------------------
   Kokkos is initialized here if LAMMPS was started without "-k on"
   (CAC itself runs on the host); one device per MPI rank
------------------------------------------------------------------------- */

template <class DeviceType> void PairCACNEPKokkos<DeviceType>::init_style()
{
  PairCACNEP::init_style();
  dev_lists = (!legacy_lists && getenv("CAC_HOST_LISTS") == nullptr) ? 1 : 0;
  if (lmp->kokkos)
    error->all(FLERR, "pair_style cac/nep/kk offloads the CAC force kernel itself; run without '-k on'");
  if (!Kokkos::is_initialized()) {
    Kokkos::InitializationSettings args;
    args.set_map_device_id_by("mpi_rank");
    KokkosLMP::initialize(args, error);
    if (comm->me == 0)
      utils::logmesg(lmp, "  cac/nep/kk: Kokkos initialized, execution space {}\n",
                     DeviceType::execution_space::name());
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void PairCACNEPKokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairCACNEP::coeff(narg, arg);
  if (!Kokkos::is_initialized()) {
    Kokkos::InitializationSettings args;
    args.set_map_device_id_by("mpi_rank");
    KokkosLMP::initialize(args, error);
  }
  engine.setup(nep_model, type_map, atom->ntypes, error);
}

/* ----------------------------------------------------------------------
   upload the site table and site neighbor lists after a neighbor build
------------------------------------------------------------------------- */

template <class DeviceType> void PairCACNEPKokkos<DeviceType>::upload_site_table()
{
  if (!dev_lists) build_centers();
  const int *element_type = atom->element_type;

  // static part (node rows, interpolation weights, types, quadrature map): only depends on which
  // lattice sites exist, so it is reused when a neighbor rebuild left the site set unchanged

  const bool same = static_valid && cache_owner == site_owner && cache_poly == site_poly &&
      cache_type == site_type && cache_w == site_w && cache_quad_site == quad_site;
  if (!same) {

    // node rows: one block of 8 rows per element (owner,poly) used by a site, 1 row per atom

    std::unordered_map<long, int> base_of;
    base_of.reserve(2 * nsite + 16);
    node_rows.clear();
    pair_base_host.assign(nsite, 0);
    nnode_rows = 0;
    const long np = std::max(1, atom->maxpoly);
    for (int s = 0; s < nsite; s++) {
      const long key = (long) site_owner[s] * np + site_poly[s];
      auto it = base_of.find(key);
      if (it == base_of.end()) {
        it = base_of.emplace(key, nnode_rows).first;
        node_rows.emplace_back(site_owner[s], site_poly[s]);
        nnode_rows += (element_type[site_owner[s]] == 0) ? 1 : 8;
      }
      pair_base_host[s] = it->second;
    }

    d_site_base = Kokkos::View<int *, DeviceType>("cac/nep/kk:site_base", nsite);
    d_site_isatom = Kokkos::View<int *, DeviceType>("cac/nep/kk:site_isatom", nsite);
    d_site_w = Kokkos::View<double *[8], Kokkos::LayoutRight, DeviceType>("cac/nep/kk:site_w", nsite);
    d_nodes = Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType>("cac/nep/kk:nodes", nnode_rows);
    h_nodes = Kokkos::create_mirror_view(d_nodes);
    d_type = typename AT::t_int_1d("cac/nep/kk:type", nsite);
    d_x = typename AT::t_kkfloat_1d_3_lr("cac/nep/kk:x", nsite);
    d_f = typename AT::t_kkacc_1d_3("cac/nep/kk:f", nsite);
    d_eatom = typename AT::t_kkacc_1d("cac/nep/kk:eatom", nsite);
    d_vatom = typename AT::t_kkacc_1d_6("cac/nep/kk:vatom", nsite);

    auto h_base = Kokkos::create_mirror_view(d_site_base);
    auto h_isatom = Kokkos::create_mirror_view(d_site_isatom);
    auto h_w = Kokkos::create_mirror_view(d_site_w);
    auto h_type = Kokkos::create_mirror_view(d_type);
    for (int s = 0; s < nsite; s++) {
      h_base(s) = pair_base_host[s];
      h_isatom(s) = (element_type[site_owner[s]] == 0) ? 1 : 0;
      for (int r = 0; r < 8; r++) h_w(s, r) = site_w[8 * (size_t) s + r];
      h_type(s) = site_type[s];
    }
    Kokkos::deep_copy(d_site_base, h_base);
    Kokkos::deep_copy(d_site_isatom, h_isatom);
    Kokkos::deep_copy(d_site_w, h_w);
    Kokkos::deep_copy(d_type, h_type);

    d_quad_site = Kokkos::View<int *, DeviceType>("cac/nep/kk:quad_site", std::max(1, nquadpt));
    auto h_qs = Kokkos::create_mirror_view(d_quad_site);
    for (int pq = 0; pq < nquadpt; pq++) h_qs(pq) = quad_site[pq];
    Kokkos::deep_copy(d_quad_site, h_qs);
    d_quad_out = Kokkos::View<double *[4], Kokkos::LayoutRight, DeviceType>("cac/nep/kk:quad_out",
                                                                            std::max(1, nquadpt));
    h_quad_out = Kokkos::create_mirror_view(d_quad_out);

    cache_owner = site_owner;
    cache_poly = site_poly;
    cache_type = site_type;
    cache_w = site_w;
    cache_quad_site = quad_site;
    static_valid = true;
  }

  if (dev_lists) {
    lists_pending = true;
    dev_table_ncalls = site_table_ncalls;
    return;
  }

  // site neighbor lists: flat CSR to the device, expanded to the padded 2d array there

  max_site_neigh = 1;
  for (int s : centers) max_site_neigh = std::max(max_site_neigh, site_numneigh[s]);
  if (!same || max_site_neigh > neigh_cols) {
    neigh_cols = std::max(max_site_neigh, (int) (1.1 * max_site_neigh));
    d_neighbors = typename AT::t_neighbors_2d(Kokkos::view_alloc("cac/nep/kk:neighbors", Kokkos::WithoutInitializing),
                                              nsite, neigh_cols);
  }
  if (!same) d_numneigh = typename AT::t_int_1d("cac/nep/kk:numneigh", nsite);
  const size_t nflat = slist.size();
  if (d_slist.extent(0) < std::max<size_t>(1, nflat))
    d_slist = Kokkos::View<int *, DeviceType>(Kokkos::view_alloc("cac/nep/kk:slist", Kokkos::WithoutInitializing),
                                              std::max<size_t>(1, nflat + nflat / 8));
  if (d_sfirst.extent(0) != (size_t) nsite + 1)
    d_sfirst = Kokkos::View<int *, DeviceType>("cac/nep/kk:sfirst", nsite + 1);
  if (d_ilist.extent(0) < std::max<size_t>(1, centers.size()) || !same)
    d_ilist = typename AT::t_int_1d("cac/nep/kk:ilist", std::max<size_t>(1, centers.size()));

  typedef Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> HostConstInt;
  if (nflat) Kokkos::deep_copy(Kokkos::subview(d_slist, std::make_pair((size_t) 0, nflat)), HostConstInt(slist.data(), nflat));
  Kokkos::deep_copy(d_sfirst, HostConstInt(slist_first.data(), (size_t) nsite + 1));
  if (!centers.empty())
    Kokkos::deep_copy(Kokkos::subview(d_ilist, std::make_pair((size_t) 0, centers.size())),
                      HostConstInt(centers.data(), centers.size()));

  CACNepExpand<DeviceType> ex;
  ex.flat = d_slist;
  ex.first = d_sfirst;
  ex.neighbors = d_neighbors;
  ex.numneigh = d_numneigh;
  Kokkos::parallel_for("CACNepExpand", Kokkos::RangePolicy<DeviceType>(0, nsite), ex);

  engine.ensure_short_lists((int) centers.size(), max_site_neigh);
  dev_table_ncalls = site_table_ncalls;
}

/* ----------------------------------------------------------------------
   neighbor lists of all centers on the device: the cell grid (counting sort) is built on the
   host from the site positions of this rebuild, the counting/filling passes run on the device
------------------------------------------------------------------------- */

template <class DeviceType> void PairCACNEPKokkos<DeviceType>::build_device_lists()
{
  double w0 = tm_on ? MPI_Wtime() : 0.0;
  auto lap = [&](const char *what) {
    if (tm_on && getenv("CAC_TIMING_DETAIL")) {
      Kokkos::fence();
      const double w1 = MPI_Wtime();
      fprintf(stderr, "  dev-lists %s %.2f ms\n", what, 1e3 * (w1 - w0));
      w0 = w1;
    }
  };
  SiteGrid g;
  build_site_grid(g);
  lap("host grid");

  typedef Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> HInt;
  typedef Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> HDbl;
  auto upload_int = [&](Kokkos::View<int *, DeviceType> &d, const std::vector<int> &v) {
    if (d.extent(0) != v.size())
      d = Kokkos::View<int *, DeviceType>(Kokkos::view_alloc("cac/nep/kk:grid", Kokkos::WithoutInitializing), v.size());
    Kokkos::deep_copy(d, HInt(v.data(), v.size()));
  };
  auto upload_dbl = [&](Kokkos::View<double *, DeviceType> &d, const std::vector<double> &v) {
    if (d.extent(0) != v.size())
      d = Kokkos::View<double *, DeviceType>(Kokkos::view_alloc("cac/nep/kk:grid", Kokkos::WithoutInitializing), v.size());
    Kokkos::deep_copy(d, HDbl(v.data(), v.size()));
  };
  upload_dbl(d_xs, site_x);
  upload_dbl(d_xb, g.xb);
  upload_int(d_order, g.order);
  upload_int(d_bstart, g.bstart);
  upload_int(d_sbin, g.sbin);
  std::vector<int> isq(nsite, 0);
  for (int pq = 0; pq < nquadpt; pq++) isq[quad_site[pq]] = 1;
  upload_int(d_isquad, isq);
  lap("upload");

  if (d_has.extent(0) != (size_t) nsite)
    d_has = Kokkos::View<int *, DeviceType>("cac/nep/kk:has", nsite);
  if (d_numneigh.extent(0) != (size_t) nsite) d_numneigh = typename AT::t_int_1d("cac/nep/kk:numneigh", nsite);
  Kokkos::deep_copy(d_has, 0);
  Kokkos::deep_copy(d_numneigh, 0);
  if (d_ilist.extent(0) < (size_t) nsite) d_ilist = typename AT::t_int_1d("cac/nep/kk:ilist", nsite);

  const double cut = site_list_cut;
  auto setup = [&](auto &q) {
    q.xs = d_xs;
    q.xb = d_xb;
    q.order = d_order;
    q.bstart = d_bstart;
    q.sbin = d_sbin;
    q.isquad = d_isquad;
    q.has = d_has;
    q.cnt = d_numneigh;
    q.nbx = g.nb[0];
    q.nby = g.nb[1];
    q.nbz = g.nb[2];
    q.rx = g.reach[0];
    q.ry = g.reach[1];
    q.rz = g.reach[2];
    q.cutsq = cut * cut;
  };
  Kokkos::RangePolicy<DeviceType> pol(0, nsite);
  CACNepQuery<DeviceType, 0> q0;
  setup(q0);
  Kokkos::parallel_for("CACNepQuad", pol, q0);
  CACNepQuery<DeviceType, 1> q1;
  setup(q1);
  lap("quad count");
  Kokkos::parallel_for("CACNepCenters", pol, q1);
  lap("center count");

  int maxn = 1;
  CACNepMaxCount<DeviceType> mc;
  mc.cnt = d_numneigh;
  Kokkos::parallel_reduce("CACNepMax", pol, mc, Kokkos::Max<int>(maxn));
  max_site_neigh = std::max(1, maxn);
  if (max_site_neigh > neigh_cols || d_neighbors.extent(0) != (size_t) nsite) {
    neigh_cols = std::max(max_site_neigh, (int) (1.1 * max_site_neigh));
    d_neighbors = typename AT::t_neighbors_2d(Kokkos::view_alloc("cac/nep/kk:neighbors", Kokkos::WithoutInitializing),
                                              nsite, neigh_cols);
  }
  lap("max+alloc");
  CACNepQuery<DeviceType, 2> q2;
  setup(q2);
  q2.neigh = d_neighbors;
  Kokkos::parallel_for("CACNepFill", pol, q2);

  lap("fill");
  int inum = 0;
  CACNepCompact<DeviceType> cp;
  cp.has = d_has;
  cp.ilist = d_ilist;
  Kokkos::parallel_scan("CACNepCompact", pol, cp, inum);
  ncenters = inum;
  lap("compact");
  engine.ensure_short_lists(ncenters, max_site_neigh);
  lists_pending = false;
}

/* ----------------------------------------------------------------------
   one force evaluation: nodes -> device, site positions, NEP kernels on
   the site set, quadrature force densities/energies -> host
------------------------------------------------------------------------- */

template <class DeviceType> void PairCACNEPKokkos<DeviceType>::compute_sites(int, int)
{
  double t0 = tm_on ? MPI_Wtime() : 0.0, t1;
  if (dev_table_ncalls != site_table_ncalls) {
    upload_site_table();
    if (tm_on) { Kokkos::fence(); t1 = MPI_Wtime(); tm[6] += t1 - t0; t0 = t1; }
  }

  double ****nodal_positions = atom->nodal_positions;
  double **x = atom->x;
  const int *element_type = atom->element_type;
  int row = 0;
  for (const auto &op : node_rows) {
    const int owner = op.first;
    if (element_type[owner] == 0) {
      for (int d = 0; d < 3; d++) h_nodes(row, d) = x[owner][d];
      row++;
    } else {
      double **nodes = nodal_positions[owner][op.second];
      for (int r = 0; r < 8; r++, row++)
        for (int d = 0; d < 3; d++) h_nodes(row, d) = nodes[r][d];
    }
  }
  Kokkos::deep_copy(d_nodes, h_nodes);

  CACNepSitePos<DeviceType> pos;
  pos.base = d_site_base;
  pos.isatom = d_site_isatom;
  pos.w = d_site_w;
  pos.nodes = d_nodes;
  pos.x = d_x;
  Kokkos::parallel_for("CACNepSitePos", Kokkos::RangePolicy<DeviceType>(0, nsite), pos);
  if (dev_lists && lists_pending) {
    if (tm_on) { Kokkos::fence(); t1 = MPI_Wtime(); tm[7] += t1 - t0; t0 = t1; }
    build_device_lists();
    if (tm_on) { Kokkos::fence(); t1 = MPI_Wtime(); tm[2] += t1 - t0; t0 = t1; }
  }
  if (tm_on) { Kokkos::fence(); t1 = MPI_Wtime(); tm[7] += t1 - t0; t0 = t1; }

  const int cac_virial = atom->CAC_virial;
  Kokkos::deep_copy(d_f, 0.0);
  if (cac_virial) Kokkos::deep_copy(d_vatom, 0.0);

  engine.x = d_x;
  engine.f = d_f;
  engine.d_vatom = d_vatom;
  engine.type = d_type;
  engine.d_neighbors = d_neighbors;
  engine.d_numneigh = d_numneigh;
  engine.d_ilist = d_ilist;
  engine.d_eatom = d_eatom;
  engine.inum = dev_lists ? ncenters : (int) centers.size();
  engine.nall = nsite;
  engine.nmax = nsite;
  engine.need_virial = cac_virial != 0;
  engine.box_length = std::max(domain->xprd, std::max(domain->yprd, domain->zprd));
  engine.error = error;
  if (engine.inum > 0) engine.run();
  if (tm_on) { Kokkos::fence(); t1 = MPI_Wtime(); tm[8] += t1 - t0; t0 = t1; }

  CACNepGather<DeviceType> gather;
  gather.quad_site = d_quad_site;
  gather.f = d_f;
  gather.eatom = d_eatom;
  gather.out = d_quad_out;
  Kokkos::parallel_for("CACNepGather", Kokkos::RangePolicy<DeviceType>(0, nquadpt), gather);
  Kokkos::deep_copy(h_quad_out, d_quad_out);

  for (int pq = 0; pq < nquadpt; pq++) {
    quad_force[3 * (size_t) pq] = h_quad_out(pq, 0);
    quad_force[3 * (size_t) pq + 1] = h_quad_out(pq, 1);
    quad_force[3 * (size_t) pq + 2] = h_quad_out(pq, 2);
    quad_energy[pq] = h_quad_out(pq, 3);
  }

  if (cac_virial) {
    auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_vatom);
    for (int pq = 0; pq < nquadpt; pq++)
      for (int d = 0; d < 6; d++) quad_virial[6 * (size_t) pq + d] = h_v(quad_site[pq], d);
  }
  if (tm_on) tm[9] += MPI_Wtime() - t0;
}

namespace LAMMPS_NS {
template class PairCACNEPKokkos<LMPDeviceType>;
#ifdef KOKKOS_ENABLE_CUDA
template class PairCACNEPKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS

#endif
