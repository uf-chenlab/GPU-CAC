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

#ifndef LMP_CAC_SITE_DEVICE_H
#define LMP_CAC_SITE_DEVICE_H

/* ----------------------------------------------------------------------
   Device side of the CAC site table, shared by the Kokkos versions of the classical
   CAC pair styles (cac/lj/kk, cac/eam/kk, cac/sw/kk, cac/tersoff/kk).

   The CAC machinery (atom style, comm, quadrature, integration, nodal projection) stays on the
   host.  Per step prepare() uploads the nodal positions, interpolates the position of every
   lattice site on the device and, after a neighbor rebuild, builds the neighbor lists of the
   quadrature sites (and optionally of the sites around them) from a cell grid on the device.
   The pair style then runs its own kernels on d_x / d_neighbors and fills d_quad_out
   (force density, energy, virial of every quadrature point), which download() copies back.
------------------------------------------------------------------------- */

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "kokkos.h"
#include "kokkos_type.h"
#include "neighbor.h"
#include "pair_cac.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <vector>

namespace LAMMPS_NS {

namespace CACDev {

// x(s) = sum_r w(s,r) nodes(base(s)+r) for element sites, nodes(base(s)) for atoms
template <class DeviceType> struct SitePos {
  Kokkos::View<int *, DeviceType> base, isatom;
  Kokkos::View<double *[8], Kokkos::LayoutRight, DeviceType> w;
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> nodes;
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> x;
  KOKKOS_INLINE_FUNCTION void operator()(const int s) const
  {
    const int b = base(s);
    if (isatom(s)) {
      for (int d = 0; d < 3; d++) x(s, d) = nodes(b, d);
    } else {
      double p[3] = {0.0, 0.0, 0.0};
      for (int r = 0; r < 8; r++) {
        const double wr = w(s, r);
        p[0] += nodes(b + r, 0) * wr;
        p[1] += nodes(b + r, 1) * wr;
        p[2] += nodes(b + r, 2) * wr;
      }
      for (int d = 0; d < 3; d++) x(s, d) = p[d];
    }
  }
};

// neighbor lists from a cell grid over the sites.  MODE 0: quadrature sites (optionally mark the neighbors as
// centers); MODE 1: the other centers.  Candidates are screened in single precision with a conservative margin
// and confirmed in double precision, so the result is exact.  The list is written into the padded array while
// it fits (cols); cnt always holds the true number of neighbors.
template <class DeviceType, int MODE> struct Query {
  Kokkos::View<const double *, DeviceType> xs, xb;
  Kokkos::View<const float *, DeviceType> xsf, xbf;
  Kokkos::View<const int *, DeviceType> order, bstart, sbin, isquad;
  Kokkos::View<int *, DeviceType> has, cnt;
  typename ArrayTypes<DeviceType>::t_neighbors_2d neigh;
  int nbx, nby, nbz, rx, ry, rz, mark, cols;
  double cutsq;
  float cutsq_f;
  KOKKOS_INLINE_FUNCTION void operator()(const int s) const
  {
    if (MODE == 0) {
      if (!isquad(s)) return;
      has(s) = 1;
    } else {
      if (isquad(s) || !has(s)) return;
    }
    const double x0 = xs(3 * s), y0 = xs(3 * s + 1), z0 = xs(3 * s + 2);
    const float fx0 = xsf(3 * s), fy0 = xsf(3 * s + 1), fz0 = xsf(3 * s + 2);
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
          const float fdx = fx0 - xbf(3 * k);
          const float fdy = fy0 - xbf(3 * k + 1);
          const float fdz = fz0 - xbf(3 * k + 2);
          if (fdx * fdx + fdy * fdy + fdz * fdz < cutsq_f) {
            const double dx = x0 - xb(3 * k);
            const double dy = y0 - xb(3 * k + 1);
            const double dz = z0 - xb(3 * k + 2);
            if (dx * dx + dy * dy + dz * dz < cutsq) {
              const int j = order(k);
              if (j == s) continue;
              if (n < cols) neigh(s, n) = j;
              if (MODE == 0 && mark) has(j) = 1;
              n++;
            }
          }
        }
      }
    cnt(s) = n;
  }
};

// ascending order of every neighbor row (shell sort, Ciura gaps): the gathers of the kernels then stream through
// the site arrays instead of jumping from cell to cell
template <class DeviceType> struct SortRows {
  typename ArrayTypes<DeviceType>::t_neighbors_2d neigh;
  Kokkos::View<const int *, DeviceType> cnt;
  KOKKOS_INLINE_FUNCTION void operator()(const int s) const
  {
    const int n = cnt(s);
    if (n < 2) return;
    const int gaps[8] = {701, 301, 132, 57, 23, 10, 4, 1};
    for (int gi = 0; gi < 8; gi++) {
      const int gap = gaps[gi];
      if (gap >= n) continue;
      for (int i = gap; i < n; i++) {
        const int v = neigh(s, i);
        int j = i;
        while (j >= gap && neigh(s, j - gap) > v) {
          neigh(s, j) = neigh(s, j - gap);
          j -= gap;
        }
        neigh(s, j) = v;
      }
    }
  }
};

// float copy of an array
template <class DeviceType> struct ToFloat {
  Kokkos::View<const double *, DeviceType> in;
  Kokkos::View<float *, DeviceType> out;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const { out(i) = (float) in(i); }
};

template <class DeviceType> struct Compact {
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

template <class DeviceType> struct MaxCount {
  Kokkos::View<const int *, DeviceType> cnt;
  KOKKOS_INLINE_FUNCTION void operator()(const int s, int &m) const
  {
    if (cnt(s) > m) m = cnt(s);
  }
};

// quadrature output of center-scatter kernels: (f, e, v) of the site of every quadrature point
template <class DeviceType> struct GatherScatter {
  Kokkos::View<const int *, DeviceType> quad_site;
  Kokkos::View<const double *[3], Kokkos::LayoutRight, DeviceType> f;
  Kokkos::View<const double *, DeviceType> e;
  Kokkos::View<const double *[6], Kokkos::LayoutRight, DeviceType> v;
  Kokkos::View<double *[10], Kokkos::LayoutRight, DeviceType> out;
  int need_v;
  KOKKOS_INLINE_FUNCTION void operator()(const int q) const
  {
    const int s = quad_site(q);
    for (int d = 0; d < 3; d++) out(q, d) = f(s, d);
    out(q, 3) = e(s);
    for (int d = 0; d < 6; d++) out(q, 4 + d) = need_v ? v(s, d) : 0.0;
  }
};

}    // namespace CACDev

template <class DeviceType> class CACSiteDevice {
 public:
  typedef ArrayTypes<DeviceType> AT;
  typedef Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> t_x3;
  typedef Kokkos::View<double *[10], Kokkos::LayoutRight, DeviceType> t_out10;
  typedef Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> HInt;
  typedef Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> HDbl;

  PairCAC *P = nullptr;
  LAMMPS *lmp = nullptr;
  bool center_lists = true;    // also build lists for sites around the quadrature sites (many-body styles)

  // device data valid after prepare()
  int nsite = 0, nquad = 0, ncenters = 0, max_neigh = 1, neigh_cols = 0;
  t_x3 d_x;
  Kokkos::View<int *, DeviceType> d_type, d_quad_site, d_ilist, d_numneigh;
  typename AT::t_neighbors_2d d_neighbors;
  t_out10 d_quad_out;
  typename t_out10::host_mirror_type h_quad_out;
  // scatter accumulation (SW/Tersoff)
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> d_sf;
  Kokkos::View<double *, DeviceType> d_se;
  Kokkos::View<double *[6], Kokkos::LayoutRight, DeviceType> d_sv;

  void init(PairCAC *pair, LAMMPS *lammps, bool centers)
  {
    P = pair;
    lmp = lammps;
    center_lists = centers;
    dev_table_ncalls = -1;
    static_valid = false;
    lists_pending = false;
    if (!Kokkos::is_initialized()) {
      Kokkos::InitializationSettings args;
      args.set_map_device_id_by("mpi_rank");
      KokkosLMP::initialize(args, lmp->error);
    }
    pair->site_host_positions = 0;
  }

  // one step: static table (after a rebuild), node upload, site positions, neighbor lists (after a rebuild)
  // phase timers (CAC_TIMING=1): tm[6] static upload + lists, tm[7] node upload + positions, tm[8] kernels, tm[9] download
  double tick()
  {
    if (!P->tm_on) return 0.0;
    Kokkos::fence();
    return MPI_Wtime();
  }
  void tock(int slot, double t0)
  {
    if (!P->tm_on) return;
    Kokkos::fence();
    P->tm[slot] += MPI_Wtime() - t0;
  }

  void prepare()
  {
    double t0 = tick();
    if (dev_table_ncalls != P->site_table_ncalls) {
      upload_static();
      lists_pending = true;
      dev_table_ncalls = P->site_table_ncalls;
    }
    tock(6, t0);
    t0 = tick();
    pack_nodes();
    CACDev::SitePos<DeviceType> pos;
    pos.base = d_site_base;
    pos.isatom = d_site_isatom;
    pos.w = d_site_w;
    pos.nodes = d_nodes;
    pos.x = d_x;
    Kokkos::parallel_for("CACDevSitePos", Kokkos::RangePolicy<DeviceType>(0, nsite), pos);
    tock(7, t0);
    if (lists_pending) {
      t0 = tick();
      build_lists();
      lists_pending = false;
      tock(6, t0);
    }
  }

  // d_quad_out -> quad_force / quad_energy / quad_virial of the pair style
  void download()
  {
    const double t0 = tick();
    Kokkos::deep_copy(h_quad_out, d_quad_out);
    const int cac_virial = P->atom->CAC_virial;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int pq = 0; pq < nquad; pq++) {
      P->quad_force[3 * (size_t) pq] = h_quad_out(pq, 0);
      P->quad_force[3 * (size_t) pq + 1] = h_quad_out(pq, 1);
      P->quad_force[3 * (size_t) pq + 2] = h_quad_out(pq, 2);
      P->quad_energy[pq] = h_quad_out(pq, 3);
      if (cac_virial)
        for (int d = 0; d < 6; d++) P->quad_virial[6 * (size_t) pq + d] = h_quad_out(pq, 4 + d);
    }
    tock(9, t0);
  }

  // center-scatter styles: zero the accumulators before the kernel, gather per quadrature point after it
  void scatter_zero()
  {
    if (d_sf.extent(0) != (size_t) nsite) {
      d_sf = Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType>("cac/dev:sf", nsite);
      d_se = Kokkos::View<double *, DeviceType>("cac/dev:se", nsite);
      d_sv = Kokkos::View<double *[6], Kokkos::LayoutRight, DeviceType>("cac/dev:sv", nsite);
    }
    Kokkos::deep_copy(d_sf, 0.0);
    Kokkos::deep_copy(d_se, 0.0);
    Kokkos::deep_copy(d_sv, 0.0);
  }

  void scatter_gather()
  {
    CACDev::GatherScatter<DeviceType> g;
    g.quad_site = d_quad_site;
    g.f = d_sf;
    g.e = d_se;
    g.v = d_sv;
    g.out = d_quad_out;
    g.need_v = P->atom->CAC_virial ? 1 : 0;
    Kokkos::parallel_for("CACDevGather", Kokkos::RangePolicy<DeviceType>(0, nquad), g);
  }

 private:
  bigint dev_table_ncalls;
  bool static_valid, lists_pending;
  int nnode_rows = 0;
  std::vector<int> pair_base_host, row_off;
  std::vector<std::pair<int, int>> node_rows;
  Kokkos::View<int *, DeviceType> d_site_base, d_site_isatom;
  Kokkos::View<double *[8], Kokkos::LayoutRight, DeviceType> d_site_w;
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> d_nodes;
  typename Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType>::host_mirror_type h_nodes;
  Kokkos::View<double *, DeviceType> d_xs, d_xb;
  Kokkos::View<float *, DeviceType> d_xsf, d_xbf;
  Kokkos::View<int *, DeviceType> d_order, d_bstart, d_sbin, d_isquad, d_has;
  std::vector<int> cache_owner, cache_poly, cache_type, cache_quad_site;
  long cache_scale_sum = -1;
  long scale_checksum() const
  {
    long sum = 0;
    for (int e = 0; e < P->atom->nlocal + P->atom->nghost; e++)
      sum = sum * 31 + (long) P->atom->element_scale[e][0] * 1000003L + P->atom->element_scale[e][1] * 1009L + P->atom->element_scale[e][2];
    return sum;
  }

  // static part: node rows, interpolation weights, types, quadrature map; reused while the site set is unchanged
  double lap_t = 0.0;
  void lap(const char *what)
  {
    if (!P->tm_on || !getenv("CAC_TIMING_DETAIL")) return;
    Kokkos::fence();
    const double t1 = MPI_Wtime();
    if (lap_t > 0.0) fprintf(stderr, "  dev %-26s %8.2f ms\n", what, 1e3 * (t1 - lap_t));
    lap_t = t1;
  }

  void upload_static()
  {
    PairCAC &p = *P;
    lap("(start)");
    const int *element_type = p.atom->element_type;
    nsite = p.nsite;
    nquad = p.nquadpt;
    // site_w is a function of (owner, poly, cell) and the element scales: owner/poly/type/quad_site decide
    const bool same = static_valid && cache_owner == p.site_owner && cache_poly == p.site_poly &&
        cache_type == p.site_type && cache_quad_site == p.quad_site && cache_scale_sum == scale_checksum();
    lap("static check");
    if (same) return;

    std::unordered_map<long, int> base_of;
    base_of.reserve(2 * nsite + 16);
    node_rows.clear();
    row_off.clear();
    pair_base_host.assign(nsite, 0);
    nnode_rows = 0;
    const long np = std::max(1, p.atom->maxpoly);
    for (int s = 0; s < nsite; s++) {
      const long key = (long) p.site_owner[s] * np + p.site_poly[s];
      auto it = base_of.find(key);
      if (it == base_of.end()) {
        it = base_of.emplace(key, nnode_rows).first;
        node_rows.emplace_back(p.site_owner[s], p.site_poly[s]);
        row_off.push_back(nnode_rows);
        nnode_rows += (element_type[p.site_owner[s]] == 0) ? 1 : 8;
      }
      pair_base_host[s] = it->second;
    }

    lap("node rows hash");
    d_site_base = Kokkos::View<int *, DeviceType>("cac/dev:site_base", nsite);
    d_site_isatom = Kokkos::View<int *, DeviceType>("cac/dev:site_isatom", nsite);
    d_site_w = Kokkos::View<double *[8], Kokkos::LayoutRight, DeviceType>("cac/dev:site_w", nsite);
    d_nodes = Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType>("cac/dev:nodes", nnode_rows);
    h_nodes = Kokkos::create_mirror_view(d_nodes);
    d_type = Kokkos::View<int *, DeviceType>("cac/dev:type", nsite);
    d_x = t_x3("cac/dev:x", nsite);

    auto h_base = Kokkos::create_mirror_view(d_site_base);
    auto h_isatom = Kokkos::create_mirror_view(d_site_isatom);
    auto h_w = Kokkos::create_mirror_view(d_site_w);
    auto h_type = Kokkos::create_mirror_view(d_type);
    for (int s = 0; s < nsite; s++) {
      h_base(s) = pair_base_host[s];
      h_isatom(s) = (element_type[p.site_owner[s]] == 0) ? 1 : 0;
      for (int r = 0; r < 8; r++) h_w(s, r) = p.site_w[8 * (size_t) s + r];
      h_type(s) = p.site_type[s];
    }
    Kokkos::deep_copy(d_site_base, h_base);
    Kokkos::deep_copy(d_site_isatom, h_isatom);
    Kokkos::deep_copy(d_site_w, h_w);
    Kokkos::deep_copy(d_type, h_type);

    lap("static fill+upload");
    d_quad_site = Kokkos::View<int *, DeviceType>("cac/dev:quad_site", std::max(1, nquad));
    auto h_qs = Kokkos::create_mirror_view(d_quad_site);
    for (int pq = 0; pq < nquad; pq++) h_qs(pq) = p.quad_site[pq];
    Kokkos::deep_copy(d_quad_site, h_qs);
    d_quad_out = t_out10("cac/dev:quad_out", std::max(1, nquad));
    h_quad_out = Kokkos::create_mirror_view(d_quad_out);

    lap("quad upload");
    cache_owner = p.site_owner;
    cache_poly = p.site_poly;
    cache_type = p.site_type;
    cache_scale_sum = scale_checksum();
    cache_quad_site = p.quad_site;
    static_valid = true;
  }

  void pack_nodes()
  {
    PairCAC &p = *P;
    double ****nodal_positions = p.atom->nodal_positions;
    double **x = p.atom->x;
    const int *element_type = p.atom->element_type;
    const long nrows_blocks = (long) node_rows.size();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long k = 0; k < nrows_blocks; k++) {
      const int owner = node_rows[k].first;
      const int row = row_off[k];
      if (element_type[owner] == 0) {
        for (int d = 0; d < 3; d++) h_nodes(row, d) = x[owner][d];
      } else {
        double **nodes = nodal_positions[owner][node_rows[k].second];
        for (int r = 0; r < 8; r++)
          for (int d = 0; d < 3; d++) h_nodes(row + r, d) = nodes[r][d];
      }
    }
    Kokkos::deep_copy(d_nodes, h_nodes);
  }

  // neighbor lists: the cell grid (counting sort) is built on the host from the site positions of this
  // rebuild (PairCAC::update_site_positions has been called by the rebuild step), the counting/filling
  // passes run on the device
  void build_lists()
  {
    PairCAC &p = *P;
    lap("(lists start)");
    PairCAC::SiteGrid g;
    p.build_site_grid(g);
    lap("host grid");

    auto upload_int = [&](Kokkos::View<int *, DeviceType> &d, const std::vector<int> &v) {
      if (d.extent(0) != v.size())
        d = Kokkos::View<int *, DeviceType>(Kokkos::view_alloc("cac/dev:grid", Kokkos::WithoutInitializing), v.size());
      Kokkos::deep_copy(d, HInt(v.data(), v.size()));
    };
    auto upload_dbl = [&](Kokkos::View<double *, DeviceType> &d, const std::vector<double> &v) {
      if (d.extent(0) != v.size())
        d = Kokkos::View<double *, DeviceType>(Kokkos::view_alloc("cac/dev:grid", Kokkos::WithoutInitializing), v.size());
      Kokkos::deep_copy(d, HDbl(v.data(), v.size()));
    };
    upload_dbl(d_xs, p.site_x);
    upload_dbl(d_xb, g.xb);
    upload_int(d_order, g.order);
    upload_int(d_bstart, g.bstart);
    upload_int(d_sbin, g.sbin);
    std::vector<int> isq(nsite, 0);
    for (int pq = 0; pq < nquad; pq++) isq[p.quad_site[pq]] = 1;
    upload_int(d_isquad, isq);
    lap("grid upload");

    if (d_has.extent(0) != (size_t) nsite) d_has = Kokkos::View<int *, DeviceType>("cac/dev:has", nsite);
    if (d_numneigh.extent(0) != (size_t) nsite) d_numneigh = Kokkos::View<int *, DeviceType>("cac/dev:numneigh", nsite);
    Kokkos::deep_copy(d_has, 0);
    Kokkos::deep_copy(d_numneigh, 0);
    if (d_ilist.extent(0) < (size_t) nsite) d_ilist = Kokkos::View<int *, DeviceType>("cac/dev:ilist", nsite);

    const double cut = p.site_list_cut;
    // single-precision screening margin: coordinates are rounded to 2^-24 relative
    double maxabs = 0.0;
    for (int d = 0; d < 3; d++) maxabs = std::max(maxabs, std::max(std::fabs(g.lo[d]), std::fabs(g.lo[d] + g.nb[d] * g.binsize[d])));
    const double eps = 1.2e-7 * maxabs;
    const double cutsq_f = cut * cut + 16.0 * cut * eps + 16.0 * eps * eps + 1e-9;
    if (d_xsf.extent(0) != d_xs.extent(0)) d_xsf = Kokkos::View<float *, DeviceType>("cac/dev:xsf", d_xs.extent(0));
    if (d_xbf.extent(0) != d_xb.extent(0)) d_xbf = Kokkos::View<float *, DeviceType>("cac/dev:xbf", d_xb.extent(0));
    {
      CACDev::ToFloat<DeviceType> c1;
      c1.in = d_xs;
      c1.out = d_xsf;
      Kokkos::parallel_for("CACDevF1", Kokkos::RangePolicy<DeviceType>(0, d_xs.extent(0)), c1);
      CACDev::ToFloat<DeviceType> c2;
      c2.in = d_xb;
      c2.out = d_xbf;
      Kokkos::parallel_for("CACDevF2", Kokkos::RangePolicy<DeviceType>(0, d_xb.extent(0)), c2);
    }
    auto setup = [&](auto &q) {
      q.xs = d_xs;
      q.xb = d_xb;
      q.xsf = d_xsf;
      q.xbf = d_xbf;
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
      q.mark = center_lists ? 1 : 0;
      q.cutsq = cut * cut;
      q.cutsq_f = (float) cutsq_f;
      q.neigh = d_neighbors;
      q.cols = neigh_cols;
    };
    Kokkos::RangePolicy<DeviceType> pol(0, nsite);
    if (neigh_cols < 1 || d_neighbors.extent(0) != (size_t) nsite) {
      if (neigh_cols < 64) neigh_cols = 256;
      d_neighbors = typename AT::t_neighbors_2d(Kokkos::view_alloc("cac/dev:neighbors", Kokkos::WithoutInitializing),
                                                nsite, neigh_cols);
    }
    int maxn = 1;
    for (int attempt = 0; attempt < 2; attempt++) {
      Kokkos::deep_copy(d_has, 0);
      Kokkos::deep_copy(d_numneigh, 0);
      CACDev::Query<DeviceType, 0> q0;
      setup(q0);
      Kokkos::parallel_for("CACDevQuad", pol, q0);
      lap("quad lists");
      if (center_lists) {
        CACDev::Query<DeviceType, 1> q1;
        setup(q1);
        Kokkos::parallel_for("CACDevCenters", pol, q1);
        lap("center lists");
      }
      maxn = 1;
      CACDev::MaxCount<DeviceType> mc;
      mc.cnt = d_numneigh;
      Kokkos::parallel_reduce("CACDevMax", pol, mc, Kokkos::Max<int>(maxn));
      if (maxn <= neigh_cols) break;
      // a list outgrew the padded array: enlarge it and search again
      neigh_cols = (int) (1.2 * maxn);
      d_neighbors = typename AT::t_neighbors_2d(Kokkos::view_alloc("cac/dev:neighbors", Kokkos::WithoutInitializing),
                                                nsite, neigh_cols);
    }
    max_neigh = std::max(1, maxn);
    static const bool sort_dev = getenv("CAC_SORT_LISTS_DEV") != nullptr;    // measured: no gain on the GPU
    if (sort_dev) {
      CACDev::SortRows<DeviceType> sr;
      sr.neigh = d_neighbors;
      sr.cnt = d_numneigh;
      Kokkos::parallel_for("CACDevSort", pol, sr);
      lap("sort rows");
    }

    int inum = 0;
    CACDev::Compact<DeviceType> cp;
    cp.has = d_has;
    cp.ilist = d_ilist;
    Kokkos::parallel_scan("CACDevCompact", pol, cp, inum);
    ncenters = inum;
    lap("fill+compact");
  }
};

}    // namespace LAMMPS_NS

#endif
