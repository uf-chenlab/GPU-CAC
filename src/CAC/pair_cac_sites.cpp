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

/* ----------------------------------------------------------------------
   Site table for CAC pair styles (performance layer)

   The quadrature neighbor lists built by NPairCAC store every virtual
   neighbor as (element, poly, isoparametric cell coordinate).  The same
   virtual atom appears in the lists of many quadrature points, and the
   original code re-interpolated its position (and re-read its type) for
   every quadrature point at every step.  Here all quadrature points and
   list entries are de-duplicated into unique lattice sites once per
   neighbor list build; their positions are interpolated once per step and
   the per-quadrature-point lists are gathered from the site arrays.

   Pair styles can additionally implement compute_sites(), which computes
   force densities for all quadrature points from the site arrays directly
   (used to memoize per-site many-body terms such as EAM densities).
------------------------------------------------------------------------- */

#include "pair_cac.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "neighbor.h"
#include "fmt/format.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <cmath>
#include <cstring>
#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace LAMMPS_NS;

static constexpr int CELLBITS = 9;    // bits per isoparametric cell index
static constexpr int POLYBITS = 6;

/* ----------------------------------------------------------------------
   Q8 shape functions, evaluated with the same expressions as
   PairCAC::interpolation() so cached positions are bitwise identical
------------------------------------------------------------------------- */

static inline void q8_weights(double s, double t, double w, double *N)
{
  const double sp = (1 + s) / 8;
  const double sm = (1 - s) / 8;
  const double tp = 1 + t;
  const double tm = 1 - t;
  const double wp = 1 + w;
  const double wm = 1 - w;
  N[0] = sm * tm * wm;
  N[1] = sp * tm * wm;
  N[2] = sp * tp * wm;
  N[3] = sm * tp * wm;
  N[4] = sm * tm * wp;
  N[5] = sp * tm * wp;
  N[6] = sp * tp * wp;
  N[7] = sm * tp * wp;
}

/* ----------------------------------------------------------------------
   unique key of a site: real atoms are keyed by index, virtual atoms by
   (element, poly, integer cell indices in the element)
------------------------------------------------------------------------- */

uint64_t PairCAC::site_key(int owner, int poly, const double *ucell)
{
  if (atom->element_type[owner] == 0) return ((uint64_t) owner << 1) | 1;

  const int *scale = atom->element_scale[owner];
  uint64_t key = (uint64_t) owner;
  key = (key << POLYBITS) | (uint64_t) poly;
  for (int d = 0; d < 3; d++) {
    const double mapped = 2.0 / scale[d];
    long c = lround((ucell[d] + 1.0) / mapped - 0.5);
    if (c < 0 || c >= scale[d] || c >= (1L << CELLBITS))
      error->one(FLERR, "CAC site table: cell index {} out of range for element scale {}", c,
                 scale[d]);
    key = (key << CELLBITS) | (uint64_t) c;
  }
  return key << 1;
}

int PairCAC::site_lookup(int owner, int poly, const double *ucell)
{
  // real atoms: direct index; element cells: dense per-(element,poly) cell array,
  // hashed only for elements too large for the dense array

  int *slot;
  if (atom->element_type[owner] == 0) {
    slot = &atom_site[owner];
  } else {
    const int *scale = atom->element_scale[owner];
    const size_t ep = (size_t) owner * atom->maxpoly + poly;
    long base = cell_base[ep];
    const long ncells = (long) scale[0] * scale[1] * scale[2];
    if (base == -1) {
      if (cell_site.size() + ncells < (1UL << 28)) {
        base = (long) cell_site.size();
        cell_site.resize(cell_site.size() + ncells, -1);
      } else
        base = -2;
      cell_base[ep] = base;
    }
    if (base == -2) {
      const uint64_t key = site_key(owner, poly, ucell);
      auto found = site_map.find(key);
      if (found != site_map.end()) return found->second;
      site_map.emplace(key, nsite);
      slot = nullptr;
    } else {
      long c[3];
      for (int d = 0; d < 3; d++) {
        c[d] = lround((ucell[d] + 1.0) * scale[d] / 2.0 - 0.5);
        if (c[d] < 0 || c[d] >= scale[d])
          error->one(FLERR, "CAC site table: cell index {} out of range for element scale {}", c[d],
                     scale[d]);
      }
      slot = &cell_site[base + (c[0] * scale[1] + c[1]) * scale[2] + c[2]];
    }
  }
  if (slot && *slot >= 0) return *slot;

  const int sid = nsite++;
  if (slot) *slot = sid;
  site_owner.push_back(owner);
  site_poly.push_back(poly);
  site_type.push_back(atom->node_types[owner][poly]);
  if (atom->q_flag && atom->node_charges)
    site_q.push_back(atom->node_charges[owner][poly]);
  double N[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (atom->element_type[owner] != 0) q8_weights(ucell[0], ucell[1], ucell[2], N);
  site_w.insert(site_w.end(), N, N + 8);
  return sid;
}

/* ----------------------------------------------------------------------
   build the site table from the quadrature neighbor lists
   called once after every neighbor list build
------------------------------------------------------------------------- */

void PairCAC::build_site_table()
{
  const int nlocal = atom->nlocal;
  const int *element_type = atom->element_type;
  const int *poly_count = atom->poly_count;
  int *qcounts = atom->quadrature_counts;
  int *e2q = atom->e2quad_index;
  double **qdata = atom->quadrature_point_data;
  int *icounts = atom->inner_quad_lists_counts;
  int ***iindex = atom->inner_quad_lists_index;
  double ***iucell = atom->inner_quad_lists_ucell;
  int *ocounts = atom->outer_quad_lists_counts;
  int ***oindex = atom->outer_quad_lists_index;
  double ***oucell = atom->outer_quad_lists_ucell;

  if (atom->sector_flag || atom->ghost_quad_flag)
    error->all(FLERR, "CAC site table does not support sector or ghost quadrature lists");

  if (atom->inner_only_flag >= 2 && site_enumerate) {
    build_site_table_fast();
    return;
  }

  nsite = 0;
  site_map.clear();
  const int nall = atom->nlocal + atom->nghost;
  atom_site.assign(nall, -1);
  cell_base.assign((size_t) nall * std::max(1, atom->maxpoly), -1);
  cell_site.clear();
  site_owner.clear();
  site_poly.clear();
  site_type.clear();
  site_q.clear();
  site_w.clear();

  nquadpt = 0;
  for (int i = 0; i < nlocal; i++) nquadpt += qcounts[i] * poly_count[i];

  quad_site.assign(nquadpt, -1);
  quad_owner.assign(nquadpt, -1);
  quad_qi.assign(nquadpt, -1);
  quad_shape.assign(8 * (size_t) nquadpt, 0.0);
  inner_first.assign(nquadpt + 1, 0);
  outer_first.assign(nquadpt + 1, 0);

  // count list sizes first so the CSR arrays are allocated once

  site_only_lists = (atom->inner_only_flag >= 2);
  for (int pq = 0; pq < nquadpt; pq++) {
    inner_first[pq + 1] = inner_first[pq] + (site_only_lists ? 0 : icounts[pq]);
    outer_first[pq + 1] = outer_first[pq] + (outer_neighflag && !site_only_lists ? ocounts[pq] : 0);
  }
  inner_site.resize(inner_first[nquadpt]);
  outer_site.resize(outer_first[nquadpt]);

  int qi_running = 0;
  for (int i = 0; i < nlocal; i++) {
    const int npoly = poly_count[i];
    for (int iquad = 0; iquad < qcounts[i]; iquad++) {
      const double *qd = qdata[qi_running];
      for (int p = 0; p < npoly; p++) {
        const int pq = e2q[i] + npoly * iquad + p;
        quad_owner[pq] = i;
        quad_qi[pq] = qi_running;
        quad_site[pq] = site_lookup(i, p, qd);
        if (element_type[i] != 0) q8_weights(qd[3], qd[4], qd[5], &quad_shape[8 * (size_t) pq]);

        int *dst = &inner_site[inner_first[pq]];
        if (!site_only_lists)
          for (int l = 0; l < icounts[pq]; l++)
            dst[l] = site_lookup(iindex[pq][l][0], iindex[pq][l][1], iucell[pq][l]);
        if (outer_neighflag && !site_only_lists) {
          dst = &outer_site[outer_first[pq]];
          for (int l = 0; l < ocounts[pq]; l++)
            dst[l] = site_lookup(oindex[pq][l][0], oindex[pq][l][1], oucell[pq][l]);
        }
      }
      qi_running++;
    }
  }

  // enumerate every lattice site of all local and ghost elements, so the
  // quadrature lists only need to reach rc+skin (no outer list)

  if (site_enumerate) {
    const double dummy[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < nall; i++) {
      if (element_type[i] == 0) {
        site_lookup(i, 0, dummy);
        continue;
      }
      const int *scale = atom->element_scale[i];
      for (int p = 0; p < poly_count[i]; p++)
        for (int c0 = 0; c0 < scale[0]; c0++)
          for (int c1 = 0; c1 < scale[1]; c1++)
            for (int c2 = 0; c2 < scale[2]; c2++) {
              const double u[3] = {(2.0 * c0 + 1.0) / scale[0] - 1.0, (2.0 * c1 + 1.0) / scale[1] - 1.0,
                                   (2.0 * c2 + 1.0) / scale[2] - 1.0};
              site_lookup(i, p, u);
            }
    }
  }

  site_x.resize(3 * (size_t) nsite);
  quad_force.assign(3 * (size_t) nquadpt, 0.0);
  quad_energy.assign(nquadpt, 0.0);
  quad_virial.assign(6 * (size_t) nquadpt, 0.0);

  site_table_ncalls = neighbor->ncalls;
  site_table_valid = 1;

  if (!site_table_reported) {
    bigint counts[3] = {nquadpt, nsite, (bigint) inner_site.size() + (bigint) outer_site.size()};
    bigint all[3];
    MPI_Allreduce(counts, all, 3, MPI_LMP_BIGINT, MPI_SUM, world);
    if (comm->me == 0)
      utils::logmesg(lmp, "CAC site table: {} quadrature points, {} unique sites, {} list entries\n",
                     all[0], all[1], all[2]);
    site_table_reported = 1;
  }
}

/* ----------------------------------------------------------------------
   site table of the site-only mode: the sites of an element (all its lattice sites of all polys, or
   the atom itself) are numbered directly, element after element, so the table and the map from
   quadrature points to sites are built in parallel without any lookup structure.  Site s of
   element e, poly p, cell (c0,c1,c2) is  off[e] + ((p*n0 + c0)*n1 + c1)*n2 + c2.
------------------------------------------------------------------------- */

void PairCAC::build_site_table_fast()
{
  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  const int *element_type = atom->element_type;
  const int *poly_count = atom->poly_count;
  int **element_scale = atom->element_scale;
  int *qcounts = atom->quadrature_counts;
  int *e2q = atom->e2quad_index;
  double **qdata = atom->quadrature_point_data;

  std::vector<long> off(nall + 1, 0);
  for (int e = 0; e < nall; e++) {
    long n = 1;
    if (element_type[e] != 0) n = (long) poly_count[e] * element_scale[e][0] * element_scale[e][1] * element_scale[e][2];
    off[e + 1] = off[e] + n;
  }
  if (off[nall] >= 2000000000L) error->one(FLERR, "CAC site table exceeds 2^31 sites on one rank");
  nsite = (int) off[nall];
  site_map.clear();
  atom_site.clear();
  cell_base.clear();
  cell_site.clear();

  const int qflag = atom->q_flag && atom->node_charges;
  site_owner.resize(nsite);
  site_poly.resize(nsite);
  site_type.resize(nsite);
  site_w.assign(8 * (size_t) nsite, 0.0);
  if (qflag) site_q.resize(nsite);
  else site_q.clear();

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 8)
#endif
  for (int e = 0; e < nall; e++) {
    if (element_type[e] == 0) {
      const int s = (int) off[e];
      site_owner[s] = e;
      site_poly[s] = 0;
      site_type[s] = atom->node_types[e][0];
      if (qflag) site_q[s] = atom->node_charges[e][0];
      continue;
    }
    const int n0 = element_scale[e][0], n1 = element_scale[e][1], n2 = element_scale[e][2];
    for (int p = 0; p < poly_count[e]; p++)
      for (int c0 = 0; c0 < n0; c0++) {
        const double u0 = (2.0 * c0 + 1.0) / n0 - 1.0;
        for (int c1 = 0; c1 < n1; c1++) {
          const double u1 = (2.0 * c1 + 1.0) / n1 - 1.0;
          for (int c2 = 0; c2 < n2; c2++) {
            const double u2 = (2.0 * c2 + 1.0) / n2 - 1.0;
            const int s = (int) (off[e] + ((long) (p * n0 + c0) * n1 + c1) * n2 + c2);
            site_owner[s] = e;
            site_poly[s] = p;
            site_type[s] = atom->node_types[e][p];
            if (qflag) site_q[s] = atom->node_charges[e][p];
            q8_weights(u0, u1, u2, &site_w[8 * (size_t) s]);
          }
        }
      }
  }

  nquadpt = 0;
  for (int i = 0; i < nlocal; i++) nquadpt += qcounts[i] * poly_count[i];
  quad_site.assign(nquadpt, -1);
  quad_owner.assign(nquadpt, -1);
  quad_qi.assign(nquadpt, -1);
  quad_shape.assign(8 * (size_t) nquadpt, 0.0);
  inner_first.assign(nquadpt + 1, 0);
  outer_first.assign(nquadpt + 1, 0);
  inner_site.clear();
  outer_site.clear();
  site_only_lists = 1;

  std::vector<int> qbase(nlocal + 1, 0);
  for (int i = 0; i < nlocal; i++) qbase[i + 1] = qbase[i] + qcounts[i];

  int bad = 0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 8) reduction(| : bad)
#endif
  for (int i = 0; i < nlocal; i++) {
    const int npoly = poly_count[i];
    const bool is_atom = element_type[i] == 0;
    for (int iquad = 0; iquad < qcounts[i]; iquad++) {
      const int qi_running = qbase[i] + iquad;
      const double *qd = qdata[qi_running];
      int c[3] = {0, 0, 0};
      if (!is_atom) {
        for (int d = 0; d < 3; d++) {
          c[d] = (int) lround((qd[d] + 1.0) * element_scale[i][d] / 2.0 - 0.5);
          if (c[d] < 0 || c[d] >= element_scale[i][d]) bad = 1;
        }
        if (bad) continue;
      }
      for (int p = 0; p < npoly; p++) {
        const int pq = e2q[i] + npoly * iquad + p;
        quad_owner[pq] = i;
        quad_qi[pq] = qi_running;
        if (is_atom) {
          quad_site[pq] = (int) off[i];
        } else {
          const int n1 = element_scale[i][1], n2 = element_scale[i][2], n0 = element_scale[i][0];
          quad_site[pq] = (int) (off[i] + ((long) (p * n0 + c[0]) * n1 + c[1]) * n2 + c[2]);
          q8_weights(qd[3], qd[4], qd[5], &quad_shape[8 * (size_t) pq]);
        }
      }
    }
  }
  if (bad) error->one(FLERR, "CAC site table: quadrature point outside its element");

  site_x.resize(3 * (size_t) nsite);
  quad_force.assign(3 * (size_t) nquadpt, 0.0);
  quad_energy.assign(nquadpt, 0.0);
  quad_virial.assign(6 * (size_t) nquadpt, 0.0);

  site_table_ncalls = neighbor->ncalls;
  site_table_valid = 1;

  if (!site_table_reported) {
    bigint counts[3] = {nquadpt, nsite, 0};
    bigint all[3];
    MPI_Allreduce(counts, all, 3, MPI_LMP_BIGINT, MPI_SUM, world);
    if (comm->me == 0)
      utils::logmesg(lmp, "CAC site table: {} quadrature points, {} unique sites, {} list entries\n",
                     all[0], all[1], all[2]);
    site_table_reported = 1;
  }
}

/* ----------------------------------------------------------------------
   interpolate the current position of every site once
------------------------------------------------------------------------- */

void PairCAC::update_site_positions()
{
  double ****nodal_positions = atom->nodal_positions;
  double **x = atom->x;
  const int *element_type = atom->element_type;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int s = 0; s < nsite; s++) {
    const int owner = site_owner[s];
    double *xs = &site_x[3 * (size_t) s];
    if (element_type[owner] == 0) {
      xs[0] = x[owner][0];
      xs[1] = x[owner][1];
      xs[2] = x[owner][2];
    } else {
      double **nodes = nodal_positions[owner][site_poly[s]];
      const double *N = &site_w[8 * (size_t) s];
      double px = 0, py = 0, pz = 0;
      for (int r = 0; r < 8; r++) {
        px += nodes[r][0] * N[r];
        py += nodes[r][1] * N[r];
        pz += nodes[r][2] * N[r];
      }
      xs[0] = px;
      xs[1] = py;
      xs[2] = pz;
    }
  }
}

/* ----------------------------------------------------------------------
   neighbor lists of sites for many-body styles (cell list over all sites)
   only sites that appear in an inner list or are quadrature points get a
   list; all sites are candidates
------------------------------------------------------------------------- */

void PairCAC::build_site_grid(SiteGrid &g)
{
  const double cut = site_list_cut;
  double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
  {
    double lx = 1e300, ly = 1e300, lz = 1e300, hx = -1e300, hy = -1e300, hz = -1e300;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(min : lx, ly, lz) reduction(max : hx, hy, hz)
#endif
    for (int s = 0; s < nsite; s++) {
      const double *q = &site_x[3 * (size_t) s];
      lx = std::min(lx, q[0]); hx = std::max(hx, q[0]);
      ly = std::min(ly, q[1]); hy = std::max(hy, q[1]);
      lz = std::min(lz, q[2]); hz = std::max(hz, q[2]);
    }
    lo[0] = lx; lo[1] = ly; lo[2] = lz; hi[0] = hx; hi[1] = hy; hi[2] = hz;
  }
  const double target = 0.5 * cut;
  for (int d = 0; d < 3; d++) {
    g.lo[d] = lo[d];
    g.nb[d] = std::max(1, (int) ((hi[d] - lo[d]) / target));
    if (g.nb[d] > 1024) g.nb[d] = 1024;
    g.binsize[d] = std::max((hi[d] - lo[d]) / g.nb[d], 1e-10);
    g.reach[d] = (int) std::ceil(cut / g.binsize[d]);
  }
  const size_t nbins = (size_t) g.nb[0] * g.nb[1] * g.nb[2];
  g.sbin.resize(nsite);
  g.bstart.assign(nbins + 1, 0);
  g.order.resize(nsite);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int s = 0; s < nsite; s++) {
    int c[3];
    for (int d = 0; d < 3; d++) {
      c[d] = (int) ((site_x[3 * (size_t) s + d] - g.lo[d]) / g.binsize[d]);
      c[d] = std::min(std::max(c[d], 0), g.nb[d] - 1);
    }
    g.sbin[s] = (int) (((size_t) c[2] * g.nb[1] + c[1]) * g.nb[0] + c[0]);
  }
  for (int s = 0; s < nsite; s++) g.bstart[(size_t) g.sbin[s] + 1]++;
  for (size_t b = 0; b < nbins; b++) g.bstart[b + 1] += g.bstart[b];
  {
    std::vector<int> fill(g.bstart.begin(), g.bstart.end() - 1);
    for (int s = 0; s < nsite; s++) g.order[fill[g.sbin[s]]++] = s;
  }
  g.xb.resize(3 * (size_t) nsite);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int k = 0; k < nsite; k++)
    for (int d = 0; d < 3; d++) g.xb[3 * (size_t) k + d] = site_x[3 * (size_t) g.order[k] + d];
}

void PairCAC::build_site_lists()
{
  double w0 = tm_on ? MPI_Wtime() : 0.0;
  auto lap = [&](const char *what) {
    if (tm_on && getenv("CAC_TIMING_DETAIL")) {
      const double w1 = MPI_Wtime();
      fprintf(stderr, "  site-lists %s %.2f ms\n", what, 1e3 * (w1 - w0));
      w0 = w1;
    }
  };
  const double cut = site_list_cut;
  const double cutsq = cut * cut;

  SiteGrid grid;
  build_site_grid(grid);
  const int *nb = grid.nb, *reach = grid.reach;
  const std::vector<int> &sbin = grid.sbin, &bstart = grid.bstart, &order = grid.order;
  const std::vector<double> &xb = grid.xb;
  lap("sort");

  // all sites within cut of s, in cell order (self excluded)

  auto query = [&](int s, std::vector<int> &out) {
    const double x0 = site_x[3 * (size_t) s], y0 = site_x[3 * (size_t) s + 1],
                 z0 = site_x[3 * (size_t) s + 2];
    const int b = sbin[s];
    const int cx = b % nb[0], cy = (b / nb[0]) % nb[1], cz = b / (nb[0] * nb[1]);
    const int ix0 = std::max(0, cx - reach[0]), ix1 = std::min(nb[0] - 1, cx + reach[0]);
    for (int iz = std::max(0, cz - reach[2]); iz <= std::min(nb[2] - 1, cz + reach[2]); iz++)
      for (int iy = std::max(0, cy - reach[1]); iy <= std::min(nb[1] - 1, cy + reach[1]); iy++) {
        const size_t row = ((size_t) iz * nb[1] + iy) * nb[0];
        const int kbeg = bstart[row + ix0], kend = bstart[row + ix1 + 1];
        for (int k = kbeg; k < kend; k++) {
          const double dx = x0 - xb[3 * (size_t) k];
          const double dy = y0 - xb[3 * (size_t) k + 1];
          const double dz = z0 - xb[3 * (size_t) k + 2];
          if (dx * dx + dy * dy + dz * dz < cutsq) {
            const int j = order[k];
            if (j != s) out.push_back(j);
          }
        }
      }
  };

  // lists of all sites with want[s] != 0 as CSR (first, list); built by contiguous chunks of
  // sites on all threads and concatenated in site order, so the result is thread-count independent

  auto build_lists = [&](const std::vector<char> &want, std::vector<int> &first,
                         std::vector<int> &list) {
    first.assign(nsite + 1, 0);
    int nthr = 1;
#if defined(_OPENMP)
    nthr = std::max(1, omp_get_max_threads());
#endif
    std::vector<std::vector<int>> tlist(nthr);
#if defined(_OPENMP)
#pragma omp parallel num_threads(nthr)
#endif
    {
      int t = 0;
#if defined(_OPENMP)
      t = omp_get_thread_num();
#endif
      const int s0 = (int) ((long) nsite * t / nthr), s1 = (int) ((long) nsite * (t + 1) / nthr);
      std::vector<int> &out = tlist[t];
      for (int s = s0; s < s1; s++) {
        if (want[s]) query(s, out);
        first[s + 1] = (int) out.size();    // local running count, offset below
      }
    }
    // offsets: thread t starts at the total of threads < t
    size_t total = 0;
    std::vector<size_t> toff(nthr + 1, 0);
    for (int t = 0; t < nthr; t++) {
      toff[t + 1] = toff[t] + tlist[t].size();
    }
    total = toff[nthr];
    if (total >= (size_t) 2000000000) error->one(FLERR, "CAC site lists exceed 2^31 entries");
    for (int t = 0; t < nthr; t++) {
      const int s0 = (int) ((long) nsite * t / nthr), s1 = (int) ((long) nsite * (t + 1) / nthr);
      for (int s = s0; s < s1; s++) first[s + 1] += (int) toff[t];
    }
    list.resize(total);
    for (int t = 0; t < nthr; t++) std::copy(tlist[t].begin(), tlist[t].end(), list.begin() + toff[t]);
  };

  site_has_list.assign(nsite, 0);
  slist_first.assign(nsite + 1, 0);

  if (site_only_lists) {
    // P1: count the neighbours of every quadrature site and mark them as centers
    // P2: count the neighbours of the other centers
    // prefix sum -> CSR offsets; P3: fill the final array in place (all passes threaded)
    std::vector<char> is_quad(nsite, 0);
    for (int pq = 0; pq < nquadpt; pq++) is_quad[quad_site[pq]] = 1;
    std::vector<int> cnt(nsite, 0);

    // count (and optionally mark) the sites within cut of s, self excluded
    auto count_query = [&](int s, bool mark) {
      const double x0 = site_x[3 * (size_t) s], y0 = site_x[3 * (size_t) s + 1],
                   z0 = site_x[3 * (size_t) s + 2];
      const int b = sbin[s];
      const int cx = b % nb[0], cy = (b / nb[0]) % nb[1], cz = b / (nb[0] * nb[1]);
      const int ix0 = std::max(0, cx - reach[0]), ix1 = std::min(nb[0] - 1, cx + reach[0]);
      int n = 0;
      for (int iz = std::max(0, cz - reach[2]); iz <= std::min(nb[2] - 1, cz + reach[2]); iz++)
        for (int iy = std::max(0, cy - reach[1]); iy <= std::min(nb[1] - 1, cy + reach[1]); iy++) {
          const size_t row = ((size_t) iz * nb[1] + iy) * nb[0];
          for (int k = bstart[row + ix0]; k < bstart[row + ix1 + 1]; k++) {
            const double dx = x0 - xb[3 * (size_t) k];
            const double dy = y0 - xb[3 * (size_t) k + 1];
            const double dz = z0 - xb[3 * (size_t) k + 2];
            if (dx * dx + dy * dy + dz * dz < cutsq) {
              const int j = order[k];
              if (j == s) continue;
              n++;
              if (mark) {
#if defined(_OPENMP)
#pragma omp atomic write
#endif
                site_has_list[j] = 1;
              }
            }
          }
        }
      return n;
    };

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int s = 0; s < nsite; s++)
      if (is_quad[s]) {
        cnt[s] = count_query(s, site_center_lists != 0);
        site_has_list[s] = 1;
      }
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int s = 0; s < nsite; s++)
      if (!is_quad[s] && site_has_list[s]) cnt[s] = count_query(s, false);

    slist_first[0] = 0;
    for (int s = 0; s < nsite; s++) slist_first[s + 1] = slist_first[s] + cnt[s];
    if ((size_t) slist_first[nsite] >= 2000000000UL) error->one(FLERR, "CAC site lists exceed 2^31 entries");
    slist.resize(slist_first[nsite]);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int s = 0; s < nsite; s++) {
      if (cnt[s] == 0) continue;
      const double x0 = site_x[3 * (size_t) s], y0 = site_x[3 * (size_t) s + 1],
                   z0 = site_x[3 * (size_t) s + 2];
      const int b = sbin[s];
      const int cx = b % nb[0], cy = (b / nb[0]) % nb[1], cz = b / (nb[0] * nb[1]);
      const int ix0 = std::max(0, cx - reach[0]), ix1 = std::min(nb[0] - 1, cx + reach[0]);
      int *out = slist.data() + slist_first[s];
      for (int iz = std::max(0, cz - reach[2]); iz <= std::min(nb[2] - 1, cz + reach[2]); iz++)
        for (int iy = std::max(0, cy - reach[1]); iy <= std::min(nb[1] - 1, cy + reach[1]); iy++) {
          const size_t row = ((size_t) iz * nb[1] + iy) * nb[0];
          for (int k = bstart[row + ix0]; k < bstart[row + ix1 + 1]; k++) {
            const double dx = x0 - xb[3 * (size_t) k];
            const double dy = y0 - xb[3 * (size_t) k + 1];
            const double dz = z0 - xb[3 * (size_t) k + 2];
            if (dx * dx + dy * dy + dz * dz < cutsq) {
              const int j = order[k];
              if (j != s) *out++ = j;
            }
          }
        }
    }
    // neighbours in ascending site order: streams through the (element-ordered) site arrays
    const char *sort_env = getenv("CAC_SORT_LISTS");    // 1 = always, 0 = never, unset = per style default
    const bool sort_lists = sort_env ? sort_env[0] != '0' : site_sort_lists != 0;
    if (sort_lists) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 256)
#endif
      for (int s = 0; s < nsite; s++)
        if (cnt[s] > 1) std::sort(slist.begin() + slist_first[s], slist.begin() + slist_first[s + 1]);
    }
    lap("count+fill");
    return;
  }

  // lists of the original quadrature scan: quadrature sites reuse their inner list

  for (int pq = 0; pq < nquadpt; pq++) {
    site_has_list[quad_site[pq]] = 1;
    for (int k = inner_first[pq]; k < inner_first[pq + 1]; k++) site_has_list[inner_site[k]] = 1;
  }
  std::vector<int> quad_of_site(nsite, -1);
  for (int pq = 0; pq < nquadpt; pq++) quad_of_site[quad_site[pq]] = pq;
  std::vector<char> want(nsite, 0);
  for (int s = 0; s < nsite; s++) want[s] = site_has_list[s] && quad_of_site[s] < 0;
  std::vector<int> cfirst, clist;
  build_lists(want, cfirst, clist);
  slist.clear();
  slist.reserve(clist.size() + inner_site.size());
  for (int s = 0; s < nsite; s++) {
    if (site_has_list[s] && quad_of_site[s] >= 0) {
      const int pq = quad_of_site[s];
      slist.insert(slist.end(), inner_site.begin() + inner_first[pq],
                   inner_site.begin() + inner_first[pq + 1]);
    } else if (site_has_list[s]) {
      slist.insert(slist.end(), clist.begin() + cfirst[s], clist.begin() + cfirst[s + 1]);
    }
    slist_first[s + 1] = (int) slist.size();
  }
}

/* ----------------------------------------------------------------------
   centers (sites with a neighbor list) and thread-private accumulation buffers for
   center-based kernels; scatter_finish() sums them onto the quadrature points
------------------------------------------------------------------------- */

int PairCAC::scatter_begin(int need_e, int need_v)
{
  int nthr = 1;
#if defined(_OPENMP)
  nthr = std::max(1, omp_get_max_threads());
#endif
  if (sc_centers_ncalls != site_table_ncalls || sc_centers_nsite != nsite) {
    sc_centers.clear();
    for (int s = 0; s < nsite; s++)
      if (site_has_list[s] && slist_first[s + 1] > slist_first[s]) sc_centers.push_back(s);
    sc_centers_ncalls = site_table_ncalls;
    sc_centers_nsite = nsite;
  }
  if ((int) sc_f.size() != nthr) {
    sc_f.assign(nthr, std::vector<double>());
    sc_e.assign(nthr, std::vector<double>());
    sc_v.assign(nthr, std::vector<double>());
  }
#if defined(_OPENMP)
#pragma omp parallel num_threads(nthr)
#endif
  {
    int t = 0;
#if defined(_OPENMP)
    t = omp_get_thread_num();
#endif
    sc_f[t].assign(3 * (size_t) nsite, 0.0);
    if (need_e) sc_e[t].assign(nsite, 0.0);
    if (need_v) sc_v[t].assign(6 * (size_t) nsite, 0.0);
  }
  return nthr;
}

void PairCAC::scatter_finish(int need_e, int need_v)
{
  const int nthr = (int) sc_f.size();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(nthr)
#endif
  for (int pq = 0; pq < nquadpt; pq++) {
    const size_t s = (size_t) quad_site[pq];
    double f0 = 0, f1 = 0, f2 = 0, e = 0, v[6] = {0, 0, 0, 0, 0, 0};
    for (int t = 0; t < nthr; t++) {
      f0 += sc_f[t][3 * s];
      f1 += sc_f[t][3 * s + 1];
      f2 += sc_f[t][3 * s + 2];
      if (need_e) e += sc_e[t][s];
      if (need_v)
        for (int d = 0; d < 6; d++) v[d] += sc_v[t][6 * s + d];
    }
    quad_force[3 * (size_t) pq] = f0;
    quad_force[3 * (size_t) pq + 1] = f1;
    quad_force[3 * (size_t) pq + 2] = f2;
    quad_energy[pq] = e;
    if (need_v)
      for (int d = 0; d < 6; d++) quad_virial[6 * (size_t) pq + d] = v[d];
  }
}

/* ----------------------------------------------------------------------
   gather replacements for interpolation() and init_quad_arrays()
------------------------------------------------------------------------- */

void PairCAC::interpolation_from_sites(int /*iii*/)
{
  const double *xq = &site_x[3 * (size_t) quad_site[pqi]];
  current_position[0] = xq[0];
  current_position[1] = xq[1];
  current_position[2] = xq[2];

  const int *src = &inner_site[inner_first[pqi]];
  const int n = inner_first[pqi + 1] - inner_first[pqi];
  for (int l = 0; l < n; l++) {
    const double *xs = &site_x[3 * (size_t) src[l]];
    inner_neighbor_coords[l][0] = xs[0];
    inner_neighbor_coords[l][1] = xs[1];
    inner_neighbor_coords[l][2] = xs[2];
  }
  if (outer_neighflag) {
    src = &outer_site[outer_first[pqi]];
    const int no = outer_first[pqi + 1] - outer_first[pqi];
    for (int l = 0; l < no; l++) {
      const double *xs = &site_x[3 * (size_t) src[l]];
      outer_neighbor_coords[l][0] = xs[0];
      outer_neighbor_coords[l][1] = xs[1];
      outer_neighbor_coords[l][2] = xs[2];
    }
  }
}

void PairCAC::init_quad_arrays_from_sites()
{
  const int qflag = atom->q_flag && !site_q.empty();
  const int *src = &inner_site[inner_first[pqi]];
  const int n = inner_first[pqi + 1] - inner_first[pqi];
  for (int l = 0; l < n; l++) {
    inner_neighbor_types[l] = site_type[src[l]];
    if (qflag) inner_neighbor_charges[l] = site_q[src[l]];
  }
  if (outer_neighflag) {
    src = &outer_site[outer_first[pqi]];
    const int no = outer_first[pqi + 1] - outer_first[pqi];
    for (int l = 0; l < no; l++) {
      outer_neighbor_types[l] = site_type[src[l]];
      if (qflag) outer_neighbor_charges[l] = site_q[src[l]];
    }
  }
}

/* ----------------------------------------------------------------------
   inverse of the Q8 consistent mass matrix in isoparametric space
   (2-point Gauss rule, weights 1); identical for every Q8 element
------------------------------------------------------------------------- */

void PairCAC::compute_minv()
{
  const double g = 0.5773502691896258;
  const double abs_[2] = {-g, g};
  double M[8][16];
  for (int a = 0; a < 8; a++)
    for (int b = 0; b < 16; b++) M[a][b] = 0.0;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++) {
        double N[8];
        q8_weights(abs_[i], abs_[j], abs_[k], N);
        for (int a = 0; a < 8; a++)
          for (int b = 0; b < 8; b++) M[a][b] += N[a] * N[b];
      }
  for (int a = 0; a < 8; a++) M[a][8 + a] = 1.0;

  // Gauss-Jordan with partial pivoting

  for (int c = 0; c < 8; c++) {
    int piv = c;
    for (int r = c + 1; r < 8; r++)
      if (fabs(M[r][c]) > fabs(M[piv][c])) piv = r;
    if (fabs(M[piv][c]) < 1e-14) error->all(FLERR, "CAC Q8 mass matrix is singular");
    if (piv != c)
      for (int b = 0; b < 16; b++) std::swap(M[c][b], M[piv][b]);
    const double inv = 1.0 / M[c][c];
    for (int b = 0; b < 16; b++) M[c][b] *= inv;
    for (int r = 0; r < 8; r++) {
      if (r == c) continue;
      const double f = M[r][c];
      if (f != 0.0)
        for (int b = 0; b < 16; b++) M[r][b] -= f * M[c][b];
    }
  }
  for (int a = 0; a < 8; a++)
    for (int b = 0; b < 8; b++) Minv_q8[a][b] = M[a][8 + b];
  minv_ready = 1;
}

/* ----------------------------------------------------------------------
   site-kernel compute: compute_sites() fills quadrature force densities,
   which are then projected onto the nodes with the constant inverse mass
   matrix; energy and virial tallies follow PairCAC::compute()
------------------------------------------------------------------------- */

void PairCAC::compute_site_path(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  quad_eflag = eflag;
  if (!minv_ready) compute_minv();

  double t0 = tm_on ? MPI_Wtime() : 0.0, t1;
  if (!site_table_valid || site_table_ncalls != neighbor->ncalls) {
    build_site_table();
    if (tm_on) { t1 = MPI_Wtime(); tm[0] += t1 - t0; t0 = t1; }
    update_site_positions();
    if (tm_on) { t1 = MPI_Wtime(); tm[1] += t1 - t0; t0 = t1; }
    if (site_need_lists && build_lists_on_host()) build_site_lists();
    if (tm_on) { t1 = MPI_Wtime(); tm[2] += t1 - t0; }
  } else if (site_host_positions) {
    update_site_positions();
    if (tm_on) tm[5] += MPI_Wtime() - t0;
  }

  std::fill(quad_force.begin(), quad_force.end(), 0.0);
  std::fill(quad_energy.begin(), quad_energy.end(), 0.0);
  if (atom->CAC_virial) std::fill(quad_virial.begin(), quad_virial.end(), 0.0);

  t0 = tm_on ? MPI_Wtime() : 0.0;
  compute_sites(eflag, vflag);
  if (tm_on) { tm[3] += MPI_Wtime() - t0; t0 = MPI_Wtime(); }

  // debugging/validation aid: CAC_DUMP_QUAD=<file> writes position, force density and energy
  // of every quadrature point (per MPI rank) at every call

  if (const char *fname = getenv("CAC_DUMP_QUAD")) {
    if (!site_host_positions) update_site_positions();
    std::string fn = fmt::format("{}.{}", fname, comm->me);
    FILE *fp = fopen(fn.c_str(), "w");
    if (fp) {
      for (int pq = 0; pq < nquadpt; pq++) {
        const double *xq = &site_x[3 * (size_t) quad_site[pq]];
        const double *fq = &quad_force[3 * (size_t) pq];
        fprintf(fp, "%d %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g", pq, site_type[quad_site[pq]],
                xq[0], xq[1], xq[2], fq[0], fq[1], fq[2], quad_energy[pq]);
        for (int d = 0; d < 6; d++) fprintf(fp, " %.17g", atom->CAC_virial ? quad_virial[6 * (size_t) pq + d] : 0.0);
        fprintf(fp, "\n");
      }
      fclose(fp);
    }
  }

  double ****nodal_forces = atom->nodal_forces;
  double ****nodal_virial = atom->nodal_virial;
  const int *element_type = atom->element_type;
  const int *poly_count = atom->poly_count;
  int **element_scale = atom->element_scale;
  const int *nodes_count_list = atom->nodes_per_element_list;
  const int *qcounts = atom->quadrature_counts;
  const int *e2q = atom->e2quad_index;
  double **qdata = atom->quadrature_point_data;
  const int nlocal = atom->nlocal;
  const int cac_virial = atom->CAC_virial;

  std::vector<double> eng_el(nlocal, 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < nlocal; i++) {
    const int npoly = poly_count[i];
    double element_energy = 0.0;
    if (element_type[i] == 0) {
      const int pq = e2q[i];
      for (int d = 0; d < 3; d++) nodal_forces[i][0][0][d] += quad_force[3 * (size_t) pq + d];
      if (cac_virial)
        for (int d = 0; d < 6; d++) nodal_virial[i][0][0][d] = quad_virial[6 * (size_t) pq + d];
      element_energy = qdata[quad_qi[pq]][6] * quad_energy[pq];
    } else {
      const double iso_volume = (2.0 / element_scale[i][0]) * (2.0 / element_scale[i][1]) *
          (2.0 / element_scale[i][2]);
      for (int p = 0; p < npoly; p++) {
        double col[8][3], vcol[8][6];
        for (int n = 0; n < 8; n++) {
          col[n][0] = col[n][1] = col[n][2] = 0.0;
          for (int d = 0; d < 6; d++) vcol[n][d] = 0.0;
        }
        for (int iquad = 0; iquad < qcounts[i]; iquad++) {
          const int pq = e2q[i] + npoly * iquad + p;
          const double coeff = qdata[quad_qi[pq]][6];
          const double *N = &quad_shape[8 * (size_t) pq];
          const double *fq = &quad_force[3 * (size_t) pq];
          for (int n = 0; n < 8; n++) {
            const double cw = coeff * N[n];
            col[n][0] += cw * fq[0];
            col[n][1] += cw * fq[1];
            col[n][2] += cw * fq[2];
          }
          if (cac_virial) {
            const double *vq = &quad_virial[6 * (size_t) pq];
            for (int n = 0; n < 8; n++)
              for (int d = 0; d < 6; d++) vcol[n][d] += coeff * N[n] * vq[d];
          }
          element_energy += coeff * quad_energy[pq] / iso_volume;
        }
        double **fnode = nodal_forces[i][p];
        for (int n = 0; n < 8; n++) {
          double fx = 0, fy = 0, fz = 0;
          for (int m = 0; m < 8; m++) {
            fx += Minv_q8[n][m] * col[m][0];
            fy += Minv_q8[n][m] * col[m][1];
            fz += Minv_q8[n][m] * col[m][2];
          }
          fnode[n][0] += fx;
          fnode[n][1] += fy;
          fnode[n][2] += fz;
        }
        if (cac_virial)
          for (int n = 0; n < 8; n++)
            for (int d = 0; d < 6; d++) {
              double v = 0;
              for (int m = 0; m < 8; m++) v += Minv_q8[n][m] * vcol[m][d];
              nodal_virial[i][p][n][d] = v;
            }
      }
    }

    eng_el[i] = element_energy;
  }

  for (int i = 0; i < nlocal; i++) {
    const int npoly = poly_count[i];
    if (evflag) ev_tally_full(i, 2 * eng_el[i], 0.0, 0.0, 0.0, 0.0, 0.0);

    if (vflag) {
      const int nodes_per_element = nodes_count_list[element_type[i]];
      double v[6];
      for (int p = 0; p < npoly; p++)
        for (int n = 0; n < nodes_per_element; n++) {
          for (int d = 0; d < 6; d++) v[d] = nodal_virial[i][p][n][d];
          if (element_type[i] != 0)
            for (int d = 0; d < 6; d++)
              v[d] *= (element_scale[i][0] * element_scale[i][1] * element_scale[i][2]) /
                  nodes_count_list[element_type[i]];    // integer division as in PairCAC::compute()
          v_tally_tensor(0, 0, 1, 0, v[0], v[1], v[2], v[3], v[4], v[5]);
        }
    }
  }
  if (tm_on) tm[4] += MPI_Wtime() - t0;
}
