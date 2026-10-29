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

#include "pair_cac_lj_kokkos.h"

#include "atom.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;

namespace {

// force density, energy and virial of quadrature point q: same arithmetic as PairCACLJ::compute_sites
template <class DeviceType> struct CACLJKernel {
  Kokkos::View<const double *[3], Kokkos::LayoutRight, DeviceType> x;
  Kokkos::View<const int *, DeviceType> type, quad_site, numneigh;
  typename ArrayTypes<DeviceType>::t_neighbors_2d neigh;
  Kokkos::View<const double *, DeviceType> lj1, lj2, lj3, lj4, offset;
  Kokkos::View<double *[10], Kokkos::LayoutRight, DeviceType> out;
  int nt1, need_e, need_v;
  double cutglobsq, factor_lj;
  KOKKOS_INLINE_FUNCTION void operator()(const int q) const
  {
    const int s = quad_site(q);
    const double xi0 = x(s, 0), xi1 = x(s, 1), xi2 = x(s, 2);
    const int itype = type(s);
    double fx = 0, fy = 0, fz = 0, eq = 0;
    double vd[6] = {0, 0, 0, 0, 0, 0};
    const int nn = numneigh(s);
    for (int k = 0; k < nn; k++) {
      const int j = neigh(s, k);
      const double delx = xi0 - x(j, 0);
      const double dely = xi1 - x(j, 1);
      const double delz = xi2 - x(j, 2);
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cutglobsq) continue;
      const int jtype = type(j);
      const int ij = itype * nt1 + jtype;
      const double r2inv = 1.0 / rsq;
      const double r6inv = r2inv * r2inv * r2inv;
      const double forcelj = r6inv * (lj1(ij) * r6inv - lj2(ij));
      const double fpair = factor_lj * forcelj * r2inv;
      fx += delx * fpair;
      fy += dely * fpair;
      fz += delz * fpair;
      if (need_v) {
        vd[0] += 0.5 * delx * delx * fpair;
        vd[1] += 0.5 * dely * dely * fpair;
        vd[2] += 0.5 * delz * delz * fpair;
        vd[3] += 0.5 * delx * dely * fpair;
        vd[4] += 0.5 * delx * delz * fpair;
        vd[5] += 0.5 * dely * delz * fpair;
      }
      if (need_e) eq += r6inv * (lj3(ij) * r6inv - lj4(ij)) / 2 - offset(ij) / 2;
    }
    out(q, 0) = fx;
    out(q, 1) = fy;
    out(q, 2) = fz;
    out(q, 3) = eq;
    for (int d = 0; d < 6; d++) out(q, 4 + d) = vd[d];
  }
};

}    // namespace

template <class DeviceType> PairCACLJKokkos<DeviceType>::PairCACLJKokkos(LAMMPS *lmp) : PairCACLJ(lmp)
{
  tables_ready = false;
}

template <class DeviceType> PairCACLJKokkos<DeviceType>::~PairCACLJKokkos() = default;

template <class DeviceType> void PairCACLJKokkos<DeviceType>::init_style()
{
  PairCACLJ::init_style();
  if (site_legacy) error->all(FLERR, "pair_style cac/lj/kk needs the site-only lists: unset CAC_LEGACY_LISTS");
  if (lmp->kokkos) error->all(FLERR, "pair_style cac/lj/kk offloads the CAC force kernel itself; run without '-k on'");
  dev.init(this, lmp, false);
  tables_ready = false;
}

template <class DeviceType> void PairCACLJKokkos<DeviceType>::compute_sites(int, int)
{
  const int nt = atom->ntypes;
  const int nt1 = nt + 1;
  if (!tables_ready) {
    auto mk = [&](Kokkos::View<double *, DeviceType> &d, double **a) {
      d = Kokkos::View<double *, DeviceType>("cac/lj/kk:table", (size_t) nt1 * nt1);
      auto h = Kokkos::create_mirror_view(d);
      for (int i = 0; i < nt1; i++)
        for (int j = 0; j < nt1; j++) h((size_t) i * nt1 + j) = (i >= 1 && j >= 1) ? a[i][j] : 0.0;
      Kokkos::deep_copy(d, h);
    };
    mk(d_lj1, lj1);
    mk(d_lj2, lj2);
    mk(d_lj3, lj3);
    mk(d_lj4, lj4);
    mk(d_offset, offset);
    tables_ready = true;
  }

  dev.prepare();
  const double tk = dev.tick();

  CACLJKernel<DeviceType> k;
  k.x = dev.d_x;
  k.type = dev.d_type;
  k.quad_site = dev.d_quad_site;
  k.numneigh = dev.d_numneigh;
  k.neigh = dev.d_neighbors;
  k.lj1 = d_lj1;
  k.lj2 = d_lj2;
  k.lj3 = d_lj3;
  k.lj4 = d_lj4;
  k.offset = d_offset;
  k.out = dev.d_quad_out;
  k.nt1 = nt1;
  k.need_e = quad_eflag ? 1 : 0;
  k.need_v = atom->CAC_virial ? 1 : 0;
  k.cutglobsq = cut_global_s * cut_global_s;
  k.factor_lj = force->special_lj[0];
  Kokkos::parallel_for("CACLJKernel", Kokkos::RangePolicy<DeviceType>(0, dev.nquad), k);
  dev.tock(8, tk);

  dev.download();
}

namespace LAMMPS_NS {
template class PairCACLJKokkos<LMPDeviceType>;
#ifdef KOKKOS_ENABLE_CUDA
template class PairCACLJKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
