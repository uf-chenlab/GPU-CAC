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

#include "pair_cac_eam_kokkos.h"

#include "atom.h"
#include "error.h"

using namespace LAMMPS_NS;

namespace {

// embedding density and derivative of every center: same arithmetic as PairCACEAM::compute_sites
template <class DeviceType> struct CACEAMDensity {
  Kokkos::View<const double *[3], Kokkos::LayoutRight, DeviceType> x;
  Kokkos::View<const int *, DeviceType> type, numneigh, ilist, type2rhor, type2frho;
  typename ArrayTypes<DeviceType>::t_neighbors_2d neigh;
  Kokkos::View<const double *[7], Kokkos::LayoutRight, DeviceType> rhor, frho;
  Kokkos::View<double *, DeviceType> rho, fp;
  int nt1, nr, nrho;
  double cutforcesq, rdr, rdrho;
  KOKKOS_INLINE_FUNCTION void operator()(const int ii) const
  {
    const int i = ilist(ii);
    const double xi0 = x(i, 0), xi1 = x(i, 1), xi2 = x(i, 2);
    const int itype = type(i);
    double rhoi = 0.0;
    const int nn = numneigh(i);
    for (int l = 0; l < nn; l++) {
      const int j = neigh(i, l);
      const double dx = xi0 - x(j, 0);
      const double dy = xi1 - x(j, 1);
      const double dz = xi2 - x(j, 2);
      const double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq >= cutforcesq) continue;
      double p = sqrt(rsq) * rdr + 1.0;
      int m = static_cast<int>(p);
      m = m < nr - 1 ? m : nr - 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;
      const int row = type2rhor(type(j) * nt1 + itype) * (nr + 1) + m;
      rhoi += ((rhor(row, 3) * p + rhor(row, 4)) * p + rhor(row, 5)) * p + rhor(row, 6);
    }
    rho(i) = rhoi;
    double p = rhoi * rdrho + 1.0;
    int m = static_cast<int>(p);
    m = m < nrho - 1 ? m : nrho - 1;
    m = m > 1 ? m : 1;
    p -= m;
    p = p < 1.0 ? p : 1.0;
    const int row = type2frho(itype) * (nrho + 1) + m;
    fp(i) = (frho(row, 0) * p + frho(row, 1)) * p + frho(row, 2);
  }
};

// force density, energy and virial of quadrature point q
template <class DeviceType> struct CACEAMForce {
  Kokkos::View<const double *[3], Kokkos::LayoutRight, DeviceType> x;
  Kokkos::View<const int *, DeviceType> type, quad_site, numneigh, type2rhor, type2frho, type2z2r;
  typename ArrayTypes<DeviceType>::t_neighbors_2d neigh;
  Kokkos::View<const double *[7], Kokkos::LayoutRight, DeviceType> rhor, frho, z2r;
  Kokkos::View<const double *, DeviceType> scale, rho, fp;
  Kokkos::View<double *[10], Kokkos::LayoutRight, DeviceType> out;
  int nt1, nr, nrho, need_e, need_v;
  double cutforcesq, rdr, rdrho, rhomax;
  KOKKOS_INLINE_FUNCTION void operator()(const int q) const
  {
    const int s = quad_site(q);
    const double xi0 = x(s, 0), xi1 = x(s, 1), xi2 = x(s, 2);
    const int itype = type(s);
    const double fpi = fp(s);
    double fx = 0, fy = 0, fz = 0, eq = 0;
    double vd[6] = {0, 0, 0, 0, 0, 0};

    if (need_e) {
      double p = rho(s) * rdrho + 1.0;
      int m = static_cast<int>(p);
      m = m < nrho - 1 ? m : nrho - 1;
      m = m > 1 ? m : 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;
      const int row = type2frho(itype) * (nrho + 1) + m;
      double phi = ((frho(row, 3) * p + frho(row, 4)) * p + frho(row, 5)) * p + frho(row, 6);
      if (rho(s) > rhomax) phi += fpi * (rho(s) - rhomax);
      eq += phi * scale(itype * nt1 + itype);
    }

    const int nn = numneigh(s);
    for (int k = 0; k < nn; k++) {
      const int j = neigh(s, k);
      const double delx = xi0 - x(j, 0);
      const double dely = xi1 - x(j, 1);
      const double delz = xi2 - x(j, 2);
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cutforcesq) continue;
      const int jtype = type(j);
      const double r = sqrt(rsq);
      double p = r * rdr + 1.0;
      int m = static_cast<int>(p);
      m = m < nr - 1 ? m : nr - 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;

      int row = type2rhor(itype * nt1 + jtype) * (nr + 1) + m;
      const double rhoip = (rhor(row, 0) * p + rhor(row, 1)) * p + rhor(row, 2);
      row = type2rhor(jtype * nt1 + itype) * (nr + 1) + m;
      const double rhojp = (rhor(row, 0) * p + rhor(row, 1)) * p + rhor(row, 2);
      row = type2z2r(itype * nt1 + jtype) * (nr + 1) + m;
      const double z2p = (z2r(row, 0) * p + z2r(row, 1)) * p + z2r(row, 2);
      const double z2 = ((z2r(row, 3) * p + z2r(row, 4)) * p + z2r(row, 5)) * p + z2r(row, 6);

      const double recip = 1.0 / r;
      const double phi = z2 * recip;
      const double phip = z2p * recip - phi * recip;
      const double psip = fpi * rhojp + fp(j) * rhoip + phip;
      const double sc = scale(itype * nt1 + jtype);
      const double fpair = -sc * psip * recip;

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
      if (need_e) eq += 0.5 * sc * phi;
    }
    out(q, 0) = fx;
    out(q, 1) = fy;
    out(q, 2) = fz;
    out(q, 3) = eq;
    for (int d = 0; d < 6; d++) out(q, 4 + d) = vd[d];
  }
};

}    // namespace

template <class DeviceType> void PairCACEAMKokkosCore<DeviceType>::init(PairCACEAM &pair, LAMMPS *lmp)
{
  if (pair.site_legacy) pair.error->all(FLERR, "pair_style cac/eam/*/kk needs the site-only lists: unset CAC_LEGACY_LISTS");
  if (lmp->kokkos) pair.error->all(FLERR, "pair_style cac/eam/*/kk offloads the CAC force kernel itself; run without '-k on'");
  dev.init(&pair, lmp, true);
  tables_ready = false;
}

template <class DeviceType> void PairCACEAMKokkosCore<DeviceType>::compute(PairCACEAM &P)
{
  const int nt = P.atom->ntypes;
  const int nt1 = nt + 1;
  if (!tables_ready) {
    auto mkspline = [&](Kokkos::View<double *[7], Kokkos::LayoutRight, DeviceType> &d, double ***sp, int ntab, int n) {
      d = Kokkos::View<double *[7], Kokkos::LayoutRight, DeviceType>("cac/eam/kk:spline", (size_t) ntab * (n + 1));
      auto h = Kokkos::create_mirror_view(d);
      for (int a = 0; a < ntab; a++)
        for (int m = 0; m <= n; m++)
          for (int c = 0; c < 7; c++) h((size_t) a * (n + 1) + m, c) = sp[a][m][c];
      Kokkos::deep_copy(d, h);
    };
    mkspline(d_frho, P.frho_spline, P.nfrho, P.nrho);
    mkspline(d_rhor, P.rhor_spline, P.nrhor, P.nr);
    mkspline(d_z2r, P.z2r_spline, P.nz2r, P.nr);

    d_type2frho = Kokkos::View<int *, DeviceType>("cac/eam/kk:t2f", nt1);
    d_type2rhor = Kokkos::View<int *, DeviceType>("cac/eam/kk:t2r", (size_t) nt1 * nt1);
    d_type2z2r = Kokkos::View<int *, DeviceType>("cac/eam/kk:t2z", (size_t) nt1 * nt1);
    d_scale = Kokkos::View<double *, DeviceType>("cac/eam/kk:scale", (size_t) nt1 * nt1);
    auto h1 = Kokkos::create_mirror_view(d_type2frho);
    auto h2 = Kokkos::create_mirror_view(d_type2rhor);
    auto h3 = Kokkos::create_mirror_view(d_type2z2r);
    auto h4 = Kokkos::create_mirror_view(d_scale);
    for (int i = 0; i < nt1; i++) {
      h1(i) = (i >= 1) ? P.type2frho[i] : 0;
      for (int j = 0; j < nt1; j++) {
        const bool ok = (i >= 1 && j >= 1);
        h2((size_t) i * nt1 + j) = ok ? P.type2rhor[i][j] : 0;
        h3((size_t) i * nt1 + j) = ok ? P.type2z2r[i][j] : 0;
        h4((size_t) i * nt1 + j) = ok ? P.scale[i][j] : 0.0;
      }
    }
    Kokkos::deep_copy(d_type2frho, h1);
    Kokkos::deep_copy(d_type2rhor, h2);
    Kokkos::deep_copy(d_type2z2r, h3);
    Kokkos::deep_copy(d_scale, h4);
    tables_ready = true;
  }

  dev.prepare();
  const double tk = dev.tick();
  if (d_rho.extent(0) != (size_t) dev.nsite) {
    d_rho = Kokkos::View<double *, DeviceType>("cac/eam/kk:rho", dev.nsite);
    d_fp = Kokkos::View<double *, DeviceType>("cac/eam/kk:fp", dev.nsite);
  }

  CACEAMDensity<DeviceType> kd;
  kd.x = dev.d_x;
  kd.type = dev.d_type;
  kd.numneigh = dev.d_numneigh;
  kd.ilist = dev.d_ilist;
  kd.type2rhor = d_type2rhor;
  kd.type2frho = d_type2frho;
  kd.neigh = dev.d_neighbors;
  kd.rhor = d_rhor;
  kd.frho = d_frho;
  kd.rho = d_rho;
  kd.fp = d_fp;
  kd.nt1 = nt1;
  kd.nr = P.nr;
  kd.nrho = P.nrho;
  kd.cutforcesq = P.cutforcesq;
  kd.rdr = P.rdr;
  kd.rdrho = P.rdrho;
  Kokkos::parallel_for("CACEAMDensity", Kokkos::RangePolicy<DeviceType>(0, dev.ncenters), kd);

  CACEAMForce<DeviceType> kf;
  kf.x = dev.d_x;
  kf.type = dev.d_type;
  kf.quad_site = dev.d_quad_site;
  kf.numneigh = dev.d_numneigh;
  kf.type2rhor = d_type2rhor;
  kf.type2frho = d_type2frho;
  kf.type2z2r = d_type2z2r;
  kf.neigh = dev.d_neighbors;
  kf.rhor = d_rhor;
  kf.frho = d_frho;
  kf.z2r = d_z2r;
  kf.scale = d_scale;
  kf.rho = d_rho;
  kf.fp = d_fp;
  kf.out = dev.d_quad_out;
  kf.nt1 = nt1;
  kf.nr = P.nr;
  kf.nrho = P.nrho;
  kf.need_e = P.quad_eflag ? 1 : 0;
  kf.need_v = P.atom->CAC_virial ? 1 : 0;
  kf.cutforcesq = P.cutforcesq;
  kf.rdr = P.rdr;
  kf.rdrho = P.rdrho;
  kf.rhomax = P.rhomax;
  Kokkos::parallel_for("CACEAMForce", Kokkos::RangePolicy<DeviceType>(0, dev.nquad), kf);
  dev.tock(8, tk);

  dev.download();
}

namespace LAMMPS_NS {
template class PairCACEAMKokkosCore<LMPDeviceType>;
template class PairCACEAMKokkos<LMPDeviceType>;
template class PairCACEAMAlloyKokkos<LMPDeviceType>;
template class PairCACEAMFSKokkos<LMPDeviceType>;
#ifdef KOKKOS_ENABLE_CUDA
template class PairCACEAMKokkosCore<LMPHostType>;
template class PairCACEAMKokkos<LMPHostType>;
template class PairCACEAMAlloyKokkos<LMPHostType>;
template class PairCACEAMFSKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
