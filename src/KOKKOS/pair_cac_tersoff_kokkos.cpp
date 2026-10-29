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

#include "pair_cac_tersoff_kokkos.h"

#include "atom.h"
#include "error.h"

#include <cmath>

using namespace LAMMPS_NS;

namespace {

constexpr int MAXSHORT = 64;    // neighbors within the potential cutoff of one center
constexpr double THIRD_ = 1.0 / 3.0;
constexpr double PI2_ = 1.57079632679489661923;
constexpr double PI4_ = 0.78539816339744830962;

typedef PairCACTersoff::Param TParam;

// the Tersoff functions of PairCACTersoff, written as device functions with identical expressions
KOKKOS_INLINE_FUNCTION double ters_fc(double r, const TParam &p)
{
  const double R = p.bigr, D = p.bigd;
  if (r < R - D) return 1.0;
  if (r > R + D) return 0.0;
  return 0.5 * (1.0 - sin(PI2_ * (r - R) / D));
}

KOKKOS_INLINE_FUNCTION double ters_fc_d(double r, const TParam &p)
{
  const double R = p.bigr, D = p.bigd;
  if (r < R - D) return 0.0;
  if (r > R + D) return 0.0;
  return -(PI4_ / D) * cos(PI2_ * (r - R) / D);
}

KOKKOS_INLINE_FUNCTION double ters_fa(double r, const TParam &p)
{
  if (r > p.bigr + p.bigd) return 0.0;
  return -p.bigb * exp(-p.lam2 * r) * ters_fc(r, p);
}

KOKKOS_INLINE_FUNCTION double ters_fa_d(double r, const TParam &p)
{
  if (r > p.bigr + p.bigd) return 0.0;
  return p.bigb * exp(-p.lam2 * r) * (p.lam2 * ters_fc(r, p) - ters_fc_d(r, p));
}

KOKKOS_INLINE_FUNCTION double ters_bij(double zeta, const TParam &p)
{
  const double tmp = p.beta * zeta;
  if (tmp > p.c1) return 1.0 / sqrt(tmp);
  if (tmp > p.c2) return (1.0 - pow(tmp, -p.powern) / (2.0 * p.powern)) / sqrt(tmp);
  if (tmp < p.c4) return 1.0;
  if (tmp < p.c3) return 1.0 - pow(tmp, p.powern) / (2.0 * p.powern);
  return pow(1.0 + pow(tmp, p.powern), -1.0 / (2.0 * p.powern));
}

KOKKOS_INLINE_FUNCTION double ters_bij_d(double zeta, const TParam &p)
{
  const double tmp = p.beta * zeta;
  if (tmp > p.c1) return p.beta * -0.5 * pow(tmp, -1.5);
  if (tmp > p.c2)
    return p.beta * (-0.5 * pow(tmp, -1.5) * (1.0 - (1.0 + 1.0 / (2.0 * p.powern)) * pow(tmp, -p.powern)));
  if (tmp < p.c4) return 0.0;
  if (tmp < p.c3) return -0.5 * p.beta * pow(tmp, p.powern - 1.0);
  const double tmp_n = pow(tmp, p.powern);
  return -0.5 * pow(1.0 + tmp_n, -1.0 - (1.0 / (2.0 * p.powern))) * tmp_n / zeta;
}

KOKKOS_INLINE_FUNCTION double ters_gijk(const double costheta, const TParam &p)
{
  const double ters_c = p.c * p.c;
  const double ters_d = p.d * p.d;
  const double hcth = p.h - costheta;
  return p.gamma * (1.0 + ters_c / ters_d - ters_c / (ters_d + hcth * hcth));
}

KOKKOS_INLINE_FUNCTION double ters_gijk_d(const double costheta, const TParam &p)
{
  const double ters_c = p.c * p.c;
  const double ters_d = p.d * p.d;
  const double hcth = p.h - costheta;
  const double numerator = -2.0 * ters_c * hcth;
  const double denominator = 1.0 / (ters_d + hcth * hcth);
  return p.gamma * numerator * denominator * denominator;
}

KOKKOS_INLINE_FUNCTION double vdot(const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
KOKKOS_INLINE_FUNCTION void vscale(const double k, const double *x, double *y)
{
  y[0] = k * x[0];
  y[1] = k * x[1];
  y[2] = k * x[2];
}
KOKKOS_INLINE_FUNCTION void vscaleadd(const double k, const double *x, const double *y, double *z)
{
  z[0] = k * x[0] + y[0];
  z[1] = k * x[1] + y[1];
  z[2] = k * x[2] + y[2];
}

KOKKOS_INLINE_FUNCTION void costheta_d(const double *rij_hat, double rij, const double *rik_hat, double rik,
                                       double *dri, double *drj, double *drk)
{
  const double cos_theta = vdot(rij_hat, rik_hat);
  vscaleadd(-cos_theta, rij_hat, rik_hat, drj);
  vscale(1.0 / rij, drj, drj);
  vscaleadd(-cos_theta, rik_hat, rij_hat, drk);
  vscale(1.0 / rik, drk, drk);
  dri[0] = drj[0] + drk[0];
  dri[1] = drj[1] + drk[1];
  dri[2] = drj[2] + drk[2];
  vscale(-1.0, dri, dri);
}

KOKKOS_INLINE_FUNCTION void ters_zetaterm_d(double prefactor, const double *rij_hat, double rij, const double *rik_hat,
                                            double rik, double *dri, double *drj, double *drk, const TParam &param)
{
  double ex_delr, ex_delr_d, tmp;
  double dcosdri[3], dcosdrj[3], dcosdrk[3];

  const double fc = ters_fc(rik, param);
  const double dfc = ters_fc_d(rik, param);
  if (param.powermint == 3) tmp = pow(param.lam3 * (rij - rik), 3.0);
  else tmp = param.lam3 * (rij - rik);

  if (tmp > 69.0776) ex_delr = 1.e30;
  else if (tmp < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(tmp);

  if (param.powermint == 3) ex_delr_d = 3.0 * pow(param.lam3, 3.0) * pow(rij - rik, 2.0) * ex_delr;
  else ex_delr_d = param.lam3 * ex_delr;

  const double cos_theta = vdot(rij_hat, rik_hat);
  const double gijk = ters_gijk(cos_theta, param);
  const double gijk_d = ters_gijk_d(cos_theta, param);
  costheta_d(rij_hat, rij, rik_hat, rik, dcosdri, dcosdrj, dcosdrk);

  vscale(-dfc * gijk * ex_delr, rik_hat, dri);
  vscaleadd(fc * gijk_d * ex_delr, dcosdri, dri, dri);
  vscaleadd(fc * gijk * ex_delr_d, rik_hat, dri, dri);
  vscaleadd(-fc * gijk * ex_delr_d, rij_hat, dri, dri);
  vscale(prefactor, dri, dri);

  vscale(fc * gijk_d * ex_delr, dcosdrj, drj);
  vscaleadd(fc * gijk * ex_delr_d, rij_hat, drj, drj);
  vscale(prefactor, drj, drj);

  vscale(dfc * gijk * ex_delr, rik_hat, drk);
  vscaleadd(fc * gijk_d * ex_delr, dcosdrk, drk, drk);
  vscaleadd(-fc * gijk * ex_delr_d, rik_hat, drk, drk);
  vscale(prefactor, drk, drk);
}

KOKKOS_INLINE_FUNCTION void ters_repulsive(const TParam &p, double rsq, double &fforce, int eflag, double &eng)
{
  const double r = sqrt(rsq);
  const double tmp_fc = ters_fc(r, p);
  const double tmp_fc_d = ters_fc_d(r, p);
  const double tmp_exp = exp(-p.lam1 * r);
  fforce = -p.biga * tmp_exp * (tmp_fc_d - tmp_fc * p.lam1) / r;
  if (eflag) eng = tmp_fc * p.biga * tmp_exp;
}

KOKKOS_INLINE_FUNCTION double ters_zeta(const TParam &p, double rsqij, double rsqik, const double *delrij,
                                        const double *delrik)
{
  const double rij = sqrt(rsqij);
  const double rik = sqrt(rsqik);
  const double costheta = (delrij[0] * delrik[0] + delrij[1] * delrik[1] + delrij[2] * delrik[2]) / (rij * rik);
  double arg, ex_delr;
  if (p.powermint == 3) arg = pow(p.lam3 * (rij - rik), 3.0);
  else arg = p.lam3 * (rij - rik);
  if (arg > 69.0776) ex_delr = 1.e30;
  else if (arg < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(arg);
  return ters_fc(rik, p) * ters_gijk(costheta, p) * ex_delr;
}

KOKKOS_INLINE_FUNCTION void ters_force_zeta(const TParam &p, double rsq, double zeta_ij, double &fforce,
                                            double &prefactor, int eflag, double &eng)
{
  const double r = sqrt(rsq);
  const double fa = ters_fa(r, p);
  const double fa_d = ters_fa_d(r, p);
  const double bij = ters_bij(zeta_ij, p);
  fforce = 0.5 * bij * fa_d / r;
  prefactor = -0.5 * fa * ters_bij_d(zeta_ij, p);
  if (eflag) eng = 0.5 * bij * fa;
}

KOKKOS_INLINE_FUNCTION void ters_attractive(const TParam &p, double prefactor, double rsqij, double rsqik,
                                            const double *delrij, const double *delrik, double *fi, double *fj,
                                            double *fk)
{
  double rij_hat[3], rik_hat[3];
  const double rij = sqrt(rsqij);
  const double rijinv = 1.0 / rij;
  vscale(rijinv, delrij, rij_hat);
  const double rik = sqrt(rsqik);
  const double rikinv = 1.0 / rik;
  vscale(rikinv, delrik, rik_hat);
  ters_zetaterm_d(prefactor, rij_hat, rij, rik_hat, rik, fi, fj, fk, p);
}

template <class DeviceType> struct CACTersoffKernel {
  Kokkos::View<const double *[3], Kokkos::LayoutRight, DeviceType> x;
  Kokkos::View<const int *, DeviceType> type, numneigh, ilist, e2p, map;
  typename ArrayTypes<DeviceType>::t_neighbors_2d neigh;
  Kokkos::View<const TParam *, DeviceType> params;
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> sf;
  Kokkos::View<double *, DeviceType> se;
  Kokkos::View<double *[6], Kokkos::LayoutRight, DeviceType> sv;
  Kokkos::View<int *, DeviceType> err;
  int ne, need_e, need_v;
  double cutshortsq;

  KOKKOS_INLINE_FUNCTION void addv(int s, double v0, double v1, double v2, double v3, double v4, double v5) const
  {
    Kokkos::atomic_add(&sv(s, 0), v0);
    Kokkos::atomic_add(&sv(s, 1), v1);
    Kokkos::atomic_add(&sv(s, 2), v2);
    Kokkos::atomic_add(&sv(s, 3), v3);
    Kokkos::atomic_add(&sv(s, 4), v4);
    Kokkos::atomic_add(&sv(s, 5), v5);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const int ci) const
  {
    const int i = ilist(ci);
    const int itype = map(type(i));
    const double xi0 = x(i, 0), xi1 = x(i, 1), xi2 = x(i, 2);
    int nbj[MAXSHORT], nbt[MAXSHORT];
    double dr[MAXSHORT][3], rr[MAXSHORT];
    int ns = 0;
    const int nn = numneigh(i);
    for (int l = 0; l < nn; l++) {
      const int j = neigh(i, l);
      const double d0 = x(j, 0) - xi0, d1 = x(j, 1) - xi1, d2 = x(j, 2) - xi2;
      const double r2 = d0 * d0 + d1 * d1 + d2 * d2;
      if (r2 < cutshortsq) {
        if (ns < MAXSHORT) {
          nbj[ns] = j;
          nbt[ns] = map(type(j));
          dr[ns][0] = d0;
          dr[ns][1] = d1;
          dr[ns][2] = d2;
          rr[ns] = r2;
          ns++;
        } else
          Kokkos::atomic_store(&err(0), 1);
      }
    }

    // repulsive pair terms, each pair once (center with the smaller index)
    for (int l = 0; l < ns; l++) {
      const int j = nbj[l];
      if (j < i) continue;
      const int jt = nbt[l];
      const int ijparam = e2p((itype * ne + jt) * ne + jt);
      const double rsq = rr[l];
      if (rsq >= params(ijparam).cutsq) continue;
      double fpair = 0.0, eng = 0.0;
      ters_repulsive(params(ijparam), rsq, fpair, need_e, eng);
      const double delx = -dr[l][0], dely = -dr[l][1], delz = -dr[l][2];
      Kokkos::atomic_add(&sf(i, 0), delx * fpair);
      Kokkos::atomic_add(&sf(i, 1), dely * fpair);
      Kokkos::atomic_add(&sf(i, 2), delz * fpair);
      Kokkos::atomic_add(&sf(j, 0), -delx * fpair);
      Kokkos::atomic_add(&sf(j, 1), -dely * fpair);
      Kokkos::atomic_add(&sf(j, 2), -delz * fpair);
      if (need_e) {
        Kokkos::atomic_add(&se(i), 0.5 * eng);
        Kokkos::atomic_add(&se(j), 0.5 * eng);
      }
      if (need_v) {
        const double v0 = 0.5 * delx * delx * fpair, v1 = 0.5 * dely * dely * fpair, v2 = 0.5 * delz * delz * fpair,
                     v3 = 0.5 * delx * dely * fpair, v4 = 0.5 * delx * delz * fpair, v5 = 0.5 * dely * delz * fpair;
        addv(i, v0, v1, v2, v3, v4, v5);
        addv(j, v0, v1, v2, v3, v4, v5);
      }
    }

    // attractive terms of the ordered pairs (i,j)
    for (int l = 0; l < ns; l++) {
      const int jt = nbt[l];
      const int ijparam = e2p((itype * ne + jt) * ne + jt);
      const double rsq1 = rr[l];
      if (rsq1 >= params(ijparam).cutsq) continue;
      const int j = nbj[l];
      const double *delr1 = dr[l];

      double zeta_ij = 0.0;
      for (int k = 0; k < ns; k++) {
        if (k == l) continue;
        const int ijkparam = e2p((itype * ne + jt) * ne + nbt[k]);
        const double rsq2 = rr[k];
        if (rsq2 >= params(ijkparam).cutsq) continue;
        zeta_ij += ters_zeta(params(ijkparam), rsq1, rsq2, delr1, dr[k]);
      }

      double fpair = 0.0, prefactor = 0.0, eng = 0.0;
      ters_force_zeta(params(ijparam), rsq1, zeta_ij, fpair, prefactor, need_e, eng);
      for (int d = 0; d < 3; d++) {
        Kokkos::atomic_add(&sf(i, d), delr1[d] * fpair);
        Kokkos::atomic_add(&sf(j, d), -delr1[d] * fpair);
      }
      if (need_e) {
        Kokkos::atomic_add(&se(i), 0.5 * eng);
        Kokkos::atomic_add(&se(j), 0.5 * eng);
      }
      if (need_v) {
        const double v0 = -0.5 * delr1[0] * delr1[0] * fpair, v1 = -0.5 * delr1[1] * delr1[1] * fpair,
                     v2 = -0.5 * delr1[2] * delr1[2] * fpair, v3 = -0.5 * delr1[0] * delr1[1] * fpair,
                     v4 = -0.5 * delr1[0] * delr1[2] * fpair, v5 = -0.5 * delr1[1] * delr1[2] * fpair;
        addv(i, v0, v1, v2, v3, v4, v5);
        addv(j, v0, v1, v2, v3, v4, v5);
      }

      for (int k = 0; k < ns; k++) {
        if (k == l) continue;
        const int ijkparam = e2p((itype * ne + jt) * ne + nbt[k]);
        const double rsq2 = rr[k];
        if (rsq2 >= params(ijkparam).cutsq) continue;
        const double *delr2 = dr[k];
        const int kk = nbj[k];
        double fi[3], fj[3], fk[3];
        ters_attractive(params(ijkparam), prefactor, rsq1, rsq2, delr1, delr2, fi, fj, fk);
        for (int d = 0; d < 3; d++) {
          Kokkos::atomic_add(&sf(i, d), fi[d]);
          Kokkos::atomic_add(&sf(j, d), fj[d]);
          Kokkos::atomic_add(&sf(kk, d), fk[d]);
        }
        if (need_v) {
          const double w0 = THIRD_ * (delr1[0] * fj[0] + delr2[0] * fk[0]);
          const double w1 = THIRD_ * (delr1[1] * fj[1] + delr2[1] * fk[1]);
          const double w2 = THIRD_ * (delr1[2] * fj[2] + delr2[2] * fk[2]);
          const double w3 = THIRD_ * (delr1[0] * fj[1] + delr2[0] * fk[1]);
          const double w4 = THIRD_ * (delr1[0] * fj[2] + delr2[0] * fk[2]);
          const double w5 = THIRD_ * (delr1[1] * fj[2] + delr2[1] * fk[2]);
          addv(i, w0, w1, w2, w3, w4, w5);
          addv(j, w0, w1, w2, w3, w4, w5);
          addv(kk, w0, w1, w2, w3, w4, w5);
        }
      }
    }
  }
};

}    // namespace

template <class DeviceType> PairCACTersoffKokkos<DeviceType>::PairCACTersoffKokkos(LAMMPS *lmp) : PairCACTersoff(lmp)
{
  tables_ready = false;
}

template <class DeviceType> PairCACTersoffKokkos<DeviceType>::~PairCACTersoffKokkos() = default;

template <class DeviceType> void PairCACTersoffKokkos<DeviceType>::init_style()
{
  PairCACTersoff::init_style();
  if (site_legacy) error->all(FLERR, "pair_style cac/tersoff/kk needs the site-only lists: unset CAC_LEGACY_LISTS");
  if (lmp->kokkos) error->all(FLERR, "pair_style cac/tersoff/kk offloads the CAC force kernel itself; run without '-k on'");
  dev.init(this, lmp, true);
  tables_ready = false;
}

template <class DeviceType> void PairCACTersoffKokkos<DeviceType>::compute_sites(int, int)
{
  if (!tables_ready) {
    d_params = Kokkos::View<TParam *, DeviceType>("cac/tersoff/kk:params", nparams);
    auto hp = Kokkos::create_mirror_view(d_params);
    for (int m = 0; m < nparams; m++) hp(m) = params[m];
    Kokkos::deep_copy(d_params, hp);
    d_e2p = Kokkos::View<int *, DeviceType>("cac/tersoff/kk:e2p", (size_t) nelements * nelements * nelements);
    auto he = Kokkos::create_mirror_view(d_e2p);
    for (int i = 0; i < nelements; i++)
      for (int j = 0; j < nelements; j++)
        for (int k = 0; k < nelements; k++) he((size_t) (i * nelements + j) * nelements + k) = elem2param[i][j][k];
    Kokkos::deep_copy(d_e2p, he);
    const int nt1 = atom->ntypes + 1;
    d_map = Kokkos::View<int *, DeviceType>("cac/tersoff/kk:map", nt1);
    auto hm = Kokkos::create_mirror_view(d_map);
    for (int t = 0; t < nt1; t++) hm(t) = (t >= 1) ? map[t] : 0;
    Kokkos::deep_copy(d_map, hm);
    d_err = Kokkos::View<int *, DeviceType>("cac/tersoff/kk:err", 1);
    tables_ready = true;
  }

  dev.prepare();
  const double tk = dev.tick();
  dev.scatter_zero();
  Kokkos::deep_copy(d_err, 0);

  CACTersoffKernel<DeviceType> k;
  k.x = dev.d_x;
  k.type = dev.d_type;
  k.numneigh = dev.d_numneigh;
  k.ilist = dev.d_ilist;
  k.e2p = d_e2p;
  k.map = d_map;
  k.neigh = dev.d_neighbors;
  k.params = d_params;
  k.sf = dev.d_sf;
  k.se = dev.d_se;
  k.sv = dev.d_sv;
  k.err = d_err;
  k.ne = nelements;
  k.need_e = quad_eflag ? 1 : 0;
  k.need_v = atom->CAC_virial ? 1 : 0;
  k.cutshortsq = cutmax * cutmax;
  Kokkos::parallel_for("CACTersoffKernel", Kokkos::RangePolicy<DeviceType>(0, dev.ncenters), k);

  dev.scatter_gather();
  dev.tock(8, tk);
  auto he = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_err);
  if (he(0)) error->one(FLERR, "cac/tersoff/kk: more than {} neighbors within the cutoff of a site", MAXSHORT);
  dev.download();
}

namespace LAMMPS_NS {
template class PairCACTersoffKokkos<LMPDeviceType>;
#ifdef KOKKOS_ENABLE_CUDA
template class PairCACTersoffKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
