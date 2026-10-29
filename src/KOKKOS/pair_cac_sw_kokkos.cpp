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

#include "pair_cac_sw_kokkos.h"

#include "atom.h"
#include "error.h"

#include <cmath>

using namespace LAMMPS_NS;

namespace {

constexpr int MAXSHORT = 64;    // neighbors within the potential cutoff of one center
constexpr double THIRD_ = 1.0 / 3.0;

typedef PairCACSW::Param SWParam;

// same expressions as PairCACSW::twobody / threebody
KOKKOS_INLINE_FUNCTION void sw_twobody(const SWParam &param, double rsq, double &fforce, int eflag, double &eng)
{
  const double r = sqrt(rsq);
  const double rinvsq = 1.0 / rsq;
  const double rp = pow(r, -param.powerp);
  const double rq = pow(r, -param.powerq);
  const double rainv = 1.0 / (r - param.cut);
  const double rainvsq = rainv * rainv * r;
  const double expsrainv = exp(param.sigma * rainv);
  fforce = (param.c1 * rp - param.c2 * rq + (param.c3 * rp - param.c4 * rq) * rainvsq) * expsrainv * rinvsq;
  if (eflag) eng = (param.c5 * rp - param.c6 * rq) * expsrainv;
}

KOKKOS_INLINE_FUNCTION void sw_threebody(const SWParam &paramij, const SWParam &paramik, const SWParam &paramijk,
                                         double rsq1, double rsq2, const double *delr1, const double *delr2,
                                         double *fj, double *fk, int eflag, double &eng)
{
  const double r1 = sqrt(rsq1);
  const double rinvsq1 = 1.0 / rsq1;
  const double rainv1 = 1.0 / (r1 - paramij.cut);
  const double gsrainv1 = paramij.sigma_gamma * rainv1;
  const double gsrainvsq1 = gsrainv1 * rainv1 / r1;
  const double expgsrainv1 = exp(gsrainv1);

  const double r2 = sqrt(rsq2);
  const double rinvsq2 = 1.0 / rsq2;
  const double rainv2 = 1.0 / (r2 - paramik.cut);
  const double gsrainv2 = paramik.sigma_gamma * rainv2;
  const double gsrainvsq2 = gsrainv2 * rainv2 / r2;
  const double expgsrainv2 = exp(gsrainv2);

  const double rinv12 = 1.0 / (r1 * r2);
  const double cs = (delr1[0] * delr2[0] + delr1[1] * delr2[1] + delr1[2] * delr2[2]) * rinv12;
  const double delcs = cs - paramijk.costheta;
  const double delcssq = delcs * delcs;

  const double facexp = expgsrainv1 * expgsrainv2;
  const double facrad = paramijk.lambda_epsilon * facexp * delcssq;
  const double frad1 = facrad * gsrainvsq1;
  const double frad2 = facrad * gsrainvsq2;
  const double facang = paramijk.lambda_epsilon2 * facexp * delcs;
  const double facang12 = rinv12 * facang;
  const double csfacang = cs * facang;
  const double csfac1 = rinvsq1 * csfacang;

  fj[0] = delr1[0] * (frad1 + csfac1) - delr2[0] * facang12;
  fj[1] = delr1[1] * (frad1 + csfac1) - delr2[1] * facang12;
  fj[2] = delr1[2] * (frad1 + csfac1) - delr2[2] * facang12;

  const double csfac2 = rinvsq2 * csfacang;

  fk[0] = delr2[0] * (frad2 + csfac2) - delr1[0] * facang12;
  fk[1] = delr2[1] * (frad2 + csfac2) - delr1[1] * facang12;
  fk[2] = delr2[2] * (frad2 + csfac2) - delr1[2] * facang12;

  if (eflag) eng = facrad;
}

template <class DeviceType> struct CACSWKernel {
  Kokkos::View<const double *[3], Kokkos::LayoutRight, DeviceType> x;
  Kokkos::View<const int *, DeviceType> type, numneigh, ilist, e2p, map;
  typename ArrayTypes<DeviceType>::t_neighbors_2d neigh;
  Kokkos::View<const SWParam *, DeviceType> params;
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

    // two-body terms: each pair once, from the center with the smaller index
    for (int l = 0; l < ns; l++) {
      const int j = nbj[l];
      if (j < i) continue;
      const int jt = nbt[l];
      const int ijparam = e2p((itype * ne + jt) * ne + jt);
      const double rsq = rr[l];
      if (rsq >= params(ijparam).cutsq) continue;
      double fpair = 0.0, eng = 0.0;
      sw_twobody(params(ijparam), rsq, fpair, need_e, eng);
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

    // three-body terms centered on i
    for (int l = 0; l < ns - 1; l++) {
      const int jt = nbt[l];
      const int ijparam = e2p((itype * ne + jt) * ne + jt);
      const double rsq1 = rr[l];
      if (rsq1 >= params(ijparam).cutsq) continue;
      const int j = nbj[l];
      for (int k = l + 1; k < ns; k++) {
        const int kt = nbt[k];
        const int ikparam = e2p((itype * ne + kt) * ne + kt);
        const int ijkparam = e2p((itype * ne + jt) * ne + kt);
        const double rsq2 = rr[k];
        if (rsq2 >= params(ikparam).cutsq) continue;
        const int kk = nbj[k];
        double fj[3], fk[3], eng = 0.0;
        sw_threebody(params(ijparam), params(ikparam), params(ijkparam), rsq1, rsq2, dr[l], dr[k], fj, fk, need_e, eng);
        for (int d = 0; d < 3; d++) {
          Kokkos::atomic_add(&sf(i, d), -(fj[d] + fk[d]));
          Kokkos::atomic_add(&sf(j, d), fj[d]);
          Kokkos::atomic_add(&sf(kk, d), fk[d]);
        }
        if (need_e) {
          const double e3 = eng / 3.0;
          Kokkos::atomic_add(&se(i), e3);
          Kokkos::atomic_add(&se(j), e3);
          Kokkos::atomic_add(&se(kk), e3);
        }
        if (need_v) {
          const double w0 = THIRD_ * (dr[l][0] * fj[0] + dr[k][0] * fk[0]);
          const double w1 = THIRD_ * (dr[l][1] * fj[1] + dr[k][1] * fk[1]);
          const double w2 = THIRD_ * (dr[l][2] * fj[2] + dr[k][2] * fk[2]);
          const double w3 = THIRD_ * (dr[l][0] * fj[1] + dr[k][0] * fk[1]);
          const double w4 = THIRD_ * (dr[l][0] * fj[2] + dr[k][0] * fk[2]);
          const double w5 = THIRD_ * (dr[l][1] * fj[2] + dr[k][1] * fk[2]);
          addv(i, w0, w1, w2, w3, w4, w5);
          addv(j, w0, w1, w2, w3, w4, w5);
          addv(kk, w0, w1, w2, w3, w4, w5);
        }
      }
    }
  }
};

}    // namespace

template <class DeviceType> PairCACSWKokkos<DeviceType>::PairCACSWKokkos(LAMMPS *lmp) : PairCACSW(lmp)
{
  tables_ready = false;
}

template <class DeviceType> PairCACSWKokkos<DeviceType>::~PairCACSWKokkos() = default;

template <class DeviceType> void PairCACSWKokkos<DeviceType>::init_style()
{
  PairCACSW::init_style();
  if (site_legacy) error->all(FLERR, "pair_style cac/sw/kk needs the site-only lists: unset CAC_LEGACY_LISTS");
  if (lmp->kokkos) error->all(FLERR, "pair_style cac/sw/kk offloads the CAC force kernel itself; run without '-k on'");
  dev.init(this, lmp, true);
  tables_ready = false;
}

template <class DeviceType> void PairCACSWKokkos<DeviceType>::compute_sites(int, int)
{
  if (!tables_ready) {
    d_params = Kokkos::View<SWParam *, DeviceType>("cac/sw/kk:params", nparams);
    auto hp = Kokkos::create_mirror_view(d_params);
    for (int m = 0; m < nparams; m++) hp(m) = params[m];
    Kokkos::deep_copy(d_params, hp);
    d_e2p = Kokkos::View<int *, DeviceType>("cac/sw/kk:e2p", (size_t) nelements * nelements * nelements);
    auto he = Kokkos::create_mirror_view(d_e2p);
    for (int i = 0; i < nelements; i++)
      for (int j = 0; j < nelements; j++)
        for (int k = 0; k < nelements; k++) he((size_t) (i * nelements + j) * nelements + k) = elem2param[i][j][k];
    Kokkos::deep_copy(d_e2p, he);
    const int nt1 = atom->ntypes + 1;
    d_map = Kokkos::View<int *, DeviceType>("cac/sw/kk:map", nt1);
    auto hm = Kokkos::create_mirror_view(d_map);
    for (int t = 0; t < nt1; t++) hm(t) = (t >= 1) ? map[t] : 0;
    Kokkos::deep_copy(d_map, hm);
    d_err = Kokkos::View<int *, DeviceType>("cac/sw/kk:err", 1);
    tables_ready = true;
  }

  dev.prepare();
  const double tk = dev.tick();
  dev.scatter_zero();
  Kokkos::deep_copy(d_err, 0);

  CACSWKernel<DeviceType> k;
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
  Kokkos::parallel_for("CACSWKernel", Kokkos::RangePolicy<DeviceType>(0, dev.ncenters), k);

  dev.scatter_gather();
  dev.tock(8, tk);
  auto he = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_err);
  if (he(0)) error->one(FLERR, "cac/sw/kk: more than {} neighbors within the cutoff of a site", MAXSHORT);
  dev.download();
}

namespace LAMMPS_NS {
template class PairCACSWKokkos<LMPDeviceType>;
#ifdef KOKKOS_ENABLE_CUDA
template class PairCACSWKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
