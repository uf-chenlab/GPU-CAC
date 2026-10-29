/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/
   Copyright (2003) Sandia Corporation.
------------------------------------------------------------------------- */

// Kokkos NEP kernels and their driver, shared by pair_style nep/kk and cac/nep/kk.
// The kernels (NEP_Compute_Functor) were moved here unchanged from pair_nep_kokkos.h.

#ifndef LMP_NEP_KOKKOS_ENGINE_H
#define LMP_NEP_KOKKOS_ENGINE_H

#include "error.h"
#include "kokkos_type.h"
#include "nep.h"
#include "nep_kokkos.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace LAMMPS_NS {

// ----------------------------------------------------------------------
// Compile-time model shape.  A value > 0 is a compile-time constant (loops unroll, per-atom arrays
// live in registers); 0 selects the runtime value (generic path).
//   LM = l_max_3body, NANG = n_max_angular+1, NRAD = n_max_radial+1, BR/BA = basis_size radial/
//   angular, NL = num_L (3-body L=1..LM, plus one slot each for 4-body and 5-body when present)
// ----------------------------------------------------------------------
template <int LM_, int NANG_, int NRAD_, int BR_, int BA_, int NL_> struct NEPShape {
  static constexpr int LM = LM_;
  static constexpr int NANG = NANG_;
  static constexpr int NRAD = NRAD_;
  static constexpr int BR = BR_;
  static constexpr int BA = BA_;
  static constexpr int NL = NL_;
};
using NEPShapeGeneric = NEPShape<0, 0, 0, 0, 0, 0>;
// GPUMD/NEP4 default hyper-parameters: n_max 4 4, basis_size 8 8, l_max 4 2 0
using NEPShapeDefault = NEPShape<4, 5, 5, 8, 8, 5>;

// Runtime dimensions (small POD that rides along in the kernel arguments)
struct NEP_Dims {
  int L_max = 0, num_L = 0;
  int n_max_radial = 0, n_max_angular = 0;
  int basis_r = 0, basis_a = 0;
  int num_types = 0, dim = 0, neurons = 0;
  bool zbl_flexibled = false, zbl_typewise = false;
  bool no_moments = false;    // testing aid: force the many-species fallback in the radial descriptor
};

// Precision model: T = per-pair math (basis functions, harmonics, pair forces); TA = per-atom
// accumulation and network (lane/neighbor reductions, descriptor sums, ANN, Fp, A, per-atom energy).
// fp64: T = TA = double.  fp32: T = float, TA = double (the FP64 work is per atom, not per pair).
// Per-type-pair and network data. Type pair index t12 = itype*ntypes+jtype.
template <class DeviceType, class T, class TA> struct NEP_Data {
  Kokkos::View<T ***, DeviceType> c_radial;     // [n][k][t12]
  Kokkos::View<T ***, DeviceType> c_angular;    // [n][k][t12]
  Kokkos::View<TA ***, DeviceType> c_radial_a;  // same in TA (W pre-contraction)
  Kokkos::View<TA ***, DeviceType> w0;          // [type][neuron][dim]
  Kokkos::View<TA **, DeviceType> b0, w1;       // [type][neuron]
  Kokkos::View<TA *, DeviceType> b1;            // [1]
  Kokkos::View<TA *, DeviceType> qscale;        // [dim]
  Kokkos::View<T *, DeviceType> rc_rad, rcinv_rad, rc_ang, rcinv_ang;    // [t12]
  Kokkos::View<T *, DeviceType> zbl_ainv, zbl_zizj, zbl_rin, zbl_rout;   // [t12]
  Kokkos::View<T *, DeviceType> zbl_para;                                // flexible: [pair][10]
  Kokkos::View<int *, DeviceType> zbl_index;                             // [t12] -> flexible pair
  // df64 copies of the angular radial-basis data (hybrid df64 mode: basis evaluated in double-float)
  Kokkos::View<df64 ***, DeviceType> c_angular_x;
  Kokkos::View<df64 *, DeviceType> rc_ang_x, rcinv_ang_x;
};

// Per-step work arrays
template <class DeviceType, class T, class TA> struct NEP_Work {
  Kokkos::View<TA **, DeviceType> Fp;     // [N][dim]   descriptor q, overwritten by dE/dq
  Kokkos::View<TA ***, DeviceType> A;     // [N][NA][dim_s] angular sums s, overwritten by A=dE/ds
  Kokkos::View<T ***, DeviceType> W;      // [N][ntypes][2*basis_r+2]: W_k, their sum, k*W_k
};

// V = number of adjacent threads (lanes) cooperating on one atom
constexpr int NEP_REG_MOM_TYPES = 4;    // FP32 register moments support up to this many types
constexpr int NEP_MAX_MOM_TYPES = 6;    // radial-descriptor moment trick supports up to this many types

template <int V> struct TagNeighbor {};
template <int V> struct TagDescRadial {};
template <int V> struct TagDescAngular {};
template <int V> struct TagANN {};
template <int V> struct TagForceRadial {};
struct TagZBL {};
template <int V> struct TagForceAngular {};

template <class DeviceType, class T, class TA, class Shape, int NG> struct NEP_Compute_Functor {
  typedef DeviceType device_type;

  NEP_Dims dims;
  NEP_Data<DeviceType, T, TA> D;
  NEP_Work<DeviceType, T, TA> Wk;

  typename ArrayTypes<DeviceType>::t_kkfloat_1d_3_lr_randomread x;
  typename ArrayTypes<DeviceType>::t_kkacc_1d_3 f;
  typename ArrayTypes<DeviceType>::t_kkacc_1d_6 v_virial;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type;
  Kokkos::View<int *, DeviceType> type_map;
  Kokkos::View<int *, DeviceType> ntype;    // NEP type of every local+ghost atom: type_map(type(i))
  Kokkos::View<KK_FLOAT *, DeviceType> rcsq_rad, rcsq_ang;    // double, for the short-list build
  // float screening of the short lists: float positions, cutoff + margin (pairs inside the margin have
  // fc = 0 and contribute exactly zero)
  Kokkos::View<float *[3], Kokkos::LayoutRight, DeviceType> x32;
  Kokkos::View<float *, DeviceType> rcf_rad, rcf_ang;
  float screen_margin = 0.0f;
  bool screen32 = true;
  // df64 positions (only used when T = df64): r_ij = x_j - x_i is then formed without native FP64
  Kokkos::View<df64 *[3], Kokkos::LayoutRight, DeviceType> xdf;

  struct PosX {
    df64 q[3];
  };
  KOKKOS_INLINE_FUNCTION PosX load_posx(const int i) const
  {
    PosX p;
    p.q[0] = xdf(i, 0); p.q[1] = xdf(i, 1); p.q[2] = xdf(i, 2);
    return p;
  }
  // hybrid helper (T = double): unit vector, 1/d and gn[n] (and gnp[n]) of one angular neighbor from df64 arithmetic
  KOKKOS_INLINE_FUNCTION void hyb_pair(const PosX &px, const int j, const int t12, const int na, const int ba,
                                       T &ux, T &uy, T &uz, T &d12inv_out, T *gn, T *gnp, const bool want_p) const
  {
    const df64 a0 = xdf(j, 0) - px.q[0], a1 = xdf(j, 1) - px.q[1], a2 = xdf(j, 2) - px.q[2];
    df64 d12, d12i;
    nep_dist(a0 * a0 + a1 * a1 + a2 * a2, d12, d12i);
    const df64 rc = D.rc_ang_x(t12), rcinv = D.rcinv_ang_x(t12);
    df64 fc, fcp(0.0f);
    df64 fn[NBA1], fnp[NBA1];
    if (want_p) {
      nep_fc_fcp(rc, rcinv, d12, fc, fcp);
      nep_fn_fnp(ba, rcinv, d12, fc, fcp, fn, fnp);
    } else {
      nep_fc(rc, rcinv, d12, fc);
      nep_fn(ba, rcinv, d12, fc, fn);
    }
#pragma unroll
    for (int n = 0; n < NAMAX; ++n) {
      if (n < na) {
        nep_acc<df64> aa, bb;
#pragma unroll
        for (int k = 0; k < NBA1; ++k) {
          if (k <= ba) {
            const df64 c = D.c_angular_x(n, k, t12);
            aa.add(fn[k], c);
            if (want_p) bb.add(fnp[k], c);
          }
        }
        gn[n] = (T) aa.get();
        if (want_p) gnp[n] = (T) bb.get();
      }
    }
    ux = (T) (a0 * d12i);
    uy = (T) (a1 * d12i);
    uz = (T) (a2 * d12i);
    d12inv_out = (T) d12i;
  }

  struct PosI {
    KK_FLOAT d[3];
    df64 q[3];
  };
  KOKKOS_INLINE_FUNCTION PosI load_pos(const int i) const
  {
    PosI p;
    if constexpr (nep_is_df64<T>::value) {
      p.q[0] = xdf(i, 0); p.q[1] = xdf(i, 1); p.q[2] = xdf(i, 2);
    } else {
      p.d[0] = x(i, 0); p.d[1] = x(i, 1); p.d[2] = x(i, 2);
    }
    return p;
  }
  KOKKOS_INLINE_FUNCTION void rij(const PosI &p, const int j, T &a, T &b, T &c) const
  {
    if constexpr (nep_is_df64<T>::value) {
      a = xdf(j, 0) - p.q[0]; b = xdf(j, 1) - p.q[1]; c = xdf(j, 2) - p.q[2];
    } else {
      a = T(x(j, 0) - p.d[0]); b = T(x(j, 1) - p.d[1]); c = T(x(j, 2) - p.d[2]);
    }
  }

  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread d_numneigh;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread d_ilist;
  typename ArrayTypes<DeviceType>::t_int_2d_dl d_neighbors_short_radial;
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_short_radial;
  typename ArrayTypes<DeviceType>::t_int_2d_dl d_neighbors_short_angular;
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_short_angular;
  // pairs inside the ZBL range (2.5 A): a compact list, so the radial force warps never run the ZBL branch
  typename ArrayTypes<DeviceType>::t_int_2d_dl d_neighbors_short_zbl;
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_short_zbl;

  typename ArrayTypes<DeviceType>::t_kkacc_1d eatom;

  bool need_virial = true;
  bool zbl_on = false;
  bool hyb = false;    // df64 hybrid, angular kernels (T = double): geometry/basis in df64, contraction in double
  bool use_fscr = false;    // fp32: neighbor force scatter goes to a float scratch (folded into f later)
  Kokkos::View<float *[4], Kokkos::LayoutRight, DeviceType> fscr;
  int natoms = 0;    // number of local atoms in the neighbor list (inum)

  // compile-time (Shape) or runtime (dims) model dimensions
  KOKKOS_INLINE_FUNCTION int L_max() const { return Shape::LM > 0 ? Shape::LM : dims.L_max; }
  KOKKOS_INLINE_FUNCTION int NA() const { return Shape::NANG > 0 ? Shape::NANG : dims.n_max_angular + 1; }
  KOKKOS_INLINE_FUNCTION int NR() const { return Shape::NRAD > 0 ? Shape::NRAD : dims.n_max_radial + 1; }
  KOKKOS_INLINE_FUNCTION int BR() const { return Shape::BR > 0 ? Shape::BR : dims.basis_r; }
  KOKKOS_INLINE_FUNCTION int BA() const { return Shape::BA > 0 ? Shape::BA : dims.basis_a; }
  KOKKOS_INLINE_FUNCTION int NLn() const { return Shape::NL > 0 ? Shape::NL : dims.num_L; }
  KOKKOS_INLINE_FUNCTION int DIM() const
  {
    return Shape::LM > 0 ? Shape::NRAD + Shape::NANG * Shape::NL : dims.dim;
  }

  static constexpr int ABC = Shape::LM > 0 ? (Shape::LM + 1) * (Shape::LM + 1) - 1 : NEP_NUM_OF_ABC;
  static constexpr int NBR1 = Shape::BR > 0 ? Shape::BR + 1 : NEP_MAX_BASIS1;
  static constexpr int NBA1 = Shape::BA > 0 ? Shape::BA + 1 : NEP_MAX_BASIS1;
  static constexpr int NRMAX = Shape::NRAD > 0 ? Shape::NRAD : NEP_MAX_BASIS1;
  static constexpr int NAMAX = Shape::NANG > 0 ? Shape::NANG : NEP_MAX_BASIS1;
  static constexpr int DIMMAX = Shape::LM > 0 ? Shape::NRAD + Shape::NANG * Shape::NL : NEP_MAX_DIM;
  static constexpr int NGEFF = Shape::NANG > 0 ? NG : 1;

  // ----------------------------------------------------------------------
  // Short neighbor lists (radial and angular cutoff), built from the full skin list
  // ----------------------------------------------------------------------
  // true if neighbor j (type pair t12) is inside the radial / angular cutoff of atom i
  KOKKOS_INLINE_FUNCTION void screen_pair(const int i, const int j, const int t12, const KK_FLOAT xd[3],
                                          const float xf[3], bool &in_r, bool &in_a, bool &in_z) const
  {
    if (screen32) {
      const float dx = xf[0] - x32(j, 0), dy = xf[1] - x32(j, 1), dz = xf[2] - x32(j, 2);
      const float rsq = dx * dx + dy * dy + dz * dz;
      const float tr = rcf_rad(t12) + screen_margin, ta = rcf_ang(t12) + screen_margin;
      const float tz = 2.5f + screen_margin;
      in_r = rsq < tr * tr;
      in_a = rsq < ta * ta;
      in_z = zbl_on && rsq < tz * tz;
    } else {
      const KK_FLOAT dx = xd[0] - x(j, 0), dy = xd[1] - x(j, 1), dz = xd[2] - x(j, 2);
      const KK_FLOAT rsq = dx * dx + dy * dy + dz * dz;
      in_r = rsq < rcsq_rad(t12);
      in_a = rsq < rcsq_ang(t12);
      in_z = zbl_on && rsq < KK_FLOAT(6.25);
    }
  }

  template <int V>
  KOKKOS_INLINE_FUNCTION void operator()(TagNeighbor<V>, const int &tid) const
  {
    const int ii0 = tid / V;
    const int lane = tid % V;
    const bool valid = ii0 < natoms;
    const int ii = valid ? ii0 : 0;
    const int i = d_ilist(ii);
    const int itype = ntype(i);
    const KK_FLOAT xd[3] = {screen32 ? KK_FLOAT(0) : x(i, 0), screen32 ? KK_FLOAT(0) : x(i, 1),
                            screen32 ? KK_FLOAT(0) : x(i, 2)};
    const float xf[3] = {screen32 ? x32(i, 0) : 0.0f, screen32 ? x32(i, 1) : 0.0f,
                         screen32 ? x32(i, 2) : 0.0f};
    const int jnum = valid ? d_numneigh[i] : 0;
    int index_radial = 0;
    int index_angular = 0;
    int index_zbl = 0;
    if (V == 1) {
      for (int jj = 0; jj < jnum; jj++) {
        int j = d_neighbors(i, jj);
        j &= NEIGHMASK;
        const int t12 = itype * dims.num_types + ntype(j);
        bool in_r, in_a, in_z;
        screen_pair(i, j, t12, xd, xf, in_r, in_a, in_z);
        if (in_r) d_neighbors_short_radial(ii, index_radial++) = j;
        if (in_a) d_neighbors_short_angular(ii, index_angular++) = j;
        if (in_z) d_neighbors_short_zbl(ii, index_zbl++) = j;
      }
    } else {
#if defined(__CUDA_ARCH__)
      // the V lanes of one atom sweep the full list in chunks of V; ballots (restricted to the
      // atom's own lanes) give each qualifying neighbor a slot, so the short lists stay compact
      unsigned lid;
      asm volatile("mov.u32 %0, %%laneid;" : "=r"(lid));
      const unsigned gmask = (V == 32) ? 0xffffffffu : (((1u << V) - 1u) << (lid & ~(unsigned) (V - 1)));
      const unsigned below = (1u << lid) - 1u;
      for (int j0 = 0; j0 < jnum; j0 += V) {
        const int jj = j0 + lane;
        int j = 0;
        bool in_r = false, in_a = false, in_z = false;
        if (jj < jnum) {
          j = d_neighbors(i, jj);
          j &= NEIGHMASK;
          const int t12 = itype * dims.num_types + ntype(j);
          screen_pair(i, j, t12, xd, xf, in_r, in_a, in_z);
        }
        const unsigned mr = __ballot_sync(gmask, in_r) & gmask;
        const unsigned ma = __ballot_sync(gmask, in_a) & gmask;
        const unsigned mz = __ballot_sync(gmask, in_z) & gmask;
        if (in_r) d_neighbors_short_radial(ii, index_radial + __popc(mr & below)) = j;
        if (in_a) d_neighbors_short_angular(ii, index_angular + __popc(ma & below)) = j;
        if (in_z) d_neighbors_short_zbl(ii, index_zbl + __popc(mz & below)) = j;
        index_radial += __popc(mr);
        index_angular += __popc(ma);
        index_zbl += __popc(mz);
      }
#endif
    }
    if (valid && lane == 0) {
      d_numneigh_short_radial(ii) = index_radial;
      d_numneigh_short_angular(ii) = index_angular;
      d_numneigh_short_zbl(ii) = index_zbl;
    }
  }

  // ----------------------------------------------------------------------
  // Radial descriptor: q[n] = sum_j sum_k fn_k(r_ij) c[n][k][t_ij]   (unscaled; stored in Fp)
  // ----------------------------------------------------------------------
  template <int V>
  KOKKOS_INLINE_FUNCTION void operator()(TagDescRadial<V>, const int &tid) const
  {
    const int ii0 = tid / V;
    const int lane = tid % V;
    const bool valid = ii0 < natoms;
    const int ii = valid ? ii0 : 0;
    const int i = d_ilist(ii);
    const int itype = ntype(i);
    const PosI pos_i = load_pos(i);
    const int nr = NR();
    const int br = BR();
    const int jnum = valid ? d_numneigh_short_radial(ii) : 0;

    T q[NRMAX];
#pragma unroll
    for (int n = 0; n < NRMAX; ++n) q[n] = T(0.0);

    // q[n] = sum_j sum_k fn_k(r_j) c[n][k][t_ij]: accumulate fn_k per neighbor type once per pair
    // (mom[jt][k]) and contract with c once per atom, instead of 45 FMAs per pair.
    const int nt = dims.num_types;
    // only profitable when FP64-bound; in FP32 the local-memory traffic of the moments costs more
    const bool use_mom = sizeof(T) == 8 && !nep_is_df64<T>::value && nt <= NEP_MAX_MOM_TYPES && !dims.no_moments;
    // mom[jt][0..nb] = sum_j 0.5*fc*T_k, mom[jt][nb+1] = sum_j 0.5*fc   (fn_k = 0.5*fc*(T_k+1))
    constexpr int MS = NBR1 + 1;
    T mom[NEP_MAX_MOM_TYPES * MS];
    if (use_mom) {
      for (int m = 0; m < NEP_MAX_MOM_TYPES * MS; ++m) mom[m] = T(0.0);
    }
    // FP32: same idea with compile-time-indexed accumulators (registers, predicated adds): no local memory
    const bool use_reg = (sizeof(T) == 4 || nep_is_df64<T>::value) && nt <= NEP_REG_MOM_TYPES && !dims.no_moments;
    T rmom[NEP_REG_MOM_TYPES][MS];
#pragma unroll
    for (int jt = 0; jt < NEP_REG_MOM_TYPES; ++jt)
#pragma unroll
      for (int k = 0; k < MS; ++k) rmom[jt][k] = T(0.0);

    // two neighbors per iteration for df64: the geometry phase is straight-line code so that the two long,
    // dependent double-float chains can be interleaved by the compiler (the kernel is latency-bound)
    constexpr int UNR = nep_is_df64<T>::value ? 2 : 1;
    for (int jj0 = lane; jj0 < jnum; jj0 += UNR * V) {
      int jtv[UNR];
      bool okv[UNR];
      T d12v[UNR], fcv[UNR], rcinvv[UNR];
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
        okv[u] = (jj0 + u * V) < jnum;
        const int jj = okv[u] ? jj0 + u * V : jj0;    // clamped: always a valid list index
        const int j = d_neighbors_short_radial(ii, jj);
        const int jtype = ntype(j);
        const int t12 = itype * dims.num_types + jtype;
        T r0, r1, r2;
        rij(pos_i, j, r0, r1, r2);
        T d12, d12inv_unused;
        nep_dist(r0 * r0 + r1 * r1 + r2 * r2, d12, d12inv_unused);
        rcinvv[u] = D.rcinv_rad(t12);
        nep_fc(D.rc_rad(t12), rcinvv[u], d12, fcv[u]);
        d12v[u] = d12;
        jtv[u] = jtype;
      }
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
      if (!okv[u]) continue;
      const int jtype = jtv[u];
      const int t12 = itype * dims.num_types + jtype;
      const T d12 = d12v[u];
      const T fc12 = fcv[u];
      const T rcinv = rcinvv[u];
      if (use_mom) {
        T Tk[NBR1];
        nep_cheb(br, rcinv, d12, Tk);
        const T hf = T(0.5) * fc12;
        T *m = mom + jtype * MS;
#pragma unroll
        for (int k = 0; k < NBR1; ++k)
          if (k <= br) m[k] += hf * Tk[k];
        m[br + 1] += hf;
      } else if (use_reg) {
        T Tk[NBR1];
        nep_cheb(br, rcinv, d12, Tk);
        const T hf = T(0.5) * fc12;
#pragma unroll
        for (int jt = 0; jt < NEP_REG_MOM_TYPES; ++jt) {
          if (jtype == jt) {
#pragma unroll
            for (int k = 0; k < NBR1; ++k)
              if (k <= br) rmom[jt][k] += hf * Tk[k];
            rmom[jt][br + 1] += hf;
          }
        }
      } else {
        T fn[NBR1];
        nep_fn(br, rcinv, d12, fc12, fn);
#pragma unroll
        for (int n = 0; n < NRMAX; ++n) {
          if (n < nr) {
            T g = T(0.0);
#pragma unroll
            for (int k = 0; k < NBR1; ++k)
              if (k <= br) g += fn[k] * D.c_radial(n, k, t12);
            q[n] += g;
          }
        }
      }
      }
    }
    if (use_mom) {
      for (int jt = 0; jt < nt; ++jt) {
        const int t12 = itype * nt + jt;
#pragma unroll
        for (int n = 0; n < NRMAX; ++n) {
          if (n < nr) {
            T g = T(0.0), csum = T(0.0);
#pragma unroll
            for (int k = 0; k < NBR1; ++k) {
              if (k <= br) {
                const T c = D.c_radial(n, k, t12);
                g += mom[jt * MS + k] * c;
                csum += c;
              }
            }
            q[n] += g + csum * mom[jt * MS + br + 1];
          }
        }
      }
    }
    if (use_reg) {
#pragma unroll
      for (int jt = 0; jt < NEP_REG_MOM_TYPES; ++jt) {
        if (jt < nt) {
          const int t12 = itype * nt + jt;
#pragma unroll
          for (int n = 0; n < NRMAX; ++n) {
            if (n < nr) {
              T g = T(0.0), csum = T(0.0);
#pragma unroll
              for (int k = 0; k < NBR1; ++k) {
                if (k <= br) {
                  const T c = D.c_radial(n, k, t12);
                  g += rmom[jt][k] * c;
                  csum += c;
                }
              }
              q[n] += g + csum * rmom[jt][br + 1];
            }
          }
        }
      }
    }
    // lane partial sums (few terms, in T) are combined in TA
    TA qa[NRMAX];
#pragma unroll
    for (int n = 0; n < NRMAX; ++n) qa[n] = nep_vsum<V>((TA) q[n]);
    if (valid && lane == 0)
      for (int n = 0; n < nr; ++n) Wk.Fp(i, n) = qa[n];
  }

  // ----------------------------------------------------------------------
  // Angular descriptor: s[n][abc] = sum_j gn_n(r_ij) Y_abc(u_ij), then q from s.
  // NG radial indices n are processed per pass sharing one evaluation of Y per neighbor.
  // ----------------------------------------------------------------------
  template <int V>
  KOKKOS_INLINE_FUNCTION void operator()(TagDescAngular<V>, const int &tid) const
  {
    const int ii0 = tid / V;
    const int lane = tid % V;
    const bool valid = ii0 < natoms;
    const int ii = valid ? ii0 : 0;
    const int i = d_ilist(ii);
    const int itype = ntype(i);
    const PosI pos_i = load_pos(i);
    const int na = NA();
    const int ba = BA();
    const int lmax = L_max();
    const int nl = NLn();
    const int dim_s = (lmax + 1) * (lmax + 1) - 1;
    const int nr = NR();
    const int jnum = valid ? d_numneigh_short_angular(ii) : 0;

    TA q_ang[DIMMAX];
#pragma unroll
    for (int d = 0; d < DIMMAX; ++d) q_ang[d] = TA(0.0);

    bool hy = false;
    if constexpr (std::is_same<T, double>::value) hy = hyb;
    PosX pos_x;
    if (hy) pos_x = load_posx(i);

    for (int n0 = 0; n0 < na; n0 += NGEFF) {
      nep_acc<T> s[NGEFF][ABC];

      for (int jj = lane; jj < jnum; jj += V) {
        const int j = d_neighbors_short_angular(ii, jj);
        const int t12 = itype * dims.num_types + ntype(j);
        T ux, uy, uz, gnh[NAMAX];
        T fn[NBA1];
        if (hy) {
          T unused[NAMAX], dinv_unused;
          hyb_pair(pos_x, j, t12, na, ba, ux, uy, uz, dinv_unused, gnh, unused, false);
        } else {
          T r0, r1, r2;
          rij(pos_i, j, r0, r1, r2);
          T d12, d12inv;
          nep_dist(r0 * r0 + r1 * r1 + r2 * r2, d12, d12inv);
          const T rc = D.rc_ang(t12);
          const T rcinv = D.rcinv_ang(t12);
          T fc12;
          nep_fc(rc, rcinv, d12, fc12);
          nep_fn(ba, rcinv, d12, fc12, fn);
          ux = r0 * d12inv;
          uy = r1 * d12inv;
          uz = r2 * d12inv;
        }
        T Y[ABC];
        nep_Y(lmax, ux, uy, uz, Y);
#pragma unroll
        for (int g = 0; g < NGEFF; ++g) {
          const int n = n0 + g;
          if (n < na) {
            T gn;
            if (hy) {
              gn = gnh[n];
            } else {
              nep_acc<T> agn;
#pragma unroll
              for (int k = 0; k < NBA1; ++k)
                if (k <= ba) agn.add(fn[k], D.c_angular(n, k, t12));
              gn = agn.get();
            }
#pragma unroll
            for (int abc = 0; abc < ABC; ++abc)
              if (abc < dim_s) s[g][abc].add(gn, Y[abc]);
          }
        }
      }

      // combine the partial sums of the V lanes in TA; every lane then holds the full s
#pragma unroll
      for (int g = 0; g < NGEFF; ++g) {
        const int n = n0 + g;
        if (n < na) {
          TA sg[ABC];
#pragma unroll
          for (int abc = 0; abc < ABC; ++abc) sg[abc] = nep_vsum<V>((TA) s[g][abc].get());
          nep_find_q(lmax, nl, na, n, sg, q_ang);
          if (valid) {
#pragma unroll
            for (int abc = 0; abc < ABC; ++abc)
              if (abc < dim_s && (abc % V) == lane) Wk.A(i, n, abc) = sg[abc];
          }
        }
      }
    }
    if (valid)
      for (int d = lane; d < nl * na; d += V) Wk.Fp(i, nr + d) = q_ang[d];
  }

  // ----------------------------------------------------------------------
  // ANN forward/backward pass (in TA, lanes split the hidden neurons), then per-atom A (in place over
  // s) and the radial pre-contraction W.
  // ----------------------------------------------------------------------
  template <int V>
  KOKKOS_INLINE_FUNCTION void operator()(TagANN<V>, const int &tid) const
  {
    const int ii0 = tid / V;
    const int lane = tid % V;
    const bool valid = ii0 < natoms;
    const int ii = valid ? ii0 : 0;
    const int i = d_ilist(ii);
    const int itype = ntype(i);
    const int dim = DIM();
    const int neurons = dims.neurons;
    const int nr = NR();
    const int na = NA();
    const int lmax = L_max();
    const int nl = NLn();
    const int dim_s = (lmax + 1) * (lmax + 1) - 1;

    TA q[DIMMAX];
    TA fp[DIMMAX];
#pragma unroll
    for (int d = 0; d < DIMMAX; ++d) {
      q[d] = TA(0.0);
      fp[d] = TA(0.0);
    }
    for (int d = 0; d < dim; ++d) q[d] = Wk.Fp(i, d) * D.qscale(d);

    TA energy = TA(0.0);
    for (int n = lane; n < neurons; n += V) {
      TA z = TA(0.0);
      for (int d = 0; d < dim; ++d) z += D.w0(itype, n, d) * q[d];
      const TA x1 = tanh(z - D.b0(itype, n));
      const TA tan_der = TA(1.0) - x1 * x1;
      energy += D.w1(itype, n) * x1;
      const TA w1_times_der = D.w1(itype, n) * tan_der;
      for (int d = 0; d < dim; ++d) fp[d] += w1_times_der * D.w0(itype, n, d);
    }
    energy = nep_vsum<V>(energy);
#pragma unroll
    for (int d = 0; d < DIMMAX; ++d) fp[d] = nep_vsum<V>(fp[d]);
    energy -= D.b1(0);
    for (int d = 0; d < dim; ++d) fp[d] *= D.qscale(d);
    if (valid && lane == 0) {
      for (int d = 0; d < dim; ++d) Wk.Fp(i, d) = fp[d];
      eatom(i) = (KK_ACC_FLOAT) energy;
    }

    // angular coefficients A, in place; lanes split the radial indices n
    if (valid && lmax > 0) {
      for (int n = lane; n < na; n += V) {
        TA s_n[ABC], a_n[ABC];
#pragma unroll
        for (int abc = 0; abc < ABC; ++abc) {
          s_n[abc] = TA(0.0);
          a_n[abc] = TA(0.0);
        }
        for (int abc = 0; abc < dim_s; ++abc) s_n[abc] = (TA) Wk.A(i, n, abc);
        nep_make_A(lmax, nl, na, n, fp + nr, s_n, a_n);
        for (int abc = 0; abc < dim_s; ++abc) Wk.A(i, n, abc) = a_n[abc];
      }
    }

    // radial pre-contraction: W(i,jt,k) = sum_n Fp_n c_radial(n,k,t_ij); lanes split (jt,k)
    if (valid) {
      const int nk = BR() + 1;
      const int nt = dims.num_types;
      for (int jt = lane; jt < nt; jt += V) {
        const int t12 = itype * nt + jt;
        TA wsum = TA(0.0);
        for (int k = 0; k < nk; ++k) {
          TA w = TA(0.0);
          for (int n = 0; n < nr; ++n) w += fp[n] * D.c_radial_a(n, k, t12);
          Wk.W(i, jt, k) = (T) w;
          wsum += w;
        }
        Wk.W(i, jt, nk) = (T) wsum;
        for (int k = 1; k < nk; ++k) {
          TA w = TA(0.0);
          for (int n = 0; n < nr; ++n) w += fp[n] * D.c_radial_a(n, k, t12);
          Wk.W(i, jt, nk + k) = (T) (TA(k) * w);
        }
      }
    }
  }

  // ----------------------------------------------------------------------
  // ZBL pair term (energy f1, derivative fp) for one radial neighbor
  // ----------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION void zbl_pair(const int t12, const T d12, const T d12inv, T &f1,
                                       T &fp) const
  {
    // evaluated in the accumulation type TA (double for fp64/df64: exp() needs no emulation)
    TA g1, gp;
    const TA r = (TA) d12, ri = (TA) d12inv;
    if (dims.zbl_flexibled) {
      TA para[10];
      const int base = D.zbl_index(t12) * 10;
      for (int k = 0; k < 10; ++k) para[k] = (TA) D.zbl_para(base + k);
      nep_f_and_fp_zbl_flex(para, (TA) D.zbl_zizj(t12), (TA) D.zbl_ainv(t12), r, ri, g1, gp);
    } else {
      nep_f_and_fp_zbl((TA) D.zbl_zizj(t12), (TA) D.zbl_ainv(t12), (TA) D.zbl_rin(t12),
                       (TA) D.zbl_rout(t12), r, ri, g1, gp);
    }
    f1 = T(g1);
    fp = T(gp);
  }

  // ----------------------------------------------------------------------
  // Radial NEP force + ZBL, one pass over the radial short list.
  // ----------------------------------------------------------------------
  template <int V>
  KOKKOS_INLINE_FUNCTION void operator()(TagForceRadial<V>, const int &tid) const
  {
    const int ii0 = tid / V;
    const int lane = tid % V;
    const bool valid = ii0 < natoms;
    const int ii = valid ? ii0 : 0;
    const int i = d_ilist(ii);
    const int itype = ntype(i);
    const PosI pos_i = load_pos(i);
    const int jnum = valid ? d_numneigh_short_radial(ii) : 0;
    const int br = BR();

    T f_i[3] = {T(0.0), T(0.0), T(0.0)};
    T v_i[6] = {T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0)};
    for (int jj = lane; jj < jnum; jj += V) {
      const int j = d_neighbors_short_radial(ii, jj);
      const int jtype = ntype(j);
      const int t12 = itype * dims.num_types + jtype;
      T r12[3];
      rij(pos_i, j, r12[0], r12[1], r12[2]);
      const T d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      T d12, d12inv;
      nep_dist(d12sq, d12, d12inv);

      const T rc = D.rc_rad(t12);
      const T rcinv = D.rcinv_rad(t12);
      T fc12, fcp12;
      nep_fc_fcp(rc, rcinv, d12, fc12, fcp12);
      const T gp = nep_radial_gp(br, rcinv, d12, fc12, fcp12, Wk.W, i, jtype);
      const T tmp12 = gp * d12inv;

      const T f12[3] = {tmp12 * r12[0], tmp12 * r12[1], tmp12 * r12[2]};
      f_i[0] += f12[0];
      f_i[1] += f12[1];
      f_i[2] += f12[2];
      if (sizeof(T) == 4 && use_fscr) {
        nep_scatter_sub_f32(&fscr(j, 0), (float) f12[0], (float) f12[1], (float) f12[2]);
      } else {
        Kokkos::atomic_sub(&f(j, 0), (KK_ACC_FLOAT) f12[0]);
        Kokkos::atomic_sub(&f(j, 1), (KK_ACC_FLOAT) f12[1]);
        Kokkos::atomic_sub(&f(j, 2), (KK_ACC_FLOAT) f12[2]);
      }

      if (need_virial) {
        v_i[0] -= r12[0] * f12[0];
        v_i[1] -= r12[1] * f12[1];
        v_i[2] -= r12[2] * f12[2];
        v_i[3] -= r12[0] * f12[1];
        v_i[4] -= r12[0] * f12[2];
        v_i[5] -= r12[1] * f12[2];
      }
    }

    // combine the V lanes in TA, then one set of atomics per atom
    TA fa[3], va[6];
#pragma unroll
    for (int k = 0; k < 3; ++k) fa[k] = nep_vsum<V>((TA) f_i[k]);
    if (need_virial) {
#pragma unroll
      for (int k = 0; k < 6; ++k) va[k] = nep_vsum<V>((TA) v_i[k]);
    }
    if (valid && lane == 0) {
      Kokkos::atomic_add(&f(i, 0), (KK_ACC_FLOAT) fa[0]);
      Kokkos::atomic_add(&f(i, 1), (KK_ACC_FLOAT) fa[1]);
      Kokkos::atomic_add(&f(i, 2), (KK_ACC_FLOAT) fa[2]);
      if (need_virial) {
        for (int k = 0; k < 6; k++) Kokkos::atomic_add(&v_virial(i, k), (KK_ACC_FLOAT) va[k]);
      }
    }
  }

  // ----------------------------------------------------------------------
  // ZBL: compact list of the few pairs closer than 2.5 A, one thread per atom
  // ----------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION void operator()(TagZBL, const int &ii) const
  {
    const int i = d_ilist(ii);
    const int itype = ntype(i);
    const PosI pos_i = load_pos(i);
    const int jnum = d_numneigh_short_zbl(ii);

    TA f_i[3] = {TA(0.0), TA(0.0), TA(0.0)};
    TA v_i[6] = {TA(0.0), TA(0.0), TA(0.0), TA(0.0), TA(0.0), TA(0.0)};
    TA e_i = TA(0.0);

    for (int jj = 0; jj < jnum; jj++) {
      const int j = d_neighbors_short_zbl(ii, jj);
      const int t12 = itype * dims.num_types + ntype(j);
      T r12[3];
      rij(pos_i, j, r12[0], r12[1], r12[2]);
      const T d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq > T(6.25)) continue;    // list is a (float-screened) superset; the physical range is 2.5 A
      T d12, d12inv;
      nep_dist(d12sq, d12, d12inv);
      T f1, fp;
      zbl_pair(t12, d12, d12inv, f1, fp);
      e_i += (TA) f1 * TA(0.5);
      const T tmp12 = fp * d12inv * T(0.5);
      const T f12[3] = {tmp12 * r12[0], tmp12 * r12[1], tmp12 * r12[2]};
      f_i[0] += (TA) f12[0];
      f_i[1] += (TA) f12[1];
      f_i[2] += (TA) f12[2];
      Kokkos::atomic_sub(&f(j, 0), (KK_ACC_FLOAT) f12[0]);
      Kokkos::atomic_sub(&f(j, 1), (KK_ACC_FLOAT) f12[1]);
      Kokkos::atomic_sub(&f(j, 2), (KK_ACC_FLOAT) f12[2]);
      if (need_virial) {
        v_i[0] -= (TA) (r12[0] * f12[0]);
        v_i[1] -= (TA) (r12[1] * f12[1]);
        v_i[2] -= (TA) (r12[2] * f12[2]);
        v_i[3] -= (TA) (r12[0] * f12[1]);
        v_i[4] -= (TA) (r12[0] * f12[2]);
        v_i[5] -= (TA) (r12[1] * f12[2]);
      }
    }
    if (jnum > 0) {
      Kokkos::atomic_add(&f(i, 0), (KK_ACC_FLOAT) f_i[0]);
      Kokkos::atomic_add(&f(i, 1), (KK_ACC_FLOAT) f_i[1]);
      Kokkos::atomic_add(&f(i, 2), (KK_ACC_FLOAT) f_i[2]);
      if (need_virial) {
        for (int k = 0; k < 6; k++) Kokkos::atomic_add(&v_virial(i, k), (KK_ACC_FLOAT) v_i[k]);
      }
      eatom(i) += (KK_ACC_FLOAT) e_i;
    }
  }

  // ----------------------------------------------------------------------
  // Angular NEP force: per neighbor, gn/gnp for all n, then one fused contraction with A.
  // ----------------------------------------------------------------------
  template <int V>
  KOKKOS_INLINE_FUNCTION void operator()(TagForceAngular<V>, const int &tid) const
  {
    const int ii0 = tid / V;
    const int lane = tid % V;
    const bool valid = ii0 < natoms;
    const int ii = valid ? ii0 : 0;
    const int i = d_ilist(ii);
    const int itype = ntype(i);
    const PosI pos_i = load_pos(i);
    const int jnum = valid ? d_numneigh_short_angular(ii) : 0;
    const int na = NA();
    const int ba = BA();
    const int lmax = L_max();
    bool hy = false;
    if constexpr (std::is_same<T, double>::value) hy = hyb;
    PosX pos_x;
    if (hy) pos_x = load_posx(i);

    T f_i[3] = {T(0.0), T(0.0), T(0.0)};
    T v_i[6] = {T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0)};

    for (int jj = lane; jj < jnum; jj += V) {
      const int j = d_neighbors_short_angular(ii, jj);
      const int t12 = itype * dims.num_types + ntype(j);
      T r12[3];
      rij(pos_i, j, r12[0], r12[1], r12[2]);
      T d12inv, ux, uy, uz;
      T gn[NAMAX], gnp[NAMAX];
      if (hy) {
        hyb_pair(pos_x, j, t12, na, ba, ux, uy, uz, d12inv, gn, gnp, true);
      } else {
        T d12;
        nep_dist(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2], d12, d12inv);
        const T rc = D.rc_ang(t12);
        const T rcinv = D.rcinv_ang(t12);
        T fc12, fcp12;
        nep_fc_fcp(rc, rcinv, d12, fc12, fcp12);
        T fn[NBA1], fnp[NBA1];
        nep_fn_fnp(ba, rcinv, d12, fc12, fcp12, fn, fnp);
#pragma unroll
        for (int n = 0; n < NAMAX; ++n) {
          if (n < na) {
            nep_acc<T> aa, bb;
#pragma unroll
            for (int k = 0; k < NBA1; ++k) {
              if (k <= ba) {
                const T c_val = D.c_angular(n, k, t12);
                aa.add(fn[k], c_val);
                bb.add(fnp[k], c_val);
              }
            }
            gn[n] = aa.get();
            gnp[n] = bb.get();
          }
        }
        ux = r12[0] * d12inv;
        uy = r12[1] * d12inv;
        uz = r12[2] * d12inv;
      }

      T S = T(0.0);
      T G[3] = {T(0.0), T(0.0), T(0.0)};
      nep_contract(lmax, na, gn, gnp, Wk.A, i, ux, uy, uz, S, G);

      const T gu = G[0] * ux + G[1] * uy + G[2] * uz;
      const T f12[3] = {S * ux + (G[0] - gu * ux) * d12inv, S * uy + (G[1] - gu * uy) * d12inv,
                        S * uz + (G[2] - gu * uz) * d12inv};

      f_i[0] += f12[0];
      f_i[1] += f12[1];
      f_i[2] += f12[2];
      Kokkos::atomic_sub(&f(j, 0), (KK_ACC_FLOAT) f12[0]);
      Kokkos::atomic_sub(&f(j, 1), (KK_ACC_FLOAT) f12[1]);
      Kokkos::atomic_sub(&f(j, 2), (KK_ACC_FLOAT) f12[2]);

      if (need_virial) {
        v_i[0] -= r12[0] * f12[0];
        v_i[1] -= r12[1] * f12[1];
        v_i[2] -= r12[2] * f12[2];
        v_i[3] -= r12[0] * f12[1];
        v_i[4] -= r12[0] * f12[2];
        v_i[5] -= r12[1] * f12[2];
      }
    }

    TA fa[3], va[6];
#pragma unroll
    for (int k = 0; k < 3; ++k) fa[k] = nep_vsum<V>((TA) f_i[k]);
    if (need_virial) {
#pragma unroll
      for (int k = 0; k < 6; ++k) va[k] = nep_vsum<V>((TA) v_i[k]);
    }
    if (valid && lane == 0) {
      Kokkos::atomic_add(&f(i, 0), (KK_ACC_FLOAT) fa[0]);
      Kokkos::atomic_add(&f(i, 1), (KK_ACC_FLOAT) fa[1]);
      Kokkos::atomic_add(&f(i, 2), (KK_ACC_FLOAT) fa[2]);
      if (need_virial) {
        for (int k = 0; k < 6; k++) Kokkos::atomic_add(&v_virial(i, k), (KK_ACC_FLOAT) va[k]);
      }
    }
  }
};

// per-atom NEP type (one load instead of the dependent pair type(i) -> type_map)
template <class DeviceType> struct NepTypeFunctor {
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type;
  Kokkos::View<int *, DeviceType> type_map;
  Kokkos::View<int *, DeviceType> ntype;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const { ntype(i) = type_map(type(i)); }
};

// float copy of the positions for the neighbor screening (a named functor: CUDA extended lambdas
// cannot live in protected members)
template <class DeviceType> struct NepX32Functor {
  typename ArrayTypes<DeviceType>::t_kkfloat_1d_3_lr_randomread x;
  Kokkos::View<float *[3], Kokkos::LayoutRight, DeviceType> x32;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const
  {
    x32(i, 0) = (float) x(i, 0);
    x32(i, 1) = (float) x(i, 1);
    x32(i, 2) = (float) x(i, 2);
  }
};

// df64 copy of the positions (hi, lo floats): lets the pair kernels form r_ij without native FP64
template <class DeviceType> struct NepXdfFunctor {
  typename ArrayTypes<DeviceType>::t_kkfloat_1d_3_lr_randomread x;
  Kokkos::View<df64 *[3], Kokkos::LayoutRight, DeviceType> xdf;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const
  {
    xdf(i, 0) = df64((double) x(i, 0));
    xdf(i, 1) = df64((double) x(i, 1));
    xdf(i, 2) = df64((double) x(i, 2));
  }
};

// f(i) += fscr(i): fold the float scatter scratch into the double force array
template <class DeviceType> struct NepFoldFunctor {
  typename ArrayTypes<DeviceType>::t_kkacc_1d_3 f;
  Kokkos::View<float *[4], Kokkos::LayoutRight, DeviceType> fscr;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const
  {
    f(i, 0) += (KK_ACC_FLOAT) fscr(i, 0);
    f(i, 1) += (KK_ACC_FLOAT) fscr(i, 1);
    f(i, 2) += (KK_ACC_FLOAT) fscr(i, 2);
  }
};

// covalent radii (A) for typewise ZBL cutoffs
inline const double *nep_covalent_radius()
{
  static const double r[94] = {
    0.426667, 0.613333, 1.6,     1.25333, 1.02667, 1.0,     0.946667, 0.84,    0.853333, 0.893333,
    1.86667,  1.66667,  1.50667, 1.38667, 1.46667, 1.36,    1.32,     1.28,    2.34667,  2.05333,
    1.77333,  1.62667,  1.61333, 1.46667, 1.42667, 1.38667, 1.33333,  1.32,    1.34667,  1.45333,
    1.49333,  1.45333,  1.53333, 1.46667, 1.52,    1.56,    2.52,     2.22667, 1.96,     1.85333,
    1.76,     1.65333,  1.53333, 1.50667, 1.50667, 1.44,    1.53333,  1.64,    1.70667,  1.68,
    1.68,     1.64,     1.76,    1.74667, 2.78667, 2.34667, 2.16,     1.96,    2.10667,  2.09333,
    2.08,     2.06667,  2.01333, 2.02667, 2.01333, 2.0,     1.98667,  1.98667, 1.97333,  2.04,
    1.94667,  1.82667,  1.74667, 1.64,    1.57333, 1.54667, 1.48,     1.49333, 1.50667,  1.76,
    1.73333,  1.73333,  1.81333, 1.74667, 1.84,    1.89333, 2.68,     2.41333, 2.22667,  2.10667,
    2.02667,  2.04,     2.05333,  2.06667};
  return r;
}

// Opt-in per-kernel timing: set NEP_KK_PROFILE=1. Fences after each kernel, so only use for profiling.
struct NepProfile {
  static constexpr int NK = 8;
  bool on;
  double t[NK] = {0, 0, 0, 0, 0, 0, 0, 0};
  const char *name[NK] = {"neigh", "descR", "descA", "ann", "f_rad", "f_ang", "setup", "reduce"};
  long calls = 0;
  std::chrono::steady_clock::time_point t0;
  NepProfile()
  {
    const char *e = std::getenv("NEP_KK_PROFILE");
    on = e && e[0] == '1';
  }
  void start()
  {
    if (on) {
      Kokkos::fence();
      t0 = std::chrono::steady_clock::now();
    }
  }
  void stop(int k)
  {
    if (!on) return;
    Kokkos::fence();
    t[k] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  ~NepProfile()
  {
    if (!on || calls == 0) return;
    double tot = 0;
    for (double v : t) tot += v;
    fprintf(stderr, "NEP_KK_PROFILE over %ld calls (ms/call):", calls);
    for (int k = 0; k < NK; k++) fprintf(stderr, "  %s=%.3f", name[k], 1e3 * t[k] / calls);
    fprintf(stderr, "  total=%.3f\n", 1e3 * tot / calls);
  }
};

inline NepProfile &nep_kk_prof()
{
  static NepProfile p;
  return p;
}

// ----------------------------------------------------------------------
// NEPKokkosEngine: model data, work arrays and kernel launches of the
// Kokkos NEP implementation, independent of where positions, neighbor
// lists and forces come from.  Used by pair_style nep/kk (LAMMPS atoms)
// and pair_style cac/nep/kk (CAC lattice sites).
//
// Semantics of one run(): for every center d_ilist(0..inum-1) the
// descriptors are built from its neighbors d_neighbors(i, 0..d_numneigh(i)-1),
// eatom(i) is set, and pair forces are scattered onto both the center and
// its neighbors in f (f must be zeroed by the caller).  Arrays indexed by
// atom (x, f, type, eatom, v) must hold at least nall entries; per-atom work
// arrays are sized by nmax.
// ----------------------------------------------------------------------

template <class DeviceType> class NEPKokkosEngine {
 public:
  typedef ArrayTypes<DeviceType> AT;

  NEP_Dims dims;
  NEP_Data<DeviceType, double, double> data_d;    // fp64: T = TA = double
  NEP_Data<DeviceType, float, float> data_f;      // fp32: T = TA = float
  NEP_Work<DeviceType, double, double> work_d;
  NEP_Work<DeviceType, float, float> work_f;
  NEP_Data<DeviceType, df64, double> data_x;      // df64: T = double-float, TA = double
  NEP_Work<DeviceType, df64, double> work_x;
  Kokkos::View<df64 *[3], Kokkos::LayoutRight, DeviceType> k_xdf;
  Kokkos::View<KK_FLOAT *, DeviceType> k_rcsq_rad, k_rcsq_ang;
  Kokkos::View<float *, DeviceType> k_rcf_rad, k_rcf_ang;
  Kokkos::View<float *[3], Kokkos::LayoutRight, DeviceType> k_x32;
  Kokkos::View<float *[4], Kokkos::LayoutRight, DeviceType> k_fscr;
  Kokkos::View<int *, DeviceType> k_type_map;
  Kokkos::View<int *, DeviceType> k_ntype;

  bool zbl_enabled = false;
  bool use_fp32 = false;    // FP32 math, FP64 force accumulation
  bool use_df64 = false;    // double-float (~1e-14) pair math on the FP32 pipe

  // inputs of one run()
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_kkacc_1d_6 d_vatom;
  typename AT::t_int_1d_randomread type;
  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  typename AT::t_kkacc_1d d_eatom;
  int inum = 0;          // number of centers
  int nall = 0;          // number of entries in x/f/type
  int nmax = 0;          // capacity for per-atom work arrays (>= nall)
  bool need_virial = false;
  double box_length = 0.0;    // largest box dimension, sets the float screening margin
  Error *error = nullptr;

  // short neighbor lists built by the TagNeighbor kernel
  typename AT::t_int_2d_dl d_neighbors_short_radial;
  typename AT::t_int_1d d_numneigh_short_radial;
  typename AT::t_int_2d_dl d_neighbors_short_angular;
  typename AT::t_int_1d d_numneigh_short_angular;
  typename AT::t_int_2d_dl d_neighbors_short_zbl;
  typename AT::t_int_1d d_numneigh_short_zbl;

  const NEP3 *model = nullptr;

  // parse model dimensions and upload all model data; type_map[t] = NEP species of LAMMPS type t
  void setup(const NEP3 &nep_model, const int *type_map, int ntypes, Error *err)
  {
    model = &nep_model;
    error = err;
    const auto &pb = nep_model.paramb;
    dims.L_max = pb.L_max;
    dims.num_L = pb.num_L;
    dims.n_max_radial = pb.n_max_radial;
    dims.n_max_angular = pb.n_max_angular;
    dims.basis_r = pb.basis_size_radial;
    dims.basis_a = pb.basis_size_angular;
    dims.num_types = pb.num_types;
    dims.dim = nep_model.annmb.dim;
    dims.neurons = nep_model.annmb.num_neurons1;
    dims.zbl_flexibled = nep_model.zbl.flexibled;
    dims.zbl_typewise = pb.use_typewise_cutoff_zbl;
    zbl_enabled = nep_model.zbl.enabled;

    if (pb.version != 4 && pb.version != 3)
      error->all(FLERR, "Kokkos NEP currently supports NEP3/NEP4 models only");
    if (pb.model_type != 0)
      error->all(FLERR, "Kokkos NEP supports potential models only (no dipole/polarizability)");
    if (dims.dim > NEP_MAX_DIM)
      error->all(FLERR, "NEP descriptor dimension exceeds the supported maximum of 96 in Kokkos NEP");
    if (dims.basis_r + 1 > NEP_MAX_BASIS1 || dims.basis_a + 1 > NEP_MAX_BASIS1)
      error->all(FLERR, "NEP basis_size too large for Kokkos NEP");

    build_data<double, double>(data_d);
    build_data<float, float>(data_f);
    build_data<df64, double>(data_x);

    const int nt = dims.num_types;
    k_rcsq_rad = Kokkos::View<KK_FLOAT *, DeviceType>("nep:rcsq_rad", nt * nt);
    k_rcsq_ang = Kokkos::View<KK_FLOAT *, DeviceType>("nep:rcsq_ang", nt * nt);
    auto h_rr = Kokkos::create_mirror_view(k_rcsq_rad);
    auto h_ra = Kokkos::create_mirror_view(k_rcsq_ang);
    for (int it = 0; it < nt; it++)
      for (int jt = 0; jt < nt; jt++) {
        const double rcr = (pb.rc_radial[it] + pb.rc_radial[jt]) * 0.5;
        const double rca = (pb.rc_angular[it] + pb.rc_angular[jt]) * 0.5;
        h_rr(it * nt + jt) = rcr * rcr;
        h_ra(it * nt + jt) = rca * rca;
      }
    Kokkos::deep_copy(k_rcsq_rad, h_rr);
    Kokkos::deep_copy(k_rcsq_ang, h_ra);
    k_rcf_rad = Kokkos::View<float *, DeviceType>("nep:rcf_rad", nt * nt);
    k_rcf_ang = Kokkos::View<float *, DeviceType>("nep:rcf_ang", nt * nt);
    auto h_fr = Kokkos::create_mirror_view(k_rcf_rad);
    auto h_fa = Kokkos::create_mirror_view(k_rcf_ang);
    for (int it = 0; it < nt; it++)
      for (int jt = 0; jt < nt; jt++) {
        h_fr(it * nt + jt) = (float) ((pb.rc_radial[it] + pb.rc_radial[jt]) * 0.5);
        h_fa(it * nt + jt) = (float) ((pb.rc_angular[it] + pb.rc_angular[jt]) * 0.5);
      }
    Kokkos::deep_copy(k_rcf_rad, h_fr);
    Kokkos::deep_copy(k_rcf_ang, h_fa);

    Kokkos::resize(k_type_map, ntypes + 1);
    auto h_type_map = Kokkos::create_mirror_view(k_type_map);
    for (int i = 0; i <= ntypes; i++) h_type_map(i) = type_map[i];
    Kokkos::deep_copy(k_type_map, h_type_map);
  }

  // (re)allocate the short neighbor lists for ignum centers with up to max_neighs neighbors
  void ensure_short_lists(const int ignum, const int max_neighs)
  {
    if (((int) d_neighbors_short_radial.extent(1) < max_neighs) ||
        ((int) d_neighbors_short_radial.extent(0) < ignum)) {
      // 1.2 buffer factor to reduce reallocations
      d_neighbors_short_radial =
          typename AT::t_int_2d_dl("NEP::neighbors_short_radial", ignum * 1.2, max_neighs);
      d_neighbors_short_angular =
          typename AT::t_int_2d_dl("NEP::neighbors_short_angular", ignum * 1.2, max_neighs);
      d_neighbors_short_zbl =
          typename AT::t_int_2d_dl("NEP::neighbors_short_zbl", ignum * 1.2, zbl_enabled ? max_neighs : 1);
    }
    if ((int) d_numneigh_short_radial.extent(0) < ignum) {
      d_numneigh_short_radial = typename AT::t_int_1d("NEP::numneighs_short_radial", ignum * 1.2);
      d_numneigh_short_angular = typename AT::t_int_1d("NEP::numneighs_short_angular", ignum * 1.2);
      d_numneigh_short_zbl = typename AT::t_int_1d("NEP::numneighs_short_zbl", ignum * 1.2);
    }
  }

  // run all kernels in the selected precision
  void run()
  {
    nep_kk_prof().calls++;
    if (use_df64) run_kernels<df64, double>(data_x, work_x);
    else if (use_fp32) run_kernels<float, float>(data_f, work_f);
    else run_kernels<double, double>(data_d, work_d);
  }

  template <class T, class TA> void build_data(NEP_Data<DeviceType, T, TA> &);
  template <class T, class TA> void run_kernels(NEP_Data<DeviceType, T, TA> &, NEP_Work<DeviceType, T, TA> &);
  template <class T, class TA, class Shape, int NG>
  void launch(NEP_Data<DeviceType, T, TA> &, NEP_Work<DeviceType, T, TA> &);
};


template <class DeviceType>
template <class T, class TA>
void NEPKokkosEngine<DeviceType>::build_data(NEP_Data<DeviceType, T, TA> &dd)
{
  const auto &pb = model->paramb;
  const auto &an = model->annmb;
  const auto &zb = model->zbl;
  const int nt = dims.num_types;
  const int nsq = nt * nt;
  const int dim = dims.dim;
  const int neurons = dims.neurons;

  dd.c_radial = Kokkos::View<T ***, DeviceType>("nep:c_radial", dims.n_max_radial + 1, dims.basis_r + 1, nsq);
  dd.c_angular = Kokkos::View<T ***, DeviceType>("nep:c_angular", dims.n_max_angular + 1, dims.basis_a + 1, nsq);
  dd.c_radial_a = Kokkos::View<TA ***, DeviceType>("nep:c_radial_a", dims.n_max_radial + 1, dims.basis_r + 1, nsq);
  dd.w0 = Kokkos::View<TA ***, DeviceType>("nep:w0", nt, neurons, dim);
  dd.b0 = Kokkos::View<TA **, DeviceType>("nep:b0", nt, neurons);
  dd.w1 = Kokkos::View<TA **, DeviceType>("nep:w1", nt, neurons);
  dd.b1 = Kokkos::View<TA *, DeviceType>("nep:b1", 1);
  dd.qscale = Kokkos::View<TA *, DeviceType>("nep:qscale", dim);
  dd.rc_rad = Kokkos::View<T *, DeviceType>("nep:rc_rad", nsq);
  dd.rcinv_rad = Kokkos::View<T *, DeviceType>("nep:rcinv_rad", nsq);
  dd.rc_ang = Kokkos::View<T *, DeviceType>("nep:rc_ang", nsq);
  dd.rcinv_ang = Kokkos::View<T *, DeviceType>("nep:rcinv_ang", nsq);
  dd.zbl_ainv = Kokkos::View<T *, DeviceType>("nep:zbl_ainv", nsq);
  dd.zbl_zizj = Kokkos::View<T *, DeviceType>("nep:zbl_zizj", nsq);
  dd.zbl_rin = Kokkos::View<T *, DeviceType>("nep:zbl_rin", nsq);
  dd.zbl_rout = Kokkos::View<T *, DeviceType>("nep:zbl_rout", nsq);
  dd.zbl_para = Kokkos::View<T *, DeviceType>("nep:zbl_para", 550);
  dd.zbl_index = Kokkos::View<int *, DeviceType>("nep:zbl_index", nsq);
  dd.c_angular_x = Kokkos::View<df64 ***, DeviceType>("nep:c_angular_x", dims.n_max_angular + 1, dims.basis_a + 1, nsq);
  dd.rc_ang_x = Kokkos::View<df64 *, DeviceType>("nep:rc_ang_x", nsq);
  dd.rcinv_ang_x = Kokkos::View<df64 *, DeviceType>("nep:rcinv_ang_x", nsq);

  auto h_cr = Kokkos::create_mirror_view(dd.c_radial);
  auto h_ca = Kokkos::create_mirror_view(dd.c_angular);
  auto h_cra = Kokkos::create_mirror_view(dd.c_radial_a);
  auto h_w0 = Kokkos::create_mirror_view(dd.w0);
  auto h_b0 = Kokkos::create_mirror_view(dd.b0);
  auto h_w1 = Kokkos::create_mirror_view(dd.w1);
  auto h_b1 = Kokkos::create_mirror_view(dd.b1);
  auto h_qs = Kokkos::create_mirror_view(dd.qscale);
  auto h_rcr = Kokkos::create_mirror_view(dd.rc_rad);
  auto h_rcri = Kokkos::create_mirror_view(dd.rcinv_rad);
  auto h_rca = Kokkos::create_mirror_view(dd.rc_ang);
  auto h_rcai = Kokkos::create_mirror_view(dd.rcinv_ang);
  auto h_zai = Kokkos::create_mirror_view(dd.zbl_ainv);
  auto h_zzz = Kokkos::create_mirror_view(dd.zbl_zizj);
  auto h_zin = Kokkos::create_mirror_view(dd.zbl_rin);
  auto h_zout = Kokkos::create_mirror_view(dd.zbl_rout);
  auto h_zp = Kokkos::create_mirror_view(dd.zbl_para);
  auto h_zidx = Kokkos::create_mirror_view(dd.zbl_index);
  auto h_cax = Kokkos::create_mirror_view(dd.c_angular_x);
  auto h_rcax = Kokkos::create_mirror_view(dd.rc_ang_x);
  auto h_rciax = Kokkos::create_mirror_view(dd.rcinv_ang_x);

  for (int n = 0; n <= dims.n_max_radial; n++)
    for (int k = 0; k <= dims.basis_r; k++)
      for (int t2 = 0; t2 < nsq; t2++) {
        h_cr(n, k, t2) = (T) an.c[(n * (dims.basis_r + 1) + k) * nsq + t2];
        h_cra(n, k, t2) = (TA) an.c[(n * (dims.basis_r + 1) + k) * nsq + t2];
      }
  for (int n = 0; n <= dims.n_max_angular; n++)
    for (int k = 0; k <= dims.basis_a; k++)
      for (int t2 = 0; t2 < nsq; t2++)
      {
        h_ca(n, k, t2) = (T) an.c[pb.num_c_radial + (n * (dims.basis_a + 1) + k) * nsq + t2];
        h_cax(n, k, t2) = df64((double) an.c[pb.num_c_radial + (n * (dims.basis_a + 1) + k) * nsq + t2]);
      }

  for (int t = 0; t < nt; t++) {
    for (int n = 0; n < neurons; n++) {
      for (int d = 0; d < dim; d++) h_w0(t, n, d) = (TA) an.w0[t][n * dim + d];
      h_b0(t, n) = (TA) an.b0[t][n];
      h_w1(t, n) = (TA) an.w1[t][n];
    }
  }
  h_b1(0) = (TA) an.b1[0];
  for (int d = 0; d < dim; d++) h_qs(d) = (TA) pb.q_scaler[d];

  const double K_C_SP = 14.399645;
  for (int it = 0; it < nt; it++) {
    for (int jt = 0; jt < nt; jt++) {
      const int t12 = it * nt + jt;
      const double rcr = (pb.rc_radial[it] + pb.rc_radial[jt]) * 0.5;
      const double rca = (pb.rc_angular[it] + pb.rc_angular[jt]) * 0.5;
      h_rcr(t12) = (T) rcr;
      h_rcri(t12) = (T) (1.0 / rcr);
      h_rca(t12) = (T) rca;
      h_rcai(t12) = (T) (1.0 / rca);
      h_rcax(t12) = df64(rca);
      h_rciax(t12) = df64(1.0 / rca);

      const int zi = pb.atomic_numbers[it] + 1;
      const int zj = pb.atomic_numbers[jt] + 1;
      const double a_inv = (pow((double) zi, 0.23) + pow((double) zj, 0.23)) * 2.134563;
      h_zai(t12) = (T) a_inv;
      h_zzz(t12) = (T) (K_C_SP * zi * zj);
      double rc_inner = zb.rc_inner, rc_outer = zb.rc_outer;
      if (pb.use_typewise_cutoff_zbl) {
        const double type_cut = (nep_covalent_radius()[zi - 1] + nep_covalent_radius()[zj - 1]) *
            pb.typewise_cutoff_zbl_factor;
        if (type_cut < rc_outer) rc_outer = type_cut;
        rc_inner = rc_outer * 0.5;
      }
      h_zin(t12) = (T) rc_inner;
      h_zout(t12) = (T) rc_outer;
      const int t1 = std::min(it, jt), t2 = std::max(it, jt);
      h_zidx(t12) = t1 * zb.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
    }
  }
  for (int i = 0; i < 550; i++) h_zp(i) = (T) zb.para[i];

  Kokkos::deep_copy(dd.c_radial, h_cr);
  Kokkos::deep_copy(dd.c_angular, h_ca);
  Kokkos::deep_copy(dd.c_radial_a, h_cra);
  Kokkos::deep_copy(dd.w0, h_w0);
  Kokkos::deep_copy(dd.b0, h_b0);
  Kokkos::deep_copy(dd.w1, h_w1);
  Kokkos::deep_copy(dd.b1, h_b1);
  Kokkos::deep_copy(dd.qscale, h_qs);
  Kokkos::deep_copy(dd.rc_rad, h_rcr);
  Kokkos::deep_copy(dd.rcinv_rad, h_rcri);
  Kokkos::deep_copy(dd.rc_ang, h_rca);
  Kokkos::deep_copy(dd.rcinv_ang, h_rcai);
  Kokkos::deep_copy(dd.zbl_ainv, h_zai);
  Kokkos::deep_copy(dd.zbl_zizj, h_zzz);
  Kokkos::deep_copy(dd.zbl_rin, h_zin);
  Kokkos::deep_copy(dd.zbl_rout, h_zout);
  Kokkos::deep_copy(dd.zbl_para, h_zp);
  Kokkos::deep_copy(dd.zbl_index, h_zidx);
  Kokkos::deep_copy(dd.c_angular_x, h_cax);
  Kokkos::deep_copy(dd.rc_ang_x, h_rcax);
  Kokkos::deep_copy(dd.rcinv_ang_x, h_rciax);
}


template <class DeviceType>
template <class T, class TA>
void NEPKokkosEngine<DeviceType>::run_kernels(NEP_Data<DeviceType, T, TA> &dd, NEP_Work<DeviceType, T, TA> &wk)
{
  const int nmax = this->nmax;
  const int dim = dims.dim;
  const int na = dims.n_max_angular + 1;
  const int dim_s = (dims.L_max + 1) * (dims.L_max + 1) - 1;
  const int nk = 2 * dims.basis_r + 2;    // W_k (basis_r+1), their sum, and k*W_k (basis_r)

  if ((int) wk.Fp.extent(0) < nmax || (int) wk.Fp.extent(1) < dim)
    wk.Fp = Kokkos::View<TA **, DeviceType>("NEP::Fp", nmax, dim);
  if ((int) wk.A.extent(0) < nmax || (int) wk.A.extent(1) < na || (int) wk.A.extent(2) < dim_s)
    wk.A = Kokkos::View<TA ***, DeviceType>("NEP::A", nmax, na, std::max(dim_s, 1));
  if ((int) wk.W.extent(0) < nmax || (int) wk.W.extent(1) < dims.num_types ||
      (int) wk.W.extent(2) < nk)
    wk.W = Kokkos::View<T ***, DeviceType>("NEP::W", nmax, dims.num_types, nk);

  if constexpr (nep_is_df64<T>::value) {
    // hybrid: the angular kernels run in native double and share the per-atom arrays with the df64 kernels
    work_d.Fp = wk.Fp;
    work_d.A = wk.A;
  }

  const bool force_generic = std::getenv("NEP_KK_GENERIC") != nullptr;    // testing aid
  const bool fast = !force_generic && dims.L_max == NEPShapeDefault::LM && dims.n_max_angular + 1 == NEPShapeDefault::NANG &&
      dims.n_max_radial + 1 == NEPShapeDefault::NRAD && dims.basis_r == NEPShapeDefault::BR &&
      dims.basis_a == NEPShapeDefault::BA && dims.num_L == NEPShapeDefault::NL;

  if (fast) {
    // number of radial indices n handled per angular-descriptor pass (registers vs. recomputation)
    int ng = nep_is_df64<T>::value ? 3 : 5;    // df64: only 3 sets of per-n accumulators fit in registers
    if (const char *e = std::getenv("NEP_KK_NG")) ng = atoi(e);
    if (ng == 5) launch<T, TA, NEPShapeDefault, 5>(dd, wk);
    else if (ng == 3) launch<T, TA, NEPShapeDefault, 3>(dd, wk);
    else if (ng == 2) launch<T, TA, NEPShapeDefault, 2>(dd, wk);
    else launch<T, TA, NEPShapeDefault, 1>(dd, wk);
  } else {
    launch<T, TA, NEPShapeGeneric, 1>(dd, wk);
  }
}


template <class DeviceType>
template <class T, class TA, class Shape, int NG>
void NEPKokkosEngine<DeviceType>::launch(NEP_Data<DeviceType, T, TA> &dd, NEP_Work<DeviceType, T, TA> &wk)
{
  NEP_Compute_Functor<DeviceType, T, TA, Shape, NG> functor;
  functor.dims = dims;
  functor.dims.no_moments = std::getenv("NEP_KK_NOMOM") != nullptr;    // testing aid
  functor.D = dd;
  functor.Wk = wk;
  functor.x = x;
  functor.f = f;
  functor.v_virial = d_vatom;
  functor.type = type;
  functor.type_map = k_type_map;
  functor.rcsq_rad = k_rcsq_rad;
  functor.rcsq_ang = k_rcsq_ang;
  functor.d_neighbors = d_neighbors;
  functor.d_numneigh = d_numneigh;
  functor.d_ilist = d_ilist;
  functor.d_neighbors_short_radial = d_neighbors_short_radial;
  functor.d_numneigh_short_radial = d_numneigh_short_radial;
  functor.d_neighbors_short_angular = d_neighbors_short_angular;
  functor.d_numneigh_short_angular = d_numneigh_short_angular;
  functor.d_neighbors_short_zbl = d_neighbors_short_zbl;
  functor.d_numneigh_short_zbl = d_numneigh_short_zbl;
  functor.eatom = d_eatom;
  functor.need_virial = need_virial;
  functor.zbl_on = zbl_enabled && std::getenv("NEP_DBG_NOZBL") == nullptr;    // NOZBL: timing diagnostic only

  functor.natoms = inum;
  if constexpr (nep_is_df64<T>::value) {
    if ((int) k_xdf.extent(0) < nall)
      k_xdf = Kokkos::View<df64 *[3], Kokkos::LayoutRight, DeviceType>("nep:xdf", (size_t)(nall * 1.1) + 16);
    NepXdfFunctor<DeviceType> cx;
    cx.x = x;
    cx.xdf = k_xdf;
    Kokkos::parallel_for("NepXdf", Kokkos::RangePolicy<DeviceType>(0, nall), cx);
    functor.xdf = k_xdf;
  }
  {
    if ((int) k_ntype.extent(0) < nall) k_ntype = Kokkos::View<int *, DeviceType>("nep:ntype", (size_t)(nall * 1.1) + 16);
    NepTypeFunctor<DeviceType> tf;
    tf.type = type;
    tf.type_map = k_type_map;
    tf.ntype = k_ntype;
    Kokkos::parallel_for("NepType", Kokkos::RangePolicy<DeviceType>(0, nall), tf);
    functor.ntype = k_ntype;
  }
  static const bool fscr_off = std::getenv("NEP_KK_NOFSCR") != nullptr;    // testing aid
  const bool use_fscr = (sizeof(T) == 4) && !fscr_off;
  if (use_fscr) {
    if ((int) k_fscr.extent(0) < nall)
      k_fscr = Kokkos::View<float *[4], Kokkos::LayoutRight, DeviceType>("nep:fscr", (size_t)(nall * 1.1) + 16);
    Kokkos::deep_copy(k_fscr, 0.0f);
    functor.fscr = k_fscr;
    functor.use_fscr = true;
  }
  functor.rcf_rad = k_rcf_rad;
  functor.rcf_ang = k_rcf_ang;
  {
    static const bool screen64 = std::getenv("NEP_KK_SCREEN64") != nullptr;    // testing aid: double screening
    functor.screen32 = !screen64;
    if (functor.screen32) {
      if ((int) k_x32.extent(0) < nall) k_x32 = Kokkos::View<float *[3], Kokkos::LayoutRight, DeviceType>("nep:x32", (size_t)(nall * 1.1) + 16);
      NepX32Functor<DeviceType> cvt;
      cvt.x = x;
      cvt.x32 = k_x32;
      Kokkos::parallel_for("NepX32", Kokkos::RangePolicy<DeviceType>(0, nall), cvt);
      functor.x32 = k_x32;
      // float rounding of coordinates (|x| <= ~2L incl. ghosts) is covered by this margin (Angstrom)
      const double L = box_length;
      functor.screen_margin = (float) (1e-6 * L + 2e-4);
    }
  }

  // lanes (adjacent threads) per atom: GPU only; host backends always use one thread per atom
  constexpr bool on_host = Kokkos::SpaceAccessibility<DeviceType, Kokkos::HostSpace>::accessible;
  // per-kernel lane counts (tuned on RTX PRO 6000 Blackwell, 25k atoms); env overrides for tuning,
  // read once.  Host backends always use one thread per atom.
  struct LaneConfig {
    int descR, descA, forceR, forceA, neigh, ann;
    explicit LaneConfig(const int mode)    // 0 fp64, 1 fp32, 2 df64
    {
      descR = (mode == 2) ? 8 : 16; descA = (mode == 1 || mode == 2) ? 1 : 4; forceR = 16; forceA = 16; neigh = 8; ann = 4;
      auto env = [](const char *name, int &v) { if (const char *e = std::getenv(name)) v = atoi(e); };
      env("NEP_KK_VDR", descR); env("NEP_KK_VDA", descA); env("NEP_KK_VFR", forceR);
      env("NEP_KK_VFA", forceA); env("NEP_KK_VN", neigh); env("NEP_KK_VANN", ann);
    }
  };
  static const LaneConfig lane_cfg(nep_is_df64<T>::value ? 2 : (sizeof(T) == 4 ? 1 : 0));
  static const LaneConfig lane_cfg_ang(0);    // df64 mode: the angular kernels are native double
  int v_descR = lane_cfg.descR, v_descA = lane_cfg.descA;
  int v_forceR = lane_cfg.forceR, v_forceA = nep_is_df64<T>::value ? lane_cfg_ang.forceA : lane_cfg.forceA;
  int v_neigh = lane_cfg.neigh, v_ann = lane_cfg.ann;
#ifndef KOKKOS_ENABLE_CUDA
  v_neigh = 1;    // the ballot-compacted neighbor build is CUDA-only
#endif
  if (on_host) v_descR = v_descA = v_forceR = v_forceA = v_neigh = v_ann = 1;

  // run fn(std::integral_constant<int,V>) for a runtime V in {1,2,4,8,16,32}
  auto dispatchV = [&](const int v, auto &&fn) {
    switch (v) {
      case 1: fn(std::integral_constant<int, 1>()); break;
      case 2: fn(std::integral_constant<int, 2>()); break;
      case 4: fn(std::integral_constant<int, 4>()); break;
      case 8: fn(std::integral_constant<int, 8>()); break;
      case 16: fn(std::integral_constant<int, 16>()); break;
      case 32: fn(std::integral_constant<int, 32>()); break;
      default: error->all(FLERR, "NEP_KK_VR/VA must be one of 1,2,4,8,16,32");
    }
  };
  // thread count: whole warps so that shuffle reductions never see a partially active warp
  auto nthreads = [&](const int v) { return ((inum * v + 31) / 32) * 32; };

  nep_kk_prof().start();
  dispatchV(v_neigh, [&](auto vc) {
    constexpr int V = decltype(vc)::value;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagNeighbor<V>>(0, nthreads(V)), functor);
  });
  nep_kk_prof().stop(0);

  nep_kk_prof().start();
  dispatchV(v_descR, [&](auto vc) {
    constexpr int V = decltype(vc)::value;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagDescRadial<V>>(0, nthreads(V)), functor);
  });
  nep_kk_prof().stop(1);

  // angular kernels: native double (the df64 functor is only used for the radial part)
  auto run_descA = [&](auto &fn) {
    dispatchV(v_descA, [&](auto vc) {
      constexpr int V = decltype(vc)::value;
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagDescAngular<V>>(0, nthreads(V)), fn);
    });
  };
  auto run_fang = [&](auto &fn) {
    dispatchV(v_forceA, [&](auto vc) {
      constexpr int V = decltype(vc)::value;
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagForceAngular<V>>(0, nthreads(V)), fn);
    });
  };
  NEP_Compute_Functor<DeviceType, double, double, Shape, 5> fang;
  if constexpr (nep_is_df64<T>::value) {
    fang.dims = dims;
    fang.D = data_d;
    fang.Wk = work_d;
    fang.x = x;
    fang.f = f;
    fang.v_virial = d_vatom;
    fang.type = type;
    fang.type_map = k_type_map;
    fang.ntype = functor.ntype;
    fang.d_ilist = d_ilist;
    fang.d_numneigh = d_numneigh;
    fang.d_neighbors = d_neighbors;
    fang.d_neighbors_short_angular = d_neighbors_short_angular;
    fang.d_numneigh_short_angular = d_numneigh_short_angular;
    fang.eatom = d_eatom;
    fang.need_virial = need_virial;
    fang.natoms = inum;
    static const bool no_hyb = std::getenv("NEP_KK_NOHYB") != nullptr;    // testing aid: angular kernels fully native double
    fang.hyb = !no_hyb;
    fang.xdf = k_xdf;
  }

  nep_kk_prof().start();
  if (dims.L_max > 0) {
    run_descA(functor);    // df64 mode: the angular descriptor runs fully in double-float (faster than the hybrid)
  }
  nep_kk_prof().stop(2);

  nep_kk_prof().start();
  dispatchV(v_ann, [&](auto vc) {
    constexpr int V = decltype(vc)::value;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagANN<V>>(0, nthreads(V)), functor);
  });
  nep_kk_prof().stop(3);

  nep_kk_prof().start();
  dispatchV(v_forceR, [&](auto vc) {
    constexpr int V = decltype(vc)::value;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagForceRadial<V>>(0, nthreads(V)), functor);
  });
  if (zbl_enabled) Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagZBL>(0, inum), functor);
  nep_kk_prof().stop(4);

  nep_kk_prof().start();
  if (dims.L_max > 0) {
    if constexpr (nep_is_df64<T>::value) run_fang(fang);
    else run_fang(functor);
  }
  nep_kk_prof().stop(5);

  if (use_fscr) {
    nep_kk_prof().start();
    NepFoldFunctor<DeviceType> fold;
    fold.f = f;
    fold.fscr = k_fscr;
    Kokkos::parallel_for("NepFold", Kokkos::RangePolicy<DeviceType>(0, nall), fold);
    nep_kk_prof().stop(7);
  }
}


}    // namespace LAMMPS_NS

#endif
