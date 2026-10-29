/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/
   Copyright (2003) Sandia Corporation.
------------------------------------------------------------------------- */

// Device-side helpers for the NEP Kokkos pair style.
//
// Everything here is templated on the compute type T (double or float) and uses explicitly typed
// constants, so that float instantiations never silently promote to double (FP64 runs at 1/64 of
// the FP32 rate on most GPUs).  Per-atom data flow (see pair_nep_kokkos.h):
//   descriptor (radial q, angular s -> q)  ->  ANN (Fp, then A and W)  ->  radial force, angular force
// where A[n][abc] = dE_i/ds[n][abc] folds the 3-, 4- and 5-body chain rule terms so that the angular
// force needs the spherical harmonics only once per neighbor (not once per radial index n).

#ifndef LMP_NEP_KOKKOS_H
#define LMP_NEP_KOKKOS_H

#include "kokkos_type.h"
#include "nep_df64.h"
#include <cmath>


namespace LAMMPS_NS {

static constexpr double NEP_KK_PI = 3.141592653589793;

constexpr int NEP_NUM_OF_ABC = 80;    // 3+5+7+...+17 for L_max = 8
constexpr int NEP_MAX_BASIS1 = 18;    // max basis_size + 1
constexpr int NEP_MAX_DIM = 96;       // max supported descriptor dimension

// ----------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION double get_z_coefficient(int L, int m, int n)
{
  const double Z_COEFFICIENT_8[9][9] = {
      {35.0, 0.0, -1260.0, 0.0, 6930.0, 0.0, -12012.0, 0.0, 6435.0},
      {0.0, -35.0, 0.0, 385.0, 0.0, -1001.0, 0.0, 715.0, 0.0},
      {-1.0, 0.0, 33.0, 0.0, -143.0, 0.0, 143.0, 0.0, 0.0},
      {0.0, 3.0, 0.0, -26.0, 0.0, 39.0, 0.0, 0.0, 0.0},
      {1.0, 0.0, -26.0, 0.0, 65.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, -1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-1.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  const double Z_COEFFICIENT_1[2][2] = {{0.0, 1.0}, {1.0, 0.0}};
  const double Z_COEFFICIENT_2[3][3] = {{-1.0, 0.0, 3.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}};
  const double Z_COEFFICIENT_3[4][4] = {
      {0.0, -3.0, 0.0, 5.0}, {-1.0, 0.0, 5.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}};
  const double Z_COEFFICIENT_4[5][5] = {{3.0, 0.0, -30.0, 0.0, 35.0},
                                        {0.0, -3.0, 0.0, 7.0, 0.0},
                                        {-1.0, 0.0, 7.0, 0.0, 0.0},
                                        {0.0, 1.0, 0.0, 0.0, 0.0},
                                        {1.0, 0.0, 0.0, 0.0, 0.0}};
  const double Z_COEFFICIENT_5[6][6] = {
      {0.0, 15.0, 0.0, -70.0, 0.0, 63.0}, {1.0, 0.0, -14.0, 0.0, 21.0, 0.0},
      {0.0, -1.0, 0.0, 3.0, 0.0, 0.0},    {-1.0, 0.0, 9.0, 0.0, 0.0, 0.0},
      {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},     {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  const double Z_COEFFICIENT_6[7][7] = {
      {-5.0, 0.0, 105.0, 0.0, -315.0, 0.0, 231.0}, {0.0, 5.0, 0.0, -30.0, 0.0, 33.0, 0.0},
      {1.0, 0.0, -18.0, 0.0, 33.0, 0.0, 0.0},      {0.0, -3.0, 0.0, 11.0, 0.0, 0.0, 0.0},
      {-1.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0},       {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  const double Z_COEFFICIENT_7[8][8] = {{0.0, -35.0, 0.0, 315.0, 0.0, -693.0, 0.0, 429.0},
                                        {-5.0, 0.0, 135.0, 0.0, -495.0, 0.0, 429.0, 0.0},
                                        {0.0, 15.0, 0.0, -110.0, 0.0, 143.0, 0.0, 0.0},
                                        {3.0, 0.0, -66.0, 0.0, 143.0, 0.0, 0.0, 0.0},
                                        {0.0, -3.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0},
                                        {-1.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                        {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  switch (L) {
    case 1: return Z_COEFFICIENT_1[m][n];
    case 2: return Z_COEFFICIENT_2[m][n];
    case 3: return Z_COEFFICIENT_3[m][n];
    case 4: return Z_COEFFICIENT_4[m][n];
    case 5: return Z_COEFFICIENT_5[m][n];
    case 6: return Z_COEFFICIENT_6[m][n];
    case 7: return Z_COEFFICIENT_7[m][n];
    case 8: return Z_COEFFICIENT_8[m][n];
    default: return 0.0;
  }
}

KOKKOS_INLINE_FUNCTION double get_C3B(int i)
{
  const double C3B[80] = {
      0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435,
      0.596831036594608, 0.596831036594608, 0.149207759148652, 0.149207759148652,
      0.139260575205408, 0.104445431404056, 0.104445431404056, 1.044454314040563,
      1.044454314040563, 0.174075719006761, 0.174075719006761, 0.011190581936149,
      0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
      1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606,
      0.013677377921960, 0.102580334414698, 0.102580334414698, 2.872249363611549,
      2.872249363611549, 0.119677056817148, 0.119677056817148, 2.154187022708661,
      2.154187022708661, 0.215418702270866, 0.215418702270866, 0.004041043476943,
      0.169723826031592, 0.169723826031592, 0.106077391269745, 0.106077391269745,
      0.424309565078979, 0.424309565078979, 0.127292869523694, 0.127292869523694,
      2.800443129521260, 2.800443129521260, 0.233370260793438, 0.233370260793438,
      0.004662742473395, 0.004079899664221, 0.004079899664221, 0.024479397985326,
      0.024479397985326, 0.012239698992663, 0.012239698992663, 0.538546755677165,
      0.538546755677165, 0.134636688919291, 0.134636688919291, 3.500553911901575,
      3.500553911901575, 0.250039565135827, 0.250039565135827, 0.000082569397966,
      0.005944996653579, 0.005944996653579, 0.104037441437634, 0.104037441437634,
      0.762941237209318, 0.762941237209318, 0.114441185581398, 0.114441185581398,
      5.950941650232678, 5.950941650232678, 0.141689086910302, 0.141689086910302,
      4.250672607309055, 4.250672607309055, 0.265667037956816, 0.265667037956816};
  return C3B[i];
}

KOKKOS_INLINE_FUNCTION double get_C4B(int i)
{
  const double C4B[5] = {-0.007499480826664, -0.134990654879954, 0.067495327439977,
                         0.404971964639861, -0.809943929279723};
  return C4B[i];
}

KOKKOS_INLINE_FUNCTION double get_C5B(int i)
{
  const double C5B[3] = {0.026596810706114, 0.053193621412227, 0.026596810706114};
  return C5B[i];
}

// ----------------------------------------------------------------------
// Math helpers
// ----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION void nep_sincos(const double x, double &s, double &c)
{
#if defined(__CUDA_ARCH__)
  sincos(x, &s, &c);
#else
  s = sin(x);
  c = cos(x);
#endif
}

KOKKOS_INLINE_FUNCTION void nep_sincos(const float x, float &s, float &c)
{
#if defined(__CUDA_ARCH__)
  sincosf(x, &s, &c);
#else
  s = sinf(x);
  c = cosf(x);
#endif
}

// sin(pi*x), cos(pi*x): CUDA's sincospi skips the multiplication by pi and uses a cheaper reduction
KOKKOS_INLINE_FUNCTION void nep_sincospi(const double x, double &s, double &c)
{
#if defined(__CUDA_ARCH__)
  sincospi(x, &s, &c);
#else
  s = sin(NEP_KK_PI * x);
  c = cos(NEP_KK_PI * x);
#endif
}

KOKKOS_INLINE_FUNCTION void nep_sincospi(const float x, float &s, float &c)
{
#if defined(__CUDA_ARCH__)
  sincospif(x, &s, &c);
#else
  s = sinf((float) NEP_KK_PI * x);
  c = cosf((float) NEP_KK_PI * x);
#endif
}

KOKKOS_INLINE_FUNCTION double nep_cospi(const double x)
{
#if defined(__CUDA_ARCH__)
  return cospi(x);
#else
  return cos(NEP_KK_PI * x);
#endif
}

KOKKOS_INLINE_FUNCTION float nep_cospi(const float x)
{
#if defined(__CUDA_ARCH__)
  return cospif(x);
#else
  return cosf((float) NEP_KK_PI * x);
#endif
}

// 1/sqrt(x): CUDA's rsqrt is cheaper than sqrt + reciprocal when both d and 1/d are needed (<= 1 ulp)
KOKKOS_INLINE_FUNCTION double nep_rsqrt(const double x)
{
#if defined(__CUDA_ARCH__)
  return rsqrt(x);
#else
  return 1.0 / sqrt(x);
#endif
}

KOKKOS_INLINE_FUNCTION float nep_rsqrt(const float x)
{
#if defined(__CUDA_ARCH__)
  return rsqrtf(x);
#else
  return 1.0f / sqrtf(x);
#endif
}

KOKKOS_INLINE_FUNCTION df64 nep_rsqrt(const df64 &x) { return df_rsqrt(x); }

// d12 and 1/d12 from d12^2
template <class T> KOKKOS_INLINE_FUNCTION void nep_dist(const T d2, T &d12, T &d12inv)
{
  d12inv = nep_rsqrt(d2);
  d12 = d2 * d12inv;
}

// Scatter -f12 onto a float scratch row (x, y, z, pad).  On sm_90+ one 16-byte vector atomic replaces three
// scalar atomics (one L2 transaction instead of three).
KOKKOS_INLINE_FUNCTION void nep_scatter_sub_f32(float *row, const float fx, const float fy, const float fz)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  atomicAdd(reinterpret_cast<float4 *>(row), make_float4(-fx, -fy, -fz, 0.0f));
#else
  Kokkos::atomic_sub(row + 0, fx);
  Kokkos::atomic_sub(row + 1, fy);
  Kokkos::atomic_sub(row + 2, fz);
#endif
}

// Sum a value across the V adjacent threads (lanes) that cooperate on one atom.  Every lane
// receives the total.  V must be a power of two <= 32 and V consecutive thread ids must map to
// consecutive hardware lanes (true for Kokkos RangePolicy on CUDA/HIP).  V == 1 (and every host
// backend) is the identity.
template <int V, class T> KOKKOS_INLINE_FUNCTION T nep_vsum(T v)
{
#if defined(__CUDA_ARCH__)
#pragma unroll
  for (int o = V / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
#elif defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
  for (int o = V / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
#endif
  return v;
}

KOKKOS_INLINE_FUNCTION void nep_sincospi(const df64 &x, df64 &s, df64 &c) { df_sincospi(x, s, c); }
KOKKOS_INLINE_FUNCTION df64 nep_cospi(const df64 &x) { return df_cospi(x); }

// ----------------------------------------------------------------------
// Cutoff and radial basis functions
// ----------------------------------------------------------------------
template <class T>
KOKKOS_INLINE_FUNCTION void nep_fc(const T rc, const T rcinv, const T d12, T &fc)
{
  if (d12 < rc) {
    const T x = d12 * rcinv;
    fc = T(0.5) * nep_cospi(x) + T(0.5);
  } else {
    fc = T(0.0);
  }
}

template <class T>
KOKKOS_INLINE_FUNCTION void nep_fc_fcp(const T rc, const T rcinv, const T d12, T &fc, T &fcp)
{
  if (d12 < rc) {
    const T x = d12 * rcinv;
    T s, c;
    nep_sincospi(x, s, c);
    fc = T(0.5) * c + T(0.5);
    fcp = T(-0.5) * T(NEP_KK_PI) * s * rcinv;
  } else {
    fc = T(0.0);
    fcp = T(0.0);
  }
}

// fn[0..nb]: Chebyshev radial basis times cutoff (nb = basis_size)
template <class T>
KOKKOS_INLINE_FUNCTION void nep_fn(const int nb, const T rcinv, const T d12, const T fc12, T *fn)
{
  const T t = d12 * rcinv - T(1.0);
  const T x = T(2.0) * t * t - T(1.0);
  fn[0] = T(1.0);
  if (nb > 0) fn[1] = x;
  for (int m = 2; m <= nb; ++m) fn[m] = T(2.0) * x * fn[m - 1] - fn[m - 2];
  for (int m = 0; m <= nb; ++m) fn[m] = (fn[m] + T(1.0)) * T(0.5) * fc12;
}

// exact small constants (the Z coefficients are integers): df64 from a float avoids a run-time double -> df64 split
template <class T> KOKKOS_FORCEINLINE_FUNCTION T nep_zc(const double v)
{
  if constexpr (nep_is_df64<T>::value) return df64((float) v);
  else return T(v);
}

// sum of products acc += a*b.  For df64 the running sum is kept un-normalized (two_sum + one correction add,
// ~12 FP32 flops instead of ~19 for a full df mul + df add); get() renormalizes.
template <class T> struct nep_acc {
  T v;
  KOKKOS_FORCEINLINE_FUNCTION nep_acc() : v(T(0.0)) {}
  KOKKOS_FORCEINLINE_FUNCTION void add(const T &a, const T &b) { v += a * b; }
  KOKKOS_FORCEINLINE_FUNCTION T get() const { return v; }
};
template <> struct nep_acc<df64> {
  float s, c;
  KOKKOS_FORCEINLINE_FUNCTION nep_acc() : s(0.0f), c(0.0f) {}
  KOKKOS_FORCEINLINE_FUNCTION void add(const df64 &a, const df64 &b)
  {
    float p, e, t;
    nepdf::two_prod(a.hi, b.hi, p, e);
    e = nepdf::ffma(a.hi, b.lo, nepdf::ffma(a.lo, b.hi, e));
    nepdf::two_sum(s, p, s, t);
    c = nepdf::fadd(c, nepdf::fadd(t, e));
  }
  KOKKOS_FORCEINLINE_FUNCTION df64 get() const
  {
    float h, l;
    nepdf::quick_two_sum(s, c, h, l);
    return df64(h, l);
  }
};

// T_k(x), x = 2t^2-1, t = d*rcinv-1  (k = 0..nb)
template <class T>
KOKKOS_INLINE_FUNCTION void nep_cheb(const int nb, const T rcinv, const T d12, T *Tk)
{
  const T t = d12 * rcinv - T(1.0);
  const T x = T(2.0) * t * t - T(1.0);
  Tk[0] = T(1.0);
  if (nb > 0) Tk[1] = x;
  for (int m = 2; m <= nb; ++m) Tk[m] = T(2.0) * x * Tk[m - 1] - Tk[m - 2];
}

template <class T>
KOKKOS_INLINE_FUNCTION void nep_fn_fnp(const int nb, const T rcinv, const T d12, const T fc12,
                                       const T fcp12, T *fn, T *fnp)
{
  const T t = d12 * rcinv - T(1.0);
  const T x = T(2.0) * t * t - T(1.0);
  const T xp = T(4.0) * t * rcinv;
  fn[0] = T(1.0);
  fnp[0] = T(0.0);
  if (nb > 0) {
    fn[1] = x;
    fnp[1] = xp;
  }
  for (int m = 2; m <= nb; ++m) {
    fn[m] = T(2.0) * x * fn[m - 1] - fn[m - 2];
    fnp[m] = T(2.0) * xp * fn[m - 1] + T(2.0) * x * fnp[m - 1] - fnp[m - 2];
  }
  for (int m = 0; m <= nb; ++m) {
    fnp[m] = fnp[m] * T(0.5) * fc12 + (fn[m] + T(1.0)) * T(0.5) * fcp12;
    fn[m] = (fn[m] + T(1.0)) * T(0.5) * fc12;
  }
}

// ----------------------------------------------------------------------
// Radial force factor from the pre-contracted coefficients W[0..nb] (+ Wsum = sum_k W_k):
//   gp = sum_k W_k dfn_k/dd,  fn_k = 0.5*(T_k(x)+1)*fc,  x = 2t^2-1,  t = d*rcinv-1
//      = 0.5*fc*xp*P'(x) + 0.5*fcp*(P(x)+Wsum),   P(x) = sum_k W_k T_k(x), xp = dx/dd = 4*t*rcinv
// P and P' are evaluated together with a Clenshaw recurrence (about half the work of building the
// fn/fnp arrays and contracting them).
// ----------------------------------------------------------------------
template <class T, class ViewW>
KOKKOS_INLINE_FUNCTION T nep_radial_gp(const int nb, const T rcinv, const T d12, const T fc12,
                                       const T fcp12, const ViewW &W, const int i, const int jt)
{
  // W layout: [0..nb] = W_k, [nb+1] = sum_k W_k, [nb+2+(k-1)] = k*W_k (k=1..nb)
  // P(x) = sum W_k T_k(x) (Clenshaw), P'(x) = sum k W_k U_{k-1}(x) (Clenshaw for U), same loop.
  const T t = d12 * rcinv - T(1.0);
  const T x = T(2.0) * t * t - T(1.0);
  const T xp = T(4.0) * t * rcinv;
  const T x2 = T(2.0) * x;
  T b1 = T(0.0), b2 = T(0.0), e1 = T(0.0), e2 = T(0.0);
  for (int k = nb; k >= 1; --k) {
    const T bk = W(i, jt, k) + x2 * b1 - b2;
    const T ek = W(i, jt, nb + 1 + k) + x2 * e1 - e2;
    b2 = b1;
    b1 = bk;
    e2 = e1;
    e1 = ek;
  }
  const T P = W(i, jt, 0) + x * b1 - b2;
  return T(0.5) * fc12 * xp * e1 + T(0.5) * fcp12 * (P + W(i, jt, nb + 1));
}

// ----------------------------------------------------------------------
// Real spherical-harmonic-like basis Y_abc(u) for L = 1..L_max, evaluated once per neighbor.
// Layout: abc = L*L-1 + {0 | 2*n1-1, 2*n1}  (m=0, then (cos,sin) pairs for m=n1)
// ----------------------------------------------------------------------
template <int L, class T>
KOKKOS_INLINE_FUNCTION void nep_Y_L(const T ux, const T uy, const T uz, T *Y)
{
  constexpr int base = L * L - 1;
  T zp[L + 1], cn[L + 1], sn[L + 1];
  zp[0] = T(1.0);
  cn[0] = T(1.0);
  sn[0] = T(0.0);
  for (int n = 1; n <= L; ++n) {
    zp[n] = uz * zp[n - 1];
    cn[n] = ux * cn[n - 1] - uy * sn[n - 1];
    sn[n] = ux * sn[n - 1] + uy * cn[n - 1];
  }
  for (int n1 = 0; n1 <= L; ++n1) {
    const int n2_start = ((L + n1) % 2 == 0) ? 0 : 1;
    T zf = T(0.0);
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) zf += nep_zc<T>(get_z_coefficient(L, n1, n2)) * zp[n2];
    if (n1 == 0) {
      Y[base] = zf;
    } else {
      Y[base + 2 * n1 - 1] = zf * cn[n1];
      Y[base + 2 * n1] = zf * sn[n1];
    }
  }
}

template <class T>
KOKKOS_INLINE_FUNCTION void nep_Y(const int L_max, const T ux, const T uy, const T uz, T *Y)
{
  if (L_max >= 1) nep_Y_L<1>(ux, uy, uz, Y);
  if (L_max >= 2) nep_Y_L<2>(ux, uy, uz, Y);
  if (L_max >= 3) nep_Y_L<3>(ux, uy, uz, Y);
  if (L_max >= 4) nep_Y_L<4>(ux, uy, uz, Y);
  if (L_max >= 5) nep_Y_L<5>(ux, uy, uz, Y);
  if (L_max >= 6) nep_Y_L<6>(ux, uy, uz, Y);
  if (L_max >= 7) nep_Y_L<7>(ux, uy, uz, Y);
  if (L_max >= 8) nep_Y_L<8>(ux, uy, uz, Y);
}

// ----------------------------------------------------------------------
// Invariants q from the angular sums s (one radial index n)
// ----------------------------------------------------------------------
template <int L, class T> KOKKOS_INLINE_FUNCTION T nep_find_q_one(const T *s)
{
  constexpr int start_index = L * L - 1;
  constexpr int num_terms = 2 * L + 1;
  T q = T(0.0);
  for (int k = 1; k < num_terms; ++k)
    q += T(get_C3B(start_index + k)) * s[start_index + k] * s[start_index + k];
  q *= T(2.0);
  q += T(get_C3B(start_index)) * s[start_index] * s[start_index];
  return q;
}

// q is indexed [slot * NA + n]
template <class T>
KOKKOS_INLINE_FUNCTION void nep_find_q(const int L_max, const int num_L, const int NA, const int n,
                                       const T *s, T *q)
{
  if (L_max >= 1) q[0 * NA + n] = nep_find_q_one<1>(s);
  if (L_max >= 2) q[1 * NA + n] = nep_find_q_one<2>(s);
  if (L_max >= 3) q[2 * NA + n] = nep_find_q_one<3>(s);
  if (L_max >= 4) q[3 * NA + n] = nep_find_q_one<4>(s);
  if (L_max >= 5) q[4 * NA + n] = nep_find_q_one<5>(s);
  if (L_max >= 6) q[5 * NA + n] = nep_find_q_one<6>(s);
  if (L_max >= 7) q[6 * NA + n] = nep_find_q_one<7>(s);
  if (L_max >= 8) q[7 * NA + n] = nep_find_q_one<8>(s);

  if (num_L >= L_max + 1) {
    q[L_max * NA + n] = T(get_C4B(0)) * s[3] * s[3] * s[3] +
        T(get_C4B(1)) * s[3] * (s[4] * s[4] + s[5] * s[5]) +
        T(get_C4B(2)) * s[3] * (s[6] * s[6] + s[7] * s[7]) +
        T(get_C4B(3)) * s[6] * (s[5] * s[5] - s[4] * s[4]) + T(get_C4B(4)) * s[4] * s[5] * s[7];
  }
  if (num_L >= L_max + 2) {
    const T s0_sq = s[0] * s[0];
    const T s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];
    q[(L_max + 1) * NA + n] = T(get_C5B(0)) * s0_sq * s0_sq +
        T(get_C5B(1)) * s0_sq * s1_sq_plus_s2_sq + T(get_C5B(2)) * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq;
  }
}

// ----------------------------------------------------------------------
// Per-atom, per-n angular coefficients A[abc] = dE_i/ds[abc]   (s and a for one n)
//   3-body (each L): q_L = C3B0 s0^2 + 2*sum C3B s_k^2  -> dq/ds = 2*C3B*s (x2 for m>0)
//   4-body: derivative of the cubic q4 w.r.t. the five L=2 components (s[3..7])
//   5-body: derivative of the quartic q5 w.r.t. the three L=1 components (s[0..2])
// fp_ang[slot*NA + n] are the angular Fp values.
// ----------------------------------------------------------------------
template <class T>
KOKKOS_INLINE_FUNCTION void nep_make_A(const int L_max, const int num_L, const int NA, const int n,
                                       const T *fp_ang, const T *s, T *a)
{
  const int dim_s = (L_max + 1) * (L_max + 1) - 1;
  for (int abc = 0; abc < dim_s; ++abc) a[abc] = T(0.0);
  for (int L = 1; L <= L_max; ++L) {
    const int base = L * L - 1;
    T fpf = T(2.0) * fp_ang[(L - 1) * NA + n];
    a[base] = s[base] * T(get_C3B(base)) * fpf;
    fpf *= T(2.0);
    for (int k = 1; k < 2 * L + 1; ++k) a[base + k] = s[base + k] * T(get_C3B(base + k)) * fpf;
  }
  if (num_L >= L_max + 1) {
    const T F4 = fp_ang[L_max * NA + n];
    const T s3 = s[3], s4 = s[4], s5 = s[5], s6 = s[6], s7 = s[7];
    a[3] += F4 * (T(get_C4B(0)) * T(3.0) * s3 * s3 + T(get_C4B(1)) * (s4 * s4 + s5 * s5) +
                  T(get_C4B(2)) * (s6 * s6 + s7 * s7));
    a[4] += F4 * (T(get_C4B(1)) * s3 * s4 * T(2.0) - T(get_C4B(3)) * s6 * s4 * T(2.0) +
                  T(get_C4B(4)) * s5 * s7);
    a[5] += F4 * (T(get_C4B(1)) * s3 * s5 * T(2.0) + T(get_C4B(3)) * s6 * s5 * T(2.0) +
                  T(get_C4B(4)) * s4 * s7);
    a[6] += F4 * (T(get_C4B(2)) * s3 * s6 * T(2.0) + T(get_C4B(3)) * (s5 * s5 - s4 * s4));
    a[7] += F4 * (T(get_C4B(2)) * s3 * s7 * T(2.0) + T(get_C4B(4)) * s4 * s5);
  }
  if (num_L >= L_max + 2) {
    const T F5 = fp_ang[(L_max + 1) * NA + n];
    const T s0 = s[0], s1 = s[1], s2 = s[2];
    const T s12 = s1 * s1 + s2 * s2;
    a[0] += F5 * (T(get_C5B(0)) * T(4.0) * s0 * s0 * s0 + T(get_C5B(1)) * s12 * T(2.0) * s0);
    a[1] += F5 * (T(get_C5B(1)) * s0 * s0 * s1 * T(2.0) + T(get_C5B(2)) * s12 * s1 * T(4.0));
    a[2] += F5 * (T(get_C5B(1)) * s0 * s0 * s2 * T(2.0) + T(get_C5B(2)) * s12 * s2 * T(4.0));
  }
}

// ----------------------------------------------------------------------
// Fused angular force contraction for one neighbor (unit vector u, radial functions gn[n], gnp[n]):
//   f12 = sum_{n,abc} A[n][abc] * d( gn_n(d) * Y_abc(u) ) / d r12
//       = S*u + (G - (G.u)u)/d,
//   S = sum_abc C_abc*Y_abc,  G = sum_abc B_abc*gradY_abc(raw),
//   B_abc = sum_n A[n][abc]*gn[n],  C_abc = sum_n A[n][abc]*gnp[n].
// ----------------------------------------------------------------------
template <int L, class T, class ViewA>
KOKKOS_INLINE_FUNCTION void nep_contract_L(const int NA, const T *gn, const T *gnp, const ViewA &A,
                                           const int i, const T ux, const T uy, const T uz, T &S,
                                           T *G)
{
  constexpr int base = L * L - 1;
  T zp[L + 1], cn[L + 1], sn[L + 1];
  zp[0] = T(1.0);
  cn[0] = T(1.0);
  sn[0] = T(0.0);
  for (int n = 1; n <= L; ++n) {
    zp[n] = uz * zp[n - 1];
    cn[n] = ux * cn[n - 1] - uy * sn[n - 1];
    sn[n] = ux * sn[n - 1] + uy * cn[n - 1];
  }
  for (int n1 = 0; n1 <= L; ++n1) {
    const int n2_start = ((L + n1) % 2 == 0) ? 0 : 1;
    T zf = T(0.0), zfp = T(0.0);
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      const T zc = nep_zc<T>(get_z_coefficient(L, n1, n2));
      zf += zc * zp[n2];
      if (n2 > 0) zfp += zc * T(n2) * zp[n2 - 1];
    }
    if (n1 == 0) {
      nep_acc<T> aB, aC;
      for (int n = 0; n < NA; ++n) {
        const T a = A(i, n, base);
        aB.add(a, gn[n]);
        aC.add(a, gnp[n]);
      }
      const T B = aB.get(), C = aC.get();
      S += C * zf;
      G[2] += B * zfp;
    } else {
      const int ire = base + 2 * n1 - 1;
      nep_acc<T> aBre, aCre, aBim, aCim;
      for (int n = 0; n < NA; ++n) {
        const T are = A(i, n, ire);
        const T aim = A(i, n, ire + 1);
        aBre.add(are, gn[n]);
        aCre.add(are, gnp[n]);
        aBim.add(aim, gn[n]);
        aCim.add(aim, gnp[n]);
      }
      const T Bre = aBre.get(), Cre = aCre.get(), Bim = aBim.get(), Cim = aCim.get();
      S += zf * (Cre * cn[n1] + Cim * sn[n1]);
      const T nz = T(n1) * zf;
      G[0] += nz * (Bre * cn[n1 - 1] + Bim * sn[n1 - 1]);
      G[1] += nz * (Bim * cn[n1 - 1] - Bre * sn[n1 - 1]);
      G[2] += zfp * (Bre * cn[n1] + Bim * sn[n1]);
    }
  }
}

template <class T, class ViewA>
KOKKOS_INLINE_FUNCTION void nep_contract(const int L_max, const int NA, const T *gn, const T *gnp,
                                         const ViewA &A, const int i, const T ux, const T uy,
                                         const T uz, T &S, T *G)
{
  if (L_max >= 1) nep_contract_L<1>(NA, gn, gnp, A, i, ux, uy, uz, S, G);
  if (L_max >= 2) nep_contract_L<2>(NA, gn, gnp, A, i, ux, uy, uz, S, G);
  if (L_max >= 3) nep_contract_L<3>(NA, gn, gnp, A, i, ux, uy, uz, S, G);
  if (L_max >= 4) nep_contract_L<4>(NA, gn, gnp, A, i, ux, uy, uz, S, G);
  if (L_max >= 5) nep_contract_L<5>(NA, gn, gnp, A, i, ux, uy, uz, S, G);
  if (L_max >= 6) nep_contract_L<6>(NA, gn, gnp, A, i, ux, uy, uz, S, G);
  if (L_max >= 7) nep_contract_L<7>(NA, gn, gnp, A, i, ux, uy, uz, S, G);
  if (L_max >= 8) nep_contract_L<8>(NA, gn, gnp, A, i, ux, uy, uz, S, G);
}

// ----------------------------------------------------------------------
// ZBL
// ----------------------------------------------------------------------
template <class T>
KOKKOS_INLINE_FUNCTION void nep_find_fc_and_fcp_zbl(const T r1, const T r2, const T d12, T &fc,
                                                    T &fcp)
{
  if (d12 < r1) {
    fc = T(1.0);
    fcp = T(0.0);
  } else if (d12 < r2) {
    const T pi_factor = T(NEP_KK_PI) / (r2 - r1);
    T s, c;
    nep_sincos(pi_factor * (d12 - r1), s, c);
    fc = c * T(0.5) + T(0.5);
    fcp = -s * pi_factor * T(0.5);
  } else {
    fc = T(0.0);
    fcp = T(0.0);
  }
}

template <class T>
KOKKOS_INLINE_FUNCTION void nep_phi_and_phip_zbl(const T a, const T b, const T x, T &phi, T &phip)
{
  const T tmp = a * exp(-b * x);
  phi += tmp;
  phip -= b * tmp;
}

// universal ZBL: fixed screening-function parameters
template <class T>
KOKKOS_INLINE_FUNCTION void nep_f_and_fp_zbl(const T zizj, const T a_inv, const T rc_inner,
                                             const T rc_outer, const T d12, const T d12inv, T &f,
                                             T &fp)
{
  const T x = d12 * a_inv;
  f = fp = T(0.0);
  nep_phi_and_phip_zbl(T(0.18175), T(3.1998), x, f, fp);
  nep_phi_and_phip_zbl(T(0.50986), T(0.94229), x, f, fp);
  nep_phi_and_phip_zbl(T(0.28022), T(0.4029), x, f, fp);
  nep_phi_and_phip_zbl(T(0.02817), T(0.20162), x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  T fc, fcp;
  nep_find_fc_and_fcp_zbl(rc_inner, rc_outer, d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

// flexible ZBL (matches nep.cpp): zbl_para[0..1] = (r_inner, r_outer), [2..9] = 4 x (A, b)
template <class T>
KOKKOS_INLINE_FUNCTION void nep_f_and_fp_zbl_flex(const T *zbl_para, const T zizj, const T a_inv,
                                                  const T d12, const T d12inv, T &f, T &fp)
{
  const T x = d12 * a_inv;
  f = fp = T(0.0);
  nep_phi_and_phip_zbl(zbl_para[2], zbl_para[3], x, f, fp);
  nep_phi_and_phip_zbl(zbl_para[4], zbl_para[5], x, f, fp);
  nep_phi_and_phip_zbl(zbl_para[6], zbl_para[7], x, f, fp);
  nep_phi_and_phip_zbl(zbl_para[8], zbl_para[9], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  T fc, fcp;
  nep_find_fc_and_fcp_zbl(zbl_para[0], zbl_para[1], d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

}    // namespace LAMMPS_NS

#endif
