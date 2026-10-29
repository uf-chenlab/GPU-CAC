/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/
   Copyright (2003) Sandia Corporation.
------------------------------------------------------------------------- */

// "Double-float" (df64) arithmetic for the NEP Kokkos pair style: a number is the unevaluated sum of two
// floats (hi + lo, |lo| <= ulp(hi)/2), giving a ~48-bit significand (relative error ~1e-14 per operation).
// On GPUs with a 1/64 FP64:FP32 rate (RTX / L4 class) one df64 operation (8-20 FP32 instructions) is several
// times cheaper than one native double operation.
//
// The error-free transformations must not be re-associated or contracted into FMAs by the compiler; on the
// device the CUDA rounding intrinsics (__fadd_rn / __fmul_rn) are used, which are never contracted.  On the
// host the code relies on the default (non fast-math) floating-point model.

#ifndef LMP_NEP_DF64_H
#define LMP_NEP_DF64_H

#ifndef NEP_DF64_STANDALONE
#include "kokkos_type.h"
#endif
#include <cmath>
#include <type_traits>

namespace LAMMPS_NS {

namespace nepdf {

KOKKOS_FORCEINLINE_FUNCTION float fadd(const float a, const float b)
{
#if defined(__CUDA_ARCH__)
  return __fadd_rn(a, b);
#else
  volatile float r = a + b;    // volatile: keep the rounded sum (no excess precision / contraction)
  return r;
#endif
}

KOKKOS_FORCEINLINE_FUNCTION float fmul(const float a, const float b)
{
#if defined(__CUDA_ARCH__)
  return __fmul_rn(a, b);
#else
  volatile float r = a * b;
  return r;
#endif
}

KOKKOS_FORCEINLINE_FUNCTION float ffma(const float a, const float b, const float c)
{
#if defined(__CUDA_ARCH__)
  return __fmaf_rn(a, b, c);
#else
  return fmaf(a, b, c);
#endif
}

KOKKOS_FORCEINLINE_FUNCTION void two_sum(const float a, const float b, float &s, float &e)
{
  s = fadd(a, b);
  const float bb = fadd(s, -a);
  e = fadd(fadd(a, -fadd(s, -bb)), fadd(b, -bb));
}

KOKKOS_FORCEINLINE_FUNCTION void quick_two_sum(const float a, const float b, float &s, float &e)
{
  s = fadd(a, b);
  e = fadd(b, -fadd(s, -a));
}

KOKKOS_FORCEINLINE_FUNCTION void two_prod(const float a, const float b, float &p, float &e)
{
  p = fmul(a, b);
  e = ffma(a, b, -p);
}

}    // namespace nepdf

struct df64 {
  float hi, lo;

  KOKKOS_FORCEINLINE_FUNCTION df64() : hi(0.0f), lo(0.0f) {}
  KOKKOS_FORCEINLINE_FUNCTION df64(const float h, const float l) : hi(h), lo(l) {}
  KOKKOS_FORCEINLINE_FUNCTION df64(const float f) : hi(f), lo(0.0f) {}
  KOKKOS_FORCEINLINE_FUNCTION df64(const int i) : hi((float) i), lo(0.0f) {}
  KOKKOS_FORCEINLINE_FUNCTION df64(const double d)
  {
    hi = (float) d;
    lo = (float) (d - (double) hi);
  }
  KOKKOS_FORCEINLINE_FUNCTION explicit operator double() const { return (double) hi + (double) lo; }
  KOKKOS_FORCEINLINE_FUNCTION explicit operator float() const { return hi + lo; }
};

template <class T> struct nep_is_df64 : std::false_type {};
template <> struct nep_is_df64<df64> : std::true_type {};

KOKKOS_FORCEINLINE_FUNCTION df64 operator-(const df64 &a) { return df64(-a.hi, -a.lo); }

// sloppy add: absolute error ~ 2^-46 (|a| + |b|), which is what matters for the accumulations here
KOKKOS_FORCEINLINE_FUNCTION df64 operator+(const df64 &a, const df64 &b)
{
  float s, e;
  nepdf::two_sum(a.hi, b.hi, s, e);
  e = nepdf::fadd(e, nepdf::fadd(a.lo, b.lo));
  float h, l;
  nepdf::quick_two_sum(s, e, h, l);
  return df64(h, l);
}

KOKKOS_FORCEINLINE_FUNCTION df64 operator-(const df64 &a, const df64 &b) { return a + (-b); }

KOKKOS_FORCEINLINE_FUNCTION df64 operator*(const df64 &a, const df64 &b)
{
  float p, e;
  nepdf::two_prod(a.hi, b.hi, p, e);
  e = nepdf::ffma(a.hi, b.lo, nepdf::ffma(a.lo, b.hi, e));
  float h, l;
  nepdf::quick_two_sum(p, e, h, l);
  return df64(h, l);
}

KOKKOS_FORCEINLINE_FUNCTION df64 &operator+=(df64 &a, const df64 &b)
{
  a = a + b;
  return a;
}
KOKKOS_FORCEINLINE_FUNCTION df64 &operator-=(df64 &a, const df64 &b)
{
  a = a - b;
  return a;
}
KOKKOS_FORCEINLINE_FUNCTION df64 &operator*=(df64 &a, const df64 &b)
{
  a = a * b;
  return a;
}

// df * float (cheaper than df * df)
KOKKOS_FORCEINLINE_FUNCTION df64 mul_f(const df64 &a, const float b)
{
  float p, e;
  nepdf::two_prod(a.hi, b, p, e);
  e = nepdf::ffma(a.lo, b, e);
  float h, l;
  nepdf::quick_two_sum(p, e, h, l);
  return df64(h, l);
}

// a / b (three correction steps)
KOKKOS_FORCEINLINE_FUNCTION df64 operator/(const df64 &a, const df64 &b)
{
  const float q1 = a.hi / b.hi;
  df64 r = a - mul_f(b, q1);
  const float q2 = r.hi / b.hi;
  r = r - mul_f(b, q2);
  const float q3 = r.hi / b.hi;
  float h, l;
  nepdf::quick_two_sum(q1, q2, h, l);
  return df64(h, l) + df64(q3);
}

KOKKOS_FORCEINLINE_FUNCTION bool operator<(const df64 &a, const df64 &b)
{
  return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
KOKKOS_FORCEINLINE_FUNCTION bool operator>(const df64 &a, const df64 &b) { return b < a; }
KOKKOS_FORCEINLINE_FUNCTION bool operator<=(const df64 &a, const df64 &b) { return !(b < a); }
KOKKOS_FORCEINLINE_FUNCTION bool operator>=(const df64 &a, const df64 &b) { return !(a < b); }

// ----------------------------------------------------------------------
// 1/sqrt(a) (two Newton steps from the float estimate) and sqrt via Karp-Markstein
// ----------------------------------------------------------------------
KOKKOS_FORCEINLINE_FUNCTION df64 df_rsqrt(const df64 &a)
{
  float y0;
#if defined(__CUDA_ARCH__)
  y0 = rsqrtf(a.hi);
#else
  y0 = 1.0f / sqrtf(a.hi);
#endif
  df64 y(y0);
  for (int it = 0; it < 2; ++it) {
    // y <- y + 0.5*y*(1 - a*y*y)
    const df64 yy = y * y;
    const df64 r = df64(1.0f) - a * yy;
    y = y + mul_f(y, 0.5f * r.hi);
  }
  return y;
}

// ----------------------------------------------------------------------
// sin(pi*x), cos(pi*x) for x in [0, 1]
// ----------------------------------------------------------------------
namespace nepdf {

// Taylor coefficients, w = z^2:  sin z = z * sum s_k w^k,  cos z = sum c_k w^k   (z <= pi/4)
KOKKOS_FORCEINLINE_FUNCTION double sin_coef(const int k)
{
  const double c[9] = {1.0,
                       -1.0 / 6.0,
                       1.0 / 120.0,
                       -1.0 / 5040.0,
                       1.0 / 362880.0,
                       -1.0 / 39916800.0,
                       1.0 / 6227020800.0,
                       -1.0 / 1307674368000.0,
                       1.0 / 355687428096000.0};
  return c[k];
}

KOKKOS_FORCEINLINE_FUNCTION double cos_coef(const int k)
{
  const double c[9] = {1.0,
                       -1.0 / 2.0,
                       1.0 / 24.0,
                       -1.0 / 720.0,
                       1.0 / 40320.0,
                       -1.0 / 3628800.0,
                       1.0 / 479001600.0,
                       -1.0 / 87178291200.0,
                       1.0 / 20922789888000.0};
  return c[k];
}

// sin(z)/z and cos(z) for 0 <= z <= pi/4 ; terms k >= 5 only need float precision
KOKKOS_FORCEINLINE_FUNCTION df64 series_sin_over_z(const df64 &w)
{
  float tail = (float) sin_coef(8);
  for (int k = 7; k >= 5; --k) tail = ffma(tail, w.hi, (float) sin_coef(k));
  df64 acc(tail);
  for (int k = 4; k >= 0; --k) acc = acc * w + df64(sin_coef(k));
  return acc;
}

KOKKOS_FORCEINLINE_FUNCTION df64 series_cos(const df64 &w)
{
  float tail = (float) cos_coef(8);
  for (int k = 7; k >= 5; --k) tail = ffma(tail, w.hi, (float) cos_coef(k));
  df64 acc(tail);
  for (int k = 4; k >= 0; --k) acc = acc * w + df64(cos_coef(k));
  return acc;
}

}    // namespace nepdf

// sin(pi x) and cos(pi x) with octant reduction: x in [0,1]
KOKKOS_INLINE_FUNCTION void df_sincospi(const df64 &x, df64 &s, df64 &c)
{
  const df64 PI_D(3.141592653589793238462643383279502884);
  bool neg_c = false;
  df64 xr = x;
  if (xr.hi > 0.5f) {
    xr = df64(1.0f) - xr;    // sin(pi x) = sin(pi (1-x)), cos(pi x) = -cos(pi (1-x))
    neg_c = true;
  }
  bool swap = false;
  if (xr.hi > 0.25f) {
    xr = df64(0.5f) - xr;    // sin(pi xr) = cos(pi (0.5-xr)), cos(pi xr) = sin(pi (0.5-xr))
    swap = true;
  }
  const df64 z = xr * PI_D;
  const df64 w = z * z;
  const df64 sn = z * nepdf::series_sin_over_z(w);
  const df64 cs = nepdf::series_cos(w);
  s = swap ? cs : sn;
  c = swap ? sn : cs;
  if (neg_c) c = -c;
}

// cos(pi x) only (single series after octant reduction)
KOKKOS_INLINE_FUNCTION df64 df_cospi(const df64 &x)
{
  const df64 PI_D(3.141592653589793238462643383279502884);
  bool neg = false;
  df64 xr = x;
  if (xr.hi > 0.5f) {
    xr = df64(1.0f) - xr;
    neg = true;
  }
  df64 r;
  if (xr.hi > 0.25f) {
    xr = df64(0.5f) - xr;    // cos(pi xr) = sin(pi (0.5-xr))
    const df64 z = xr * PI_D;
    r = z * nepdf::series_sin_over_z(z * z);
  } else {
    const df64 z = xr * PI_D;
    r = nepdf::series_cos(z * z);
  }
  return neg ? -r : r;
}

}    // namespace LAMMPS_NS

#endif
