#pragma once

#include <tuple>

#include <hpc_vector3.hpp>
#include <hpc_array_traits.hpp>
#include <hpc_tensor_detail.hpp>

namespace hpc {

template <typename Scalar>
class matrix3x3 {
public:
  using scalar_type = Scalar;
private:
  scalar_type raw[3][3];
public:
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3(
      Scalar const a, Scalar const b, Scalar const c,
      Scalar const d, Scalar const e, Scalar const f,
      Scalar const g, Scalar const h, Scalar const i) noexcept
  {
    raw[0][0] = a;
    raw[0][1] = b;
    raw[0][2] = c;
    raw[1][0] = d;
    raw[1][1] = e;
    raw[1][2] = f;
    raw[2][0] = g;
    raw[2][1] = h;
    raw[2][2] = i;
  }
  HPC_ALWAYS_INLINE matrix3x3() noexcept = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr matrix3x3 identity() noexcept {
    return matrix3x3(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr matrix3x3 zero() noexcept {
    return matrix3x3(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr Scalar operator()(int const i, int const j) const noexcept {
    return raw[i][j];
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE Scalar& operator()(int const i, int const j) noexcept {
    return raw[i][j];
  }
};

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(matrix3x3<T> left, matrix3x3<T> right) noexcept {
  return matrix3x3<T>(
      left(0, 0) + right(0, 0),
      left(0, 1) + right(0, 1),
      left(0, 2) + right(0, 2),
      left(1, 0) + right(1, 0),
      left(1, 1) + right(1, 1),
      left(1, 2) + right(1, 2),
      left(2, 0) + right(2, 0),
      left(2, 1) + right(2, 1),
      left(2, 2) + right(2, 2));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE matrix3x3<T>&
operator+=(matrix3x3<T>& left, matrix3x3<T> right) noexcept {
  left = left + right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(matrix3x3<L> left, R right) noexcept {
  return matrix3x3<L>(
      left(0, 0) + right,
      left(0, 1),
      left(0, 2),
      left(1, 0),
      left(1, 1) + right,
      left(1, 2),
      left(2, 0),
      left(2, 1),
      left(2, 2) + right);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE matrix3x3<L>&
operator+=(matrix3x3<L>& left, R right) noexcept {
  left = left + right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(L left, matrix3x3<R> right) noexcept {
  return right + left;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(matrix3x3<T> left, matrix3x3<T> right) noexcept {
  return matrix3x3<T>(
      left(0, 0) - right(0, 0),
      left(0, 1) - right(0, 1),
      left(0, 2) - right(0, 2),
      left(1, 0) - right(1, 0),
      left(1, 1) - right(1, 1),
      left(1, 2) - right(1, 2),
      left(2, 0) - right(2, 0),
      left(2, 1) - right(2, 1),
      left(2, 2) - right(2, 2));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE matrix3x3<T>&
operator-=(matrix3x3<T>& left, matrix3x3<T> right) noexcept {
  left = left - right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(matrix3x3<L> left, R right) noexcept {
  return matrix3x3<L>(
      left(0, 0) - right.raw,
      left(0, 1),
      left(0, 2),
      left(1, 0),
      left(1, 1) - right.raw,
      left(1, 2),
      left(2, 0),
      left(2, 1),
      left(2, 2) - right.raw);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE matrix3x3<L>&
operator-=(matrix3x3<L>& left, R const right) noexcept {
  left = left - right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(matrix3x3<L> left, matrix3x3<R> right) noexcept {
  return matrix3x3<decltype(L() * R())>(
      left(0, 0) * right(0, 0) + left(0, 1) * right(1, 0) + left(0, 2) * right(2, 0),
      left(0, 0) * right(0, 1) + left(0, 1) * right(1, 1) + left(0, 2) * right(2, 1),
      left(0, 0) * right(0, 2) + left(0, 1) * right(1, 2) + left(0, 2) * right(2, 2),
      left(1, 0) * right(0, 0) + left(1, 1) * right(1, 0) + left(1, 2) * right(2, 0),
      left(1, 0) * right(0, 1) + left(1, 1) * right(1, 1) + left(1, 2) * right(2, 1),
      left(1, 0) * right(0, 2) + left(1, 1) * right(1, 2) + left(1, 2) * right(2, 2),
      left(2, 0) * right(0, 0) + left(2, 1) * right(1, 0) + left(2, 2) * right(2, 0),
      left(2, 0) * right(0, 1) + left(2, 1) * right(1, 1) + left(2, 2) * right(2, 1),
      left(2, 0) * right(0, 2) + left(2, 1) * right(1, 2) + left(2, 2) * right(2, 2));
}

template <class L, class R>
HPC_HOST_DEVICE matrix3x3<L>&
operator*=(matrix3x3<L>& left, matrix3x3<R> right) noexcept {
  left = left * right;
  return left;
}

template <class L, class R>
HPC_HOST_DEVICE constexpr auto
operator*(matrix3x3<L> left, vector3<R> right) noexcept {
  return vector3<decltype(L() * R())>(
      left(0, 0) * right(0) + left(0, 1) * right(1) + left(0, 2) * right(2),
      left(1, 0) * right(0) + left(1, 1) * right(1) + left(1, 2) * right(2),
      left(2, 0) * right(0) + left(2, 1) * right(1) + left(2, 2) * right(2));
}

template <class L, class R>
HPC_HOST_DEVICE constexpr auto
operator*(vector3<L> left, matrix3x3<R> right) noexcept {
  return vector3<decltype(L() * R())>(
      left(0) * right(0, 0) + left(1) * right(1, 0) + left(2) * right(2, 0),
      left(0) * right(0, 1) + left(1) * right(1, 1) + left(2) * right(2, 1),
      left(0) * right(0, 2) + left(1) * right(1, 2) + left(2) * right(2, 2));
}

template <class L, class R>
HPC_HOST_DEVICE constexpr auto
operator*(matrix3x3<L> left, R right) noexcept {
  return matrix3x3<decltype(L() * R())>(
      left(0, 0) * right, left(0, 1) * right, left(0, 2) * right,
      left(1, 0) * right, left(1, 1) * right, left(1, 2) * right,
      left(2, 0) * right, left(2, 1) * right, left(2, 2) * right);
}

template <class L, class R>
HPC_HOST_DEVICE matrix3x3<L>&
operator*=(matrix3x3<L>& left, R right) noexcept {
  left = left * right;
  return left;
}

template <class L, class R>
HPC_HOST_DEVICE constexpr auto
operator*(L left, matrix3x3<R> right) noexcept {
  return right * left;
}

template <class L, class R>
HPC_HOST_DEVICE constexpr auto
operator/(matrix3x3<L> left, R right) noexcept {
  return matrix3x3<decltype(L() / R())>(
      left(0, 0) / right, left(0, 1) / right, left(0, 2) / right,
      left(1, 0) / right, left(1, 1) / right, left(1, 2) / right,
      left(2, 0) / right, left(2, 1) / right, left(2, 2) / right);
}

template <class L, class R>
HPC_HOST_DEVICE matrix3x3<L>&
operator/=(matrix3x3<L>& left, R right) noexcept {
  left = left / right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
inner_product(matrix3x3<L> const left, matrix3x3<R> const right) noexcept {
  return (
    left(0, 0)*right(0, 0) + left(0, 1)*right(0, 1) + left(0, 2)*right(0, 2) +
    left(1, 0)*right(1, 0) + left(1, 1)*right(1, 1) + left(1, 2)*right(1, 2) +
    left(2, 0)*right(2, 0) + left(2, 1)*right(2, 1) + left(2, 2)*right(2, 2)
  );
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T
norm(matrix3x3<T> const x) noexcept {
  return std::sqrt(inner_product(x, x));
}

// \return \f$ \max_{j \in {0,\cdots,N}}\Sigma_{i=0}^N |A_{ij}| \f$
template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T
norm_1(matrix3x3<T> const A) noexcept {
  auto const v0 = std::abs(A(0,0)) + std::abs(A(1,0)) + std::abs(A(2,0));
  auto const v1 = std::abs(A(0,1)) + std::abs(A(1,1)) + std::abs(A(2,1));
  auto const v2 = std::abs(A(0,2)) + std::abs(A(1,2)) + std::abs(A(2,2));
  return std::max(std::max(v0, v1), v2);
}

// \return \f$ \max_{i \in {0,\cdots,N}}\Sigma_{j=0}^N |A_{ij}| \f$
template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T
norm_infinity(matrix3x3<T> const A) noexcept {
  auto const v0 = std::abs(A(0,0)) + std::abs(A(0,1)) + std::abs(A(0,2));
  auto const v1 = std::abs(A(1,0)) + std::abs(A(1,1)) + std::abs(A(1,2));
  auto const v2 = std::abs(A(2,0)) + std::abs(A(2,1)) + std::abs(A(2,2));
  return std::max(std::max(v0, v1), v2);
}

template <class T>
HPC_HOST_DEVICE constexpr matrix3x3<T>
transpose(matrix3x3<T> x) noexcept {
  return matrix3x3<T>(
      x(0, 0),
      x(1, 0),
      x(2, 0),
      x(0, 1),
      x(1, 1),
      x(2, 1),
      x(0, 2),
      x(1, 2),
      x(2, 2));
}

template <class L, class R>
HPC_HOST_DEVICE constexpr auto
outer_product(vector3<L> left, vector3<R> right) noexcept {
  return matrix3x3<decltype(L() * R())>(
      left(0) * right(0),
      left(0) * right(1),
      left(0) * right(2),
      left(1) * right(0),
      left(1) * right(1),
      left(1) * right(2),
      left(2) * right(0),
      left(2) * right(1),
      left(2) * right(2));
}

template <typename Scalar>
HPC_HOST_DEVICE constexpr auto
determinant(matrix3x3<Scalar> const x) noexcept {
  Scalar const a = x(0, 0);
  Scalar const b = x(0, 1);
  Scalar const c = x(0, 2);
  Scalar const d = x(1, 0);
  Scalar const e = x(1, 1);
  Scalar const f = x(1, 2);
  Scalar const g = x(2, 0);
  Scalar const h = x(2, 1);
  Scalar const i = x(2, 2);
  return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) -
         (a * f * h);
}

template <typename Scalar>
HPC_HOST_DEVICE constexpr auto
det(matrix3x3<Scalar> const A) noexcept {
  return determinant(A);
}

template <class T>
HPC_HOST_DEVICE constexpr auto
inverse(matrix3x3<T> const x) {
  auto const a = x(0, 0);
  auto const b = x(0, 1);
  auto const c = x(0, 2);
  auto const d = x(1, 0);
  auto const e = x(1, 1);
  auto const f = x(1, 2);
  auto const g = x(2, 0);
  auto const h = x(2, 1);
  auto const i = x(2, 2);
  auto const A = (e * i - f * h);
  auto const D = -(b * i - c * h);
  auto const G = (b * f - c * e);
  auto const B = -(d * i - f * g);
  auto const E = (a * i - c * g);
  auto const H = -(a * f - c * d);
  auto const C = (d * h - e * g);
  auto const F = -(a * h - b * g);
  auto const I = (a * e - b * d);
  using num_t    = matrix3x3<std::remove_const_t<decltype(A)>>;
  auto const num = num_t(A, D, G, B, E, H, C, F, I);
  return num / determinant(x);
}

// Logarithm by Gregory series. Convergence guaranteed for symmetric A
template <typename T>
HPC_HOST_DEVICE constexpr auto
log_gregory(matrix3x3<T> const A)
{
  auto const max_iter  = 8192;
  auto const tol       = machine_epsilon<T>();
  auto const I         = matrix3x3<T>::identity();
  auto const IpA       = I + A;
  auto const ImA       = I - A;
  auto       S         = ImA * inverse(IpA);
  auto       norm_s    = norm(S);
  auto const C         = S * S;
  auto       B         = S;
  auto       k         = 0;
  while (norm_s > tol && ++k <= max_iter) {
    S = (2.0 * k - 1.0) * S * C / (2.0 * k + 1.0);
    B += S;
    norm_s    = norm(S);
  }
  B *= -2.0;
  return B;
}

// Inverse by full pivot. Since this is 3x3, can afford it, and avoids
// cancellation errors as much as possible. This is important for an
// explicit dynamics code that will perform a huge number of these
// calculations.
template <typename T>
HPC_HOST_DEVICE constexpr auto
inverse_full_pivot(matrix3x3<T> const A)
{
  auto S = A;
  auto B = matrix3x3<T>::identity();
  unsigned int intact_rows = (1U << 3) - 1;
  unsigned int intact_cols = intact_rows;
  // Gauss-Jordan elimination with full pivoting
  for (auto k = 0; k < 3; ++k) {
    // Determine full pivot
    auto pivot = 0.0;
    auto pivot_row = 3;
    auto pivot_col = 3;
    for (auto row = 0; row < 3; ++row) {
      if (!(intact_rows & (1 << row))) continue;
      for (auto col = 0; col < 3; ++col) {
        if (!(intact_cols & (1 << col))) continue;
        auto s = std::abs(S(row, col));
        if (s > pivot) {
          pivot_row = row;
          pivot_col = col;
          pivot = s;
        }
      }
    }
    assert(pivot_row < 3);
    assert(pivot_col < 3);
    // Gauss-Jordan elimination
    auto const t = S(pivot_row, pivot_col);
    assert(t != 0.0);
    for (auto j = 0; j < 3; ++j) {
      S(pivot_row, j) /= t;
      B(pivot_row, j) /= t;
    }

    for (auto i = 0; i < 3; ++i) {
      if (i == pivot_row) continue;
      auto const c = S(i, pivot_col);
      for (auto j = 0; j < 3; ++j) {
        S(i, j) -= c * S(pivot_row, j);
        B(i, j) -= c * B(pivot_row, j);
      }
    }
    // Eliminate current row and col from intact rows and cols
    intact_rows &= ~(1 << pivot_row);
    intact_cols &= ~(1 << pivot_col);
  }
  auto const X = transpose(S) * B;
  return X;
}

// Matrix square root by product form of Denman-Beavers iteration.
template <typename T>
HPC_HOST_DEVICE constexpr auto
sqrt_dbp(matrix3x3<T> const A)
{
  auto const eps = machine_epsilon<T>();
  auto const tol = 0.5 * std::sqrt(3.0) * eps; // 3 is dim
  auto const I = matrix3x3<T>::identity();
  auto const max_iter = 32;
  auto X = A;
  auto M = A;
  auto scale = true;
  auto k = 0;
  while (k++ < max_iter) {
    if (scale == true) {
      auto const d = std::abs(det(M));
      auto const d2 = std::sqrt(d);
      auto const d6 = std::cbrt(d2);
      auto const g = 1.0 / d6;
      X *= g;
      M *= g * g;
    }
    auto const Y = X;
    auto const N = inverse(M);
    X *= 0.5 * (I + N);
    M = 0.5 * (I + 0.5 * (M + N));
    auto const error = norm(M - I);
    auto const diff = norm(X - Y) / norm(X);
    scale = diff >= 0.01;
    if (error <= tol) break;
  }
  return std::make_pair(X, k);
}

// Matrix square root
template <typename T>
HPC_HOST_DEVICE constexpr auto
sqrt(matrix3x3<T> const A)
{
  auto X = A;
  std::tie(X, std::ignore) = sqrt_dbp(A);
  return X;
}

// Logarithmic map by Padé approximant and partial fractions
template <typename T>
HPC_HOST_DEVICE constexpr auto
log_pade_pf(matrix3x3<T> const A, int const n)
{
  auto const I = matrix3x3<T>::identity();
  auto X = 0.0 * A;
  for (auto i = 0; i < n; ++i) {
    auto const x = 0.5 * (1.0 + gauss_legendre_abscissae<T>(n, i));
    auto const w = 0.5 * gauss_legendre_weights<T>(n, i);
    auto const B = I + x * A;
    X += w * A * inverse_full_pivot(B);
  }
  return X;
}

// Logarithmic map by inverse scaling and squaring and Padé approximants
template <typename T>
HPC_HOST_DEVICE constexpr auto
log_iss(matrix3x3<T> const A)
{
  auto const I = matrix3x3<T>::identity();
  auto const c15 = pade_coefficients<T>(15);
  auto X = A;
  auto i = 5;
  auto j = 0;
  auto k = 0;
  auto m = 0;
  while (true) {
    auto const diff = norm_1(X - I);
    if (diff <= c15) {
      auto p = 2; while(pade_coefficients<T>(p) <= diff && p < 16) {++p;};
      auto q = 2; while(pade_coefficients<T>(q) <= diff / 2.0 && q < 16) {++q;};
      if ((2 * (p - q) / 3) < i || ++j == 2) {m = p + 1; break;}
    }
    std::tie(X, i) = sqrt_dbp(X); ++k;
  }
  X = (1U << k) * log_pade_pf(X - I, m);
  return X;
}

// Logarithmic map
template <typename T>
HPC_HOST_DEVICE constexpr auto
log(matrix3x3<T> const A)
{
  return log_iss(A);
}

// Project to O(N) (Orthogonal Group) using a Newton-type algorithm.
// See Higham's Functions of Matrices p210 [2008]
// \param A tensor (often a deformation-gradient-like tensor)
// \return \f$ R = \argmin_Q \|A - Q\|\f$
// This algorithm projects a given tensor in GL(N) to O(N).
// The rotation/reflection obtained through this projection is
// the orthogonal component of the real polar decomposition
template <typename T>
HPC_HOST_DEVICE constexpr auto
polar_rotation(matrix3x3<T> const A)
{
  auto const dim = 3;
  auto scale = true;
  auto const tol_scale = 0.01;
  auto const tol_conv = std::sqrt(dim) * machine_epsilon<T>();
  auto X = A;
  auto gamma = 2.0;
  auto const max_iter = 128;
  auto num_iter = 0;
  while (num_iter < max_iter) {
    auto const Y = inverse_full_pivot(X);
    auto mu = 1.0;
    if (scale == true) {
      mu = (norm_1(Y) * norm_infinity(Y)) / (norm_1(X) * norm_infinity(X));
      mu = std::sqrt(std::sqrt(mu));
    }
    auto const Z = 0.5 * (mu * X + transpose(Y) / mu);
    auto const D = Z - X;
    auto const delta = norm(D) / norm(Z);
    if (scale == true && delta < tol_scale) {
      scale = false;
    }
    bool const end_iter = norm(D) <= std::sqrt(tol_conv) ||
        (delta > 0.5 * gamma && scale == false);
    X = Z;
    gamma = delta;
    if (end_iter == true) {
      break;
    }
    num_iter++;
  }
  return X;
}

template <typename T>
HPC_HOST_DEVICE constexpr auto
symm(matrix3x3<T> const A)
{
  return 0.5 * (A + transpose(A));
}

template <typename T>
HPC_HOST_DEVICE constexpr auto
skew(matrix3x3<T> const A)
{
  return 0.5 * (A - transpose(A));
}

template <typename T>
HPC_HOST_DEVICE constexpr auto
polar_left(matrix3x3<T> const A)
{
  auto const R = polar_rotation(A);
  auto const V = symm(A * transpose(R));
  return std::make_pair(V, R);
}

template <typename T>
HPC_HOST_DEVICE constexpr auto
polar_right(matrix3x3<T> const A)
{
  auto const R = polar_rotation(A);
  auto const U = symm(transpose(R) * A);
  return std::make_pair(R, U);
}

template <class T>
HPC_HOST_DEVICE constexpr T
trace(matrix3x3<T> x) noexcept {
  return x(0, 0) + x(1, 1) + x(2, 2);
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3<T>
isotropic_part(matrix3x3<T> const x) noexcept {
  return ((1.0 / 3.0) * trace(x)) * matrix3x3<T>::identity();
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3<T>
vol(matrix3x3<T> const A) noexcept {
  return isotropic_part(A);
}

template <class T>
HPC_HOST_DEVICE constexpr matrix3x3<T>
deviatoric_part(matrix3x3<T> x) noexcept {
  auto x_dev = matrix3x3<T>(x);
  auto const a = (1.0 / 3.0) * trace(x);
  x_dev(0,0) -= a;
  x_dev(1,1) -= a;
  x_dev(2,2) -= a;
  return x_dev;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3<T>
dev(matrix3x3<T> const A) noexcept {
  return deviatoric_part(A);
}

template <class T>
class array_traits<matrix3x3<T>> {
  public:
  using value_type = T;
  using size_type = decltype(axis_index() * axis_index());
  HPC_HOST_DEVICE static constexpr size_type size() noexcept { return 9; }
  template <class Iterator>
  HPC_HOST_DEVICE static matrix3x3<T> load(Iterator it) noexcept {
    return matrix3x3<T>(
        it[0],
        it[1],
        it[2],
        it[3],
        it[4],
        it[5],
        it[6],
        it[7],
        it[8]);
  }
  template <class Iterator>
  HPC_HOST_DEVICE static void store(Iterator it, matrix3x3<T> const& value) noexcept {
    it[0] = value(0, 0);
    it[1] = value(0, 1);
    it[2] = value(0, 2);
    it[3] = value(1, 0);
    it[4] = value(1, 1);
    it[5] = value(1, 2);
    it[6] = value(2, 0);
    it[7] = value(2, 1);
    it[8] = value(2, 2);
  }
};

}
