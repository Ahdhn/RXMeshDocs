# **Dual Numbers**

Forward-mode automatic differentiation in RXMesh is built on a single arithmetic primitive, the **dual number**, implemented by the `Scalar` class. A dual number is a value that carries its own **derivatives** alongside its numerical value, so every arithmetic operation you write on it propagates derivatives automatically. 

??? tip "Acknowledgement: TinyAD"
    The initial implementation of `Scalar` was adapted from [TinyAD](https://github.com/patr-schm/TinyAD), a clean and elegant C++ forward-mode AD library by Patrick Schmidt et al. We are grateful for their work since it shaped the interface users see here. The main difference in RXMesh is that `Scalar` is usable **on the GPU**, i.e., every constructor, operator, and math function is `__host__ __device__`, and the type integrates with RXMesh's execution model.


A conventional float stores a single number. A `Scalar` stores:

- a **value** `v(x)`,
- a **gradient** `∇v ∈ R^k` with respect to a fixed-size local variable vector of length `k`,
- and, optionally, a **Hessian** `∇²v ∈ R^{k×k}`.

When you write `a * b`, RXMesh applies the product rule to both the gradient and the Hessian in the same line of code. Expressions like `sqrt`, `exp`, `log`, `sin`, and the usual arithmetic operators all work transparently, on the host and on the device.

`k` is the size of the **local variable vector** for your stencil. For example, a per-face term over a triangle with 3D vertex positions has `k = 3 vertices × 3 coords = 9`. You pick `k` once per term, and the rest of the pipeline is driven by that choice.

---

## **The `Scalar` Type**

```cpp
template <int k, typename PassiveT = float, bool WithHessian = true>
class Scalar;
```

- **`k`**: size of the local variable vector (compile-time, integer).
- **`PassiveT`**: the underlying floating-point type (`float` or `double`).
- **`WithHessian`**: if `true`, each `Scalar` also carries a `k × k` Hessian block; if `false`, only the gradient.

Typical choices:

- Scalar on 3D vertex positions per triangle with Hessian → `Scalar<9, float, true>`.
- Gradient-only minimization over a triangle → `Scalar<9, float, false>`.
- Line-search trial evaluations → plain `float`/`double` (see [passive evaluation](terms.md#active-vs-passive-pass)).

---

## **Creating Active and Passive Values**

Values inside a differentiable computation are either **active** (they track derivatives, i.e., we are differentiating with respect to them) or **passive** (constants, e.g., the rest shape or a user parameter).

```cpp
using Diff = Scalar<2, float, true>;

// Active variables seeded at indices 0 and 1 of the local variable vector.
Diff x = Diff::make_active(3.0f, 0);
Diff y = Diff::make_active(4.0f, 1);

// A constant that participates in the formula but is not differentiated.
float c = 2.5f;

// f(x, y) = x^2 + x * y + c
Diff f = x * x + x * y + c;

float       val  = f.val();   // 23.5
Eigen::Vector2f g  = f.grad(); // [2x + y, x] = [10, 3]
Eigen::Matrix2f H  = f.hess(); // [[2, 1], [1, 0]]
```

??? note "Constructors"
    - `Scalar()` — default-initialized passive zero.
    - `Scalar(PassiveT v)` — passive value.
    - `Scalar(PassiveT v, int idx)` — active value, derivative seeded at local index `idx`.    
    - `static Scalar make_active(PassiveT v, int idx)` — explicit active variable.
    - `static Scalar make_active(const Eigen::Matrix<PassiveT, k, 1>& v)` — build a `Eigen::Matrix<Scalar, k, 1>` from a passive vector, seeded so that component `i` is active at index `i`.
    - `static Scalar known_derivatives(PassiveT v, grad, hess)` — advanced, useful when you already have analytic derivatives for a sub-expression.

---

## **Accessors**

```cpp
Scalar<9, float, true> s = ...;

float                      v = s.val();   // numerical value
Eigen::Matrix<float, 9, 1> g = s.grad();  // 9-vector of first derivatives
Eigen::Matrix<float, 9, 9> H = s.hess();  // 9x9 Hessian (only if WithHessian)
```

??? note "`val()`, `grad()`, `hess()`"
    - `PassiveT val() const` — the primal value.
    - `Eigen::Matrix<PassiveT, k, 1> grad() const` — the gradient with respect to the local variable vector.
    - `Eigen::Matrix<PassiveT, k, k> hess() const` — only available when `WithHessian == true`.

---

## **Arithmetic and Math Functions**

`Scalar` overloads the usual operators and common math functions so ordinary C++ code just works:

```cpp
using Diff = Scalar<2, float, true>;
Diff a, b, c;
Diff expr = (a * b + 2.0f * c) / (1.0f + a * a);
Diff sd   = sqrt(expr * expr + 1.0f);
```

The following all work on host and device, and all of them propagate gradient and (when enabled) Hessian:

- `+`, `-`, `*`, `/`, unary `-`, and their compound forms.
- `sqrt`, `exp`, `log`, `pow`, `abs`.
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`.
- Comparisons return `bool` based on `val()`, so `if (x > 0)` behaves as expected.

`Scalar` also integrates with Eigen via `Eigen::NumTraits<Scalar<...>>`, so `Eigen::Matrix<Scalar<k, T>, m, n>` and the usual matrix algebra (multiplication, `transpose`, `determinant`, `inverse`, ...) compose naturally.

---

## **Casting Back to Passive**

When a sub-expression does not need to be differentiated, cast it back to a plain float to save registers and memory:

```cpp
Diff active = ...;
float passive = active.val();  // drop derivatives explicitly
```

There is also a generic helper for matrices of `Scalar`:

```cpp
Eigen::Matrix<Diff, 3, 3> M = ...;
Eigen::Matrix<float, 3, 3> M_passive = to_passive(M);
```

??? note "`to_passive(x)`"
    - For a `Scalar<k, T, *>`, returns the underlying `T`.
    - For a plain arithmetic `T`, returns `x` unchanged.
    - For an `Eigen::Matrix` of `Scalar`, returns the same-shape matrix of passive values.

    Use this when evaluating an energy at a trial step during line search, or when mixing differentiated and non-differentiated code.

---

## ** Example**

You can use `Scalar` outside any RXMesh problem to sanity-check a formula. For example, differentiating `f(x, y) = x² + xy`:

```cpp
using Diff = rxmesh::Scalar<2, float, true>;

Diff x = Diff::make_active(3.0f, 0);
Diff y = Diff::make_active(4.0f, 1);

Diff f = x * x + x * y;

assert(f.val() == 21.0f);
// grad = [2x + y, x] = [10, 3]
// hess = [[2, 1], [1, 0]]
```

---

## **Usage**

You normally do not write `make_active` by hand inside RXMesh terms. Instead, the active vector is produced by [`opt_var.active<...>`](terms.md#active), which reads the optimization-variable attribute at the current stencil's handles and seeds derivatives according to each handle's slot in the local variable vector. See [Terms](terms.md) for more information.
