# **Terms**

When defining an objective function for optimization, a **term** is one additive piece that defined this objective function. You describe it by writing a **device lambda** over a mesh stencil like in [`for_each<Op, blockThreads>`](../for_each.md) but the lambda operates on [`Scalar`](scalar.md) dual numbers during the active pass instead of plain floats. RXMesh takes care of the rest, i.e., the local derivatives your lambda produces are scattered into the global gradient / Hessian / Jacobian owned by the problem.

This page explains the three building blocks that are used to define a term:

- **`DiffHandle`**: the handle type passed to the lambda, which knows whether the current pass is active or passive.
- **`opt_var.active<...>()`**: the method on the optimization-variable attribute that loads its values into a pre-seeded `Eigen` vector of `Scalar` values.
- **`element_valence`**: the compile-time helper that relates a query `Op` to the local variable vector size.

---

## **The Term Lambda**

Once you own a [`DiffScalarProblem`](scalar_problem.md) or [`DiffVectorProblem`](vector_problem.md), you register terms with `add_term`:

```cpp
problem.add_term<Op::EV>(
    [=] __device__(const auto& eh, const auto& iter, auto& opt_var) {
        using ActiveT = ACTIVE_TYPE(eh);
        Eigen::Vector3<ActiveT> v0 = opt_var.template active<3>(eh, iter, 0);
        Eigen::Vector3<ActiveT> v1 = opt_var.template active<3>(eh, iter, 1);
        auto d = v0 - v1;
        return d.dot(d);
    });
```

The shape of the lambda mirrors exactly the signature you would pass to `for_each<Op, ...>`. See [supported query types](../for_each.md#supported-query-types) for the full list. The difference from plain `for_each` is that we can extract the information of whether we are running in active (with derivatives) or passive mode (no derivatives) from the input handle [`DiffHandle`](#diffhandle), i.e., extracting the type `ActiveT` which dictates whether the lambda runs on `Scalar<k, ...>` or on plain floats during any given pass.

---

## **Active vs. Passive Pass**

During optimization, we usually need to evaluate the objective function (composed of multiple terms) more than once per solver step in two different modes:

- **Active pass** where we use `Scalar` dual numbers so the lambda's return value carries gradient (and optionally Hessian) information. 
- **Passive pass** where we use plain floats/doubles, i.e., no derivatives calculation, e.g., during line search

In both cases, you write **one** lambda that is templated through `auto` parameters and then RXMesh instantiates it once per mode. 

---

## **`DiffHandle`** { #diffhandle }

Every term lambda receives a `DiffHandle` instead of a raw `VertexHandle` / `EdgeHandle` / `FaceHandle`. A `DiffHandle` wraps the original [handle](../handles.md) and additionally remembers whether the current kernel invocation is in active or passive mode.

??? info "`ACTIVE_TYPE(H)` macro"
    Inside a term lambda, you may need the active type associated with a `DiffHandle`, e.g., to load the optimization variable as an active/passive parameter or to declare an intermediate. The `ACTIVE_TYPE(H)` macro returns that type for handle type `H`. We also provide a type trait `is_scalar_v` that can be used in `constexpr` if statement as below:
    ```
    using ActiveT = ACTIVE_TYPE(eh);
    if constexpr (is_scalar_v<ActiveT>) {
        //do computation for active pass only 
    }
    ```


Inside the lambda you mostly just forward the handle to `opt_var.active<...>` and to attributes. The one method worth knowing is:

```cpp
if (dh.is_active()) {
    // Derivatives are being tracked, e.g., compute analytic shortcuts.
}
```

??? note "`DiffHandle<HandleT, ActiveT>`"
    - Implicitly convertible back to the wrapped `HandleT`, so attribute access like `my_attr(dh)` works as usual.
    - `bool is_active() const` — `true` in the active pass (the `Scalar`-valued instantiation), `false` in the passive pass.
    - The aliases `DiffVertexHandle`, `DiffEdgeHandle`, `DiffFaceHandle` are typically what you write in lambda signatures, or you can use `auto&` and let template deduction do it.
---

## **Loading Active Variables: `opt_var.active`** { #active }

`opt_var.active<VariableDim>(...)` is the bridge between the optimization-variable attribute and a `Scalar`-valued local variable vector. It is a method on the variable attribute itself, so the same call works for `VertexAttribute`, `EdgeAttribute`, and `FaceAttribute`. There are two overloads commonly used in terms (a third overload for **interaction terms** is documented in [Advanced Topics](advanced.md#interaction-terms)):

- **Element-wise** (`opt_var.active<VariableDim>(dh)`): the term depends only on the seed handle. Returns an `Eigen::Matrix<ActiveT, VariableDim, 1>` whose derivatives are seeded at the canonical positions `0 .. VariableDim - 1`. Used when the energy at a vertex/edge/face is intrinsic to that element.
- **Connectivity-based** (`opt_var.active<VariableDim>(dh, iter, slot)`): the term depends on a stencil produced by a query `Op`. Returns the `VariableDim` components at `iter[slot]`, with derivatives seeded at indices `[slot * VariableDim, (slot + 1) * VariableDim)`. Local variable count is `iter.size() * VariableDim`.

In the active pass each entry is a `Scalar`; in the passive pass each entry is the underlying `float` / `double`. You write the same lambda body in both cases. The exact derivative layout is computed by `index_mapping(VariableDim, slot, variable_local_id)`; see [Advanced Topics](advanced.md#util).

For a term on `Op::FV` with `VariableDim = 3`:

```cpp
problem.add_term<Op::FV, Scalar<9>>(
    [=] __device__(const auto& fh, const auto& iter, auto& opt_var) {
        using ActiveT = ACTIVE_TYPE(fh);
        Eigen::Vector3<ActiveT> v0 = opt_var.template active<3>(fh, iter, 0);
        Eigen::Vector3<ActiveT> v1 = opt_var.template active<3>(fh, iter, 1);
        Eigen::Vector3<ActiveT> v2 = opt_var.template active<3>(fh, iter, 2);
        // v0, v1, v2 are Eigen::Matrix<ActiveT, 3, 1>
        // Their derivatives are seeded at indices 0-2, 3-5, 6-8 respectively.
        return /* energy in terms of v0, v1, v2 */;
    });
```

??? note "`opt_var.active<VariableDim>(dh)`"
    Single-handle overload. Returns the `VariableDim` components of `opt_var` at the seed handle `dh` as an `Eigen::Matrix<ActiveT, VariableDim, 1>`. In the active pass each entry is a `Scalar` with `grad()[j] = 1` at component `j`. Used when a term does not depend on a stencil (e.g., `Op::V` per-element residuals).

??? note "`opt_var.active<VariableDim>(dh, iter, index_in_iter)`"
    Connectivity overload. Reads the `VariableDim` components of `opt_var` at `iter[index_in_iter]` and returns them as an `Eigen::Matrix<ActiveT, VariableDim, 1>`. In the active pass derivatives are seeded so that this slot is the active block in a local variable vector of size `iter.size() * VariableDim`.

Values from other attributes (rest shape, material parameters, boundary masks, ...) are read normally via `my_attr(handle, c)` or the typed read helpers `my_attr.template to_eigen<N>(handle)` / `my_attr.template to_glm<N>(handle)`, and they are treated as passive automatically.

---

## **`element_valence`** { #element_valence }

The second template argument to `add_term` is an active type `Scalar<k, PassiveT, WithHessian>`. You choose `k` based on the stencil size but RXMesh provides a helper for the common cases:

```cpp
template <Op op> constexpr int element_valence();
```

- Returns the compile-time number of output elements of the query (e.g., `3` for `Op::FV`, `4` for `Op::EVDiamond`, `2` for `Op::EV`).
- Returns `-1` for queries whose valence is not fixed at compile time (e.g., `Op::VV`, `Op::VE`, `Op::VF`). We currently do not support derivative computation for these queries. 


For a term with `VariableDim` components per mesh element, the natural `k` is `element_valence<op>() * VariableDim`. For example, a 3D-vertex term over a triangle is `3 * 3 = 9`, and over an edge diamond is `4 * 3 = 12`.
