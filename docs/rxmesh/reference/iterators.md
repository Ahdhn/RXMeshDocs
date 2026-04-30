# **Iterators**

An **iterator** is a device-side view over the neighborhood of one mesh element, produced by a neighborhood query. When you write a lambda for [`for_each<Op, blockThreads>`](../static/for_each.md#connectivity-based-for_each) or [`run_kernel`](../static/run_kernel.md), RXMesh hands you one input [handle](handles.md) and one iterator whose type depends on the query `Op`. You read the neighbors through `size()` and `operator[]`.

Users do not construct iterators manually; they are built by the library and passed into your device lambda.

---

## **Iterator Types**

`Iterator<HandleT>` is a class template parameterized by the handle type of its elements. Three aliases cover the common cases:

| Alias | Element type |
|-------|--------------|
| `VertexIterator` | `Iterator<VertexHandle>` |
| `EdgeIterator` | `Iterator<EdgeHandle>` |
| `FaceIterator` | `Iterator<FaceHandle>` |

For each query `Op`, the iterator type matches the "output" side of the query. For example, `Op::FV` (for each face, its vertices) hands you a `VertexIterator`; `Op::VF` hands you a `FaceIterator`. See [supported query types](../static/for_each.md#supported-query-types).

---

## **Public API**

All iterator methods are **`__device__` only**. You cannot instantiate or use an iterator on the host.

```cpp
rx.for_each<Op::FV, 256>(
    [=] __device__(const FaceHandle fh, const VertexIterator& fv) {
        for (uint16_t i = 0; i < fv.size(); ++i) {
            const VertexHandle vh = fv[i];
            if (!vh.is_valid()) continue;
            // ...
        }
    });
```

??? note "`uint16_t size() const`"
    Number of neighbor elements. For fixed-valence queries (e.g., `Op::FV` on a triangle mesh → `3`, `Op::EV` → `2`, `Op::EVDiamond` → `4`) the value is constant. For variable-valence queries (e.g., `Op::VV`, `Op::VE`, `Op::VF`) it depends on the specific seed element.

??? note "`HandleT operator[](uint16_t i) const`"
    Returns the `i`-th neighbor as a **handle** of type `HandleT`. Out-of-range or structurally-missing slots return a default-constructed (invalid) handle; guard with `is_valid()` when in doubt.

??? note "`HandleT front() const`"
    Shortcut for `(*this)[0]`.

??? note "`HandleT back() const`"
    Shortcut for `(*this)[size() - 1]`.

??? note "`uint16_t local(uint16_t i) const`"
    Returns the raw **local id** of the `i`-th neighbor within its patch (no handle construction). Rarely needed by user code; useful when indexing patch-local scratch buffers. Returns `INVALID16` if `i >= size()`.

---

## **No Range-For Support**

`Iterator<HandleT>` does **not** expose `begin()` / `end()`, so C++ range-based `for (auto vh : fv)` does **not** compile. Use an index loop:

```cpp
for (uint16_t i = 0; i < fv.size(); ++i) {
    VertexHandle vh = fv[i];
    // ...
}
```

---

## **Iterator Semantics by Query**

A few queries have noteworthy structural behaviour:

- **`Op::EVDiamond`**: for each edge, the iterator exposes up to **four** vertices, the two endpoints of the edge plus the two "opposite" vertices of the adjacent faces. The size is fixed at 4; boundary edges may produce one invalid slot, so always check `is_valid()`.
- **`Op::FV`, `Op::FE`, `Op::EV`**: oriented by construction; `operator[]` returns neighbors in a consistent traversal order.
- **`Op::VV`, `Op::VE`**: respect the `oriented` flag of the query (see [`for_each<Op, blockThreads>`](../static/for_each.md#connectivity-based-for_each) and [`Query::dispatch`](query.md)).

See [supported query types](../static/for_each.md#supported-query-types) for the full list.