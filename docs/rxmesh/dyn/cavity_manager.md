# **`CavityManager`**

`CavityManager<blockThreads, CavityOp>` is the device-side object that performs topology edits. Each CUDA block creates one manager for the patch it receives from the scheduler.

```cpp
auto block = cooperative_groups::this_thread_block();
ShmemAllocator shrd_alloc;

CavityManager<blockThreads, CavityOp::E> cavity(
    block, context, shrd_alloc, true);

if (cavity.patch_id() == INVALID32) {
    return;
}
```

The template `blockThreads` must match the block size used to launch the kernel. The `CavityOp` controls the deleted neighborhood, as described in [`CavityOp`](cavity_op.md).

??? note "`CavityManager(block, context, shrd_alloc, preserve_cavity, allow_touching_cavities = true, current_p = 0)`"
    - `block`: cooperative-groups thread block.
    - `context`: RXMesh `Context` passed to the kernel.
    - `shrd_alloc`: shared-memory allocator used by RXMesh queries and user scratch buffers.
    - `preserve_cavity`: keeps deleted interior elements readable during fill-in.
    - `allow_touching_cavities`: allows cavities to share a boundary when `true`.
    - `current_p`: debugging helper for processing one patch.

??? note "`uint32_t patch_id() const`"
    Returns the patch assigned to this block, or `INVALID32` when the block should exit.

---

## **1. Mark Candidate Cavities**

Use normal RXMesh queries to decide which elements should be edited. When a seed passes the predicate, call `create(seed)`.

```cpp
Query<blockThreads> query(context, cavity.patch_id());

auto should_split = [&](EdgeHandle eh, VertexIterator& iter) {
    if (edge_is_too_long(eh, iter, coords)) {
        cavity.create(eh);
    }
};

query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_split);
block.sync();
```

??? note "`template <typename HandleT> uint32_t create(HandleT seed)`"
    Marks `seed` as a candidate cavity. The seed handle type must match the source element of the `CavityOp`.

It is fine if many candidates overlap. `prologue(...)` filters them to a safe set before fill-in.

---

## **2. Commit With `prologue`**

`prologue(...)` prepares the patch for fill-in. It selects non-conflicting cavities, makes required boundary data available, deletes the cavity interiors from the patch-local view, and updates attributes that must follow migrated elements.

```cpp
if (cavity.prologue(block, shrd_alloc, coords, edge_status, v_boundary)) {
    // Safe to fill cavities here.
}
```

??? note "`template <typename... AttributesT> bool prologue(block, shrd_alloc, AttributesT&&... attributes)`"
    Returns `true` when this block can fill and commit its cavities. Pass every live attribute whose values must remain valid after topology changes.

Attributes passed to `prologue(...)` are the attributes RXMesh keeps coherent when ownership or patch placement changes. Forgetting an attribute is a common source of stale values after dynamic edits.

---

## **3. Fill Each Cavity**

`for_each_cavity(...)` calls your fill-in lambda once for each surviving cavity. The lambda receives the cavity id and boundary size.

```cpp
cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
    VertexHandle new_v = cavity.add_vertex();
    if (!new_v.is_valid()) {
        return;
    }

    // Initialize attributes on the new vertex.
    coords(new_v, 0) = ...;
    coords(new_v, 1) = ...;
    coords(new_v, 2) = ...;

    // Add edges and faces that refill the boundary.
});
```

??? note "`template <typename FillInT> void for_each_cavity(block, FillInT fill_in)`"
    Iterates over the active cavities selected by `prologue(...)`.

??? note "`int get_num_cavities() const`"
    Number of cavities selected in this patch.

??? note "`uint16_t get_cavity_size(uint16_t c) const`"
    Number of boundary edges in cavity `c`.

??? note "`VertexHandle get_cavity_vertex(uint16_t c, uint16_t i) const`"
    Returns boundary vertex `i`.

??? note "`DEdgeHandle get_cavity_edge(uint16_t c, uint16_t i) const`"
    Returns boundary edge `i` with orientation.

---

## **Fill-In Primitives**

Use the `add_*` methods only inside `for_each_cavity(...)`.

```cpp
VertexHandle v = cavity.add_vertex();
DEdgeHandle  e = cavity.add_edge(src, dst);
FaceHandle   f = cavity.add_face(e0, e1, e2);
```

??? note "`VertexHandle add_vertex()`"
    Adds a vertex to the current patch. The returned handle can be used to initialize vertex attributes. Check `is_valid()` before using it.

??? note "`DEdgeHandle add_edge(VertexHandle src, VertexHandle dst)`"
    Adds an oriented edge from `src` to `dst`. The underlying undirected edge can be accessed with `get_edge_handle()`.

??? note "`FaceHandle add_face(DEdgeHandle e0, DEdgeHandle e1, DEdgeHandle e2)`"
    Adds a triangular face. The three directed edges should form a closed loop.

If an `add_*` call returns an invalid handle, stop filling that cavity. RXMesh marks the patch for retry or slicing as needed.

---

## **Recovering a Cavity**

Sometimes a cavity passes the initial predicate but fails a more expensive fill-in check. For example, an edge collapse may create an edge that is too long. In that case, you can recover the original cavity.

```cpp
EdgeHandle src = cavity.get_creator<EdgeHandle>(c);

if (would_create_bad_triangle(src)) {
    cavity.recover(src);    
    return;
}
```

??? note "`template <typename HandleT> void recover(HandleT seed)`"
    Restores the cavity created by `seed`. Use it during fill-in after
    `prologue(...)` and before `epilogue(...)`.

??? note "`template <typename HandleT> HandleT get_creator(uint16_t cavity_id)`"
    Returns the seed handle that created cavity `cavity_id`.

??? note "`template <typename HandleT> bool is_recovered(HandleT handle)`"
    Returns whether `handle` belongs to a recovered cavity.

---

## **4. Always Call `epilogue`**

Every path through the kernel must call `epilogue(...)`.

```cpp
cavity.epilogue(block);
block.sync();

if (cavity.is_successful()) {
    // Optional post-processing, such as marking newly added edges.
}
```

??? note "`void epilogue(block)`"
    Finishes the dynamic edit, writes the updated patch state when the block
    succeeded, and releases resources used by the manager.

??? note "`bool is_successful()`"
    Returns `true` after `epilogue(...)` when this block committed its edits.

??? note "`template <typename HandleT> bool is_successful(HandleT seed)`"
    Returns whether a specific seed survived cavity selection.

Post-processing that writes status attributes is usually guarded by
`is_successful()`, because failed patches will be retried by the scheduler.
