# **`Query`**

`Query<blockThreads>` is the **device-side driver** of a neighborhood query. It is the low-level building block that [`for_each<Op, blockThreads>`](static/for_each.md#connectivity-based-for_each) wraps for the common case, and that you use directly from a kernel body when you call [`run_kernel`](static/run_kernel.md). A single `Query` instance is tied to one [patch](../getting-started/concepts.md#patches) and one block; it cooperates across the block to load per-patch connectivity into shared memory, then invokes your per-element lambda over every seed element in the patch.

---

## **Template and Instantiation**

```cpp
template <uint32_t blockThreads>
struct Query;
```

- **`blockThreads`**: the CUDA block size. The header provides explicit instantiations for `128`, `256`, `320`, `384`, `512`, `768`, and `1024`, and these are the values you should use.

---

## **Construction**

```cpp
auto block = cooperative_groups::this_thread_block();
Query<256> query(context);
```

??? note "`Query(const Context& context, uint32_t pid = blockIdx.x)`"
    Binds the query to a [`Context`](context.md) and a patch id. The default, `pid = blockIdx.x`, matches the "one block per patch" launch shape that [`prepare_launch_box`](launch_box.md) produces. `Query` is non-copyable; construct it fresh in each kernel invocation.

??? note "`int get_patch_id() const` / `const PatchInfo& get_patch_info() const`"
    Introspection for advanced kernels that need to branch on patch metadata. Rarely used in normal code.

---

## **`dispatch`**

The main entry point. `dispatch` is **collective across the block**: every thread in the block must call it, and you must pass in the cooperative-groups thread block and a [`ShmemAllocator`](shmem_allocator.md) so `Query` has room to stage connectivity.

```cpp
auto block = cooperative_groups::this_thread_block();
ShmemAllocator shrd_alloc;
Query<blockSize> query(context);

auto work = [&](FaceHandle fh, VertexIterator& fv) {
    // ... per-face lambda ...
};

query.dispatch<Op::FV>(block, shrd_alloc, work);
```

??? note "`dispatch<Op>(block, shrd_alloc, compute_op, oriented = false)`"
    Runs a `prologue` → per-element invocation of `compute_op` → `epilogue` sequence for the query `Op`. `compute_op` is a device callable with signature `(InputHandle, OutputIterator&)`, exactly the same shape as in [`for_each<Op, blockThreads>`](static/for_each.md#connectivity-based-for_each). `oriented` requests oriented traversal (only meaningful for `Op::VV` and `Op::VE`).

??? note "`dispatch<Op>(block, shrd_alloc, compute_op, active_set, oriented = false, allow_not_owned = false)`"
    Active-set variant. `active_set` is a device predicate on the source handle; only elements for which it returns `true` are visited. The first argument type of `compute_op` and `active_set` must match (enforced by a `static_assert`). `allow_not_owned` permits invoking `compute_op` on elements not owned by the current patch (off by default).

After `dispatch` returns, `shrd_alloc` is restored to its pre-dispatch byte count, so any shared memory `Query` used is reclaimed.

---

## **Manual Control: the Split API**

For kernels that want to interleave work between the steps `dispatch` normally performs, `Query` exposes the three phases as separate methods. This is an advanced API, most users should prefer `dispatch`.

```cpp
query.prologue<Op::FV>(block, shrd_alloc);
// ... custom work before per-element invocation ...
query.run_compute<Op::FV>(block, work);
// ... custom work after, e.g., writing results ...
query.epilogue(block, shrd_alloc);
```

??? note "`prologue<Op>(block, shrd_alloc, oriented = false, allow_not_owned = true)`"
    Stages connectivity for the query. Records the current `shrd_alloc` usage so `epilogue` can pop back to it.

??? note "`prologue<Op>(block, shrd_alloc, active_set, oriented = false, allow_not_owned = true)`"
    Variant with an active-set predicate.

??? note "`run_compute<Op>(block, compute_op)`"
    Invokes `compute_op` on each active source element in the patch using the staged connectivity.

??? note "`get_iterator<Op, IteratorT>(uint16_t local_id) const`"
    Returns an iterator over the neighborhood of a specific local element, between `prologue` and `epilogue`. Useful when you need random access to neighborhoods rather than iteration in source order.

??? note "`epilogue(block, shrd_alloc)`"
    Releases the shared memory staged by `prologue` by popping `shrd_alloc` back to the byte count saved at the start.

---

## **Optional Vertex Valence**

Some kernels need per-vertex valence (degree). `Query` can compute it once and cache it in shared memory. Request the extra shared memory by setting `with_vertex_valence = true` when calling [`prepare_launch_box`](launch_box.md), then use the following accessors inside the kernel:

??? note "`compute_vertex_valence(block, shrd_alloc)`"
    Collectively computes per-vertex valence for the current patch and stores it in shared memory carved from `shrd_alloc`.

??? note "`uint16_t vertex_valence(VertexHandle vh) const` / `vertex_valence(uint16_t local_v) const`"
    Returns the valence of a vertex. Only valid after `compute_vertex_valence` has been called.

---

## **Multiple Queries per Kernel**

You can run several queries in the same kernel, in two patterns:

- **Serial (one at a time)**: call `dispatch` back-to-back on the same `Query` instance. Each `dispatch` allocates, invokes your lambda, and deallocates, so peak shared memory usage is the **maximum** across the queries. Call [`prepare_launch_box`](launch_box.md) with `is_concurrent = false` (the default) to size shared memory for the max op.

- **Concurrent (multiple live buffers)**: use the split API with multiple `Query` instances so that several neighborhoods are live at once. Peak shared memory usage is the **sum** across the queries. Call [`prepare_launch_box`](launch_box.md) with `is_concurrent = true` so dynamic smem is sized accordingly.

---

## **Orientation and Manifold Requirements**

- The `oriented` flag is documented as valid for `Op::VV` and `Op::VE`; other ops ignore it or treat the mesh as oriented by default (e.g., `Op::FV`, `Op::FE`, `Op::EV`).
- `Op::EVDiamond` and `Op::EE` require an **edge-manifold** input mesh. `prepare_launch_box` errors early if the mesh is not edge-manifold.

For the full list of queries, see [supported query types](static/for_each.md#supported-query-types).

---

## **Typical In-Kernel Pattern**

For a full walkthrough with a custom kernel body, see [Writing a custom CUDA kernel](static/run_kernel.md#writing-a-custom-cuda-kernel). The gist:

```cpp
template <uint32_t blockSize>
__global__ void my_kernel(Context context)
{
    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    Query<blockSize> query(context);

    auto work = [&](FaceHandle fh, VertexIterator& fv) { /* ... */ };
    query.dispatch<Op::FV>(block, shrd_alloc, work);
}
```