# **`Context`**

`Context` is the **device-side handle** to the mesh's GPU state. It bundles the pointers, sizes, and patch info that RXMesh's kernels need to resolve handles, count elements, and compute linear indices. Every kernel launched through [`run_kernel`](static/run_kernel.md) receives a `Context` by value, and [`Query`](query.md) takes one in its constructor.

Users rarely construct or initialize a `Context` themselves, it is created and owned by `RXMeshStatic` / `RXMeshDynamic`. The typical job of this page is to document the **read-only** accessors that are useful inside a kernel.

---

## **Getting a `Context`**

`Context` is trivially copyable, so it is passed into kernels by value. If you use the library's dispatch helpers, you get one automatically:

- [`run_kernel`](static/run_kernel.md) injects `Context` as the first argument to your `__global__` function:

```cpp
template <uint32_t blockSize>
__global__ void my_kernel(Context context, /* other args */)
{
    Query<blockSize> query(context);
    // ...
}
```

- [`Query`](query.md)'s constructor takes a `Context` and a patch id. The default `pid = blockIdx.x` matches the "one block per patch" launch shape RXMesh uses.

---

## **Mesh-Wide Counts**

These accessors return total counts across the whole mesh (summed across all patches). On the device, they read from device pointers owned by `Context`.

??? note "`uint32_t get_num<HandleT>() const`"
    Total number of elements of type `HandleT` in the mesh, where `HandleT` is `VertexHandle`, `EdgeHandle`, or `FaceHandle`. Evaluates to the underlying count value.

??? note "`uint32_t* get_num_vertices()` / `get_num_edges()` / `get_num_faces()`"
    Raw device pointers to the count storage. Useful when you want to read a count atomically or pass it to another kernel. Most users prefer `get_num<HandleT>()`.

??? note "`uint32_t get_num_patches() const`"
    Total number of patches in the mesh.

---

## **Linear IDs**

`Context` exposes the device-side `linear_id` accessors referenced from [Indexing](static/indexing.md). These convert a handle into a **flat 0-based index** across the mesh, accounting for patch ownership (non-owned ghost elements are routed to their owner patch).

??? note "`uint32_t linear_id(HandleT h) const`"
    Safe linear id for a vertex, edge, or face handle. Resolves ownership through the LP hash table if the element is not owned by its own patch. The returned id is in `[0, get_num<HandleT>())`.

??? note "`uint32_t linear_id_fast(HandleT h) const`"
    Unsafe, much faster: assumes the handle already refers to the owner patch and returns `prefix[patch_id] + local_id`. Use only when you have verified ownership (e.g., in a static mesh where the input handle came from `for_each_vertex`).

??? note "`HandleT get_handle<HandleT>(uint32_t linear_id)`"
    Reverse mapping. Returns the `HandleT` whose `linear_id` equals the input. Overloads `get_vertex_handle(i)`, `get_edge_handle(i)`, `get_face_handle(i)` exist for when you do not want to specify a template argument.

??? note "`const uint32_t* prefix<HandleT>() const`"
    Device pointer to the per-patch prefix sum used by `linear_id_fast`. The inline helpers `vertex_prefix()`, `edge_prefix()`, `face_prefix()` exist for the same purpose.

---

## **Ownership**

Every element has a **single owner patch**. In kernels that cross patch boundaries, you sometimes need the canonical handle so that atomic updates land on the same memory.

??? note "`HandleT get_owner_handle(HandleT h, ...) const`"
    Walks the LP hash table until it finds the owner of `h` and returns a new `HandleT` whose `patch_id` is that owner's patch. If `h` is already owned by its own patch, returns `h` unchanged. The additional optional arguments (`patches_info`, `table`, `stash`, `check0`, `check1`) let advanced kernels pass shared-memory-cached hashtable pointers and toggle debug assertions.

---

## **Utilities**

??? note "`static void unpack_edge_dir(uint16_t edge_dir, uint16_t& edge, flag_t& dir)`"
    Splits a packed directed-edge value into its edge id and direction bit. Used internally by queries; only needed if you implement a custom kernel that reads edge-dir values from shared memory directly.