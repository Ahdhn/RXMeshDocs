# **`LaunchBox`**

`LaunchBox<blockThreads>` is a small host-side struct that holds everything RXMesh needs to launch a CUDA kernel that runs one-block-per-patch, i.e., grid size, dynamic shared-memory bytes, static shared-memory bytes, and the block size itself. You create it on the host, hand it to [`RXMeshStatic::prepare_launch_box`](#prepare_launch_box) so the library can size it for your query mix, and then pass it to [`run_kernel`](../static/run_kernel.md) or [`for_each`](../static/for_each.md).

---

## **Definition**

From `include/rxmesh/launch_box.h`:

```cpp
template <uint32_t blockThreads>
struct LaunchBox
{
    uint32_t blocks;
    uint32_t num_registers_per_thread;
    size_t   smem_bytes_dyn;
    size_t   smem_bytes_static;
    size_t   local_mem_per_thread;
    const uint32_t num_threads = blockThreads;
};
```

??? note "`uint32_t blocks`"
    CUDA grid size. Populated by `prepare_launch_box` to match the number of active patches (one block per patch). Read-only from the user's perspective after `prepare_launch_box`.

??? note "`uint32_t num_registers_per_thread`"
    Registers per thread the compiler allocated for the target kernel, populated via `cudaFuncGetAttributes`. Useful for occupancy tuning and diagnostics.

??? note "`size_t smem_bytes_dyn`"
    Dynamic shared memory in bytes. This is the value that gets forwarded as the third argument of the `<<<..., ..., smem, stream>>>` launch and is read by [`ShmemAllocator`](shmem_allocator.md) to bound its allocator. Sized by `prepare_launch_box` to cover the worst-case query in `op`, possibly increased to accommodate vertex valence and `user_shmem`.

??? note "`size_t smem_bytes_static`"
    Compiler-reported static shared memory for the kernel (`cudaFuncAttributes::sharedSizeBytes`).

??? note "`size_t local_mem_per_thread`"
    Compiler-reported local memory (per thread) for the kernel (`cudaFuncAttributes::localSizeBytes`).

??? note "`const uint32_t num_threads = blockThreads`"
    Block size, fixed at compile time by the template parameter.

The typical user code pattern:

```cpp
constexpr uint32_t blockThreads = 256;
LaunchBox<blockThreads> lb;
rx.prepare_launch_box({Op::FV}, lb, (void*)my_kernel<blockThreads>);
rx.run_kernel(lb, my_kernel<blockThreads>, /* kernel args */);
```

---

## **`prepare_launch_box`**

`RXMeshStatic::prepare_launch_box` is the host-side helper that sizes shared memory and picks the grid dimension. It must be called before the kernel, once per combination of `(kernel, op, blockThreads, options)`.

```cpp
template <uint32_t blockThreads>
void prepare_launch_box(
    const std::vector<Op>                                    op,
    LaunchBox<blockThreads>&                                 launch_box,
    const void*                                              kernel,
    const bool                                               oriented            = false,
    const bool                                               with_vertex_valence = false,
    const bool                                               is_concurrent       = false,
    std::function<size_t(uint32_t, uint32_t, uint32_t)>      user_shmem          =
        [](uint32_t v, uint32_t e, uint32_t f) { return 0; }) const;
```

??? note "`const std::vector<Op> op`"
    The list of query operations performed by the kernel. For a single query, pass `{Op::FV}`. For multiple queries, pass them all so shared memory can be sized correctly. Passing an empty vector means "no queries" and the launch box is only sized for `user_shmem`.

??? note "`LaunchBox<blockThreads>& launch_box`"
    The destination struct. Its fields are overwritten.

??? note "`const void* kernel`"
    A pointer to the target kernel, typically `(void*)my_kernel<blockThreads>`. Used to query `cudaFuncAttributes` for `num_registers_per_thread`, `smem_bytes_static`, and `local_mem_per_thread`, and to configure the kernel's dynamic shared-memory limit with `cudaFuncSetAttribute`.

??? note "`bool oriented = false`"
    If `true`, sizes shared memory assuming oriented traversal. Only meaningful for `Op::VV` and `Op::VE` queries.

??? note "`bool with_vertex_valence = false`"
    If `true`, reserves additional shared memory for the per-vertex valence buffer so that `Query::compute_vertex_valence` can be called inside the kernel. See [`Query` vertex valence](query.md#optional-vertex-valence).

??? note "`bool is_concurrent = false`"
    Controls sizing when `op.size() > 1`. If `false` (default), multiple queries run **serially**, i.e., one after another, in which case the shared memory is sized for the **max** of the per-op requirements. If `true`, multiple queries are expected to be alive at the same time (e.g., via the split [`Query` Manual Control](query.md#manual-control-the-split-api)), so shared memory is sized for the **sum**.

??? note "`std::function<size_t(uint32_t, uint32_t, uint32_t)> user_shmem`"
    A callback that receives the per-patch element counts `(num_vertices, num_edges, num_faces)` and returns the number of additional dynamic shared-memory bytes the kernel needs. Use this when your kernel allocates its own scratch buffers from [`ShmemAllocator`](shmem_allocator.md) on top of what `Query` uses. Defaults to returning `0`.

---

## **Errors and Diagnostics**

`prepare_launch_box` validates the input and logs with `RXMESH_ERROR` before returning.

`op` must contain only queries valid for the current mesh, e.g., `Op::EVDiamond` and `Op::EE` require an edge-manifold mesh.

If the requested shared memory exceeds the device's per-block limit, `prepare_launch_box` will error.