# **`ShmemAllocator`**

`ShmemAllocator` is a **per-block bump allocator** over CUDA dynamic shared memory. It is the mechanism RXMesh uses to hand pieces of the block's dynamic shared memory to [`Query`](query.md) and to any scratch buffers your kernel wants to allocate on top. Because it is a pure bump allocator, allocations are `O(1)` and ordering matters: **deallocation is LIFO**.

Use `ShmemAllocator` whenever you write a kernel that you launch via [`run_kernel`](static/run_kernel.md) and that carves its own shared-memory buffers.

---

## **Backing Memory**

All allocations come out of the same global **`extern __shared__ char SHMEM_START[]`** block that CUDA provides when you pass dynamic shared-memory bytes as the third argument of the launch configuration. Because RXMesh's launchers forward `LaunchBox::smem_bytes_dyn` as that argument, the total pool available at runtime equals the dynamic shared-memory size sized by [`prepare_launch_box`](launch_box.md#prepare_launch_box).

The allocator queries the current block's budget at runtime via the PTX special register `%dynamic_smem_size`, so `get_max_size_bytes()` is always in sync with what was actually launched, regardless of how you passed the value.

---

## **Creating an Allocator**

```cpp
ShmemAllocator shrd_alloc;  // constructed in a kernel thread
```

??? note "`__device__ ShmemAllocator()`"
    Default-constructed allocators start at `SHMEM_START` with no bytes used. Typically you instantiate one per kernel, in a register (per-thread). If you instead place it in `__shared__` memory, only a single thread per block should call `alloc` / `dealloc` to avoid races.

---

## **Allocation**

??? note "`char* alloc(uint32_t num_bytes, uint32_t byte_alignment = default_alignment)`"
    Advances the internal pointer by `num_bytes`, aligning to `byte_alignment` first (8 by default). Returns a pointer to the start of the allocation, or `nullptr` if the allocation would overflow the dynamic shared-memory budget. Asserts the same condition in debug builds.

??? note "`template <class T> T* alloc(uint32_t count)`"
    Typed convenience wrapper. Calls `alloc(count * sizeof(T), default_alignment)` and `reinterpret_cast`s the result. `static_assert`s that `sizeof(T) <= default_alignment` (i.e., `8`), since the allocator only guarantees 8-byte alignment; use the raw `alloc` with a larger alignment if you need more.

---

## **Deallocation (LIFO only)**

The deallocation APIs **subtract** from the bump pointer; they do **not** run any bookkeeping. They are safe only when you pop in reverse allocation order, mirroring a stack. Any other order silently creates overlap.

??? note "`void dealloc(uint32_t num_bytes)`"
    Rewinds the pointer by `num_bytes`. Intended to undo the most recent allocation of the same size. Asserts that the pointer does not go below `SHMEM_START`.

??? note "`template <class T> void dealloc(uint32_t count)`"
    Typed convenience wrapper for `dealloc(count * sizeof(T))`.

`Query::epilogue` uses this API to reclaim exactly what `Query::prologue` allocated, which is why a serial sequence of `Query::dispatch` calls peaks at the **max** query size rather than the **sum**.

---

## **Introspection**

??? note "`uint32_t get_max_size_bytes() const`"
    The total dynamic shared memory the block was launched with, read from the PTX special register `%dynamic_smem_size`.

??? note "`uint32_t get_allocated_size_bytes() const`"
    Bytes currently used, i.e., the offset of the bump pointer from `SHMEM_START`.

---

## **Cooperation with `prepare_launch_box` and `user_shmem`**

If your kernel uses `ShmemAllocator` beyond what `Query` needs, you must tell [`prepare_launch_box`](launch_box.md#prepare_launch_box) how much extra it should size into `LaunchBox::smem_bytes_dyn`. The `user_shmem` callback exists for exactly this: it is invoked with the per-patch `(num_vertices, num_edges, num_faces)` counts and returns the extra bytes to reserve.

```cpp
rx.prepare_launch_box(
    {Op::FV},
    lb,
    (void*)my_kernel<blockThreads>,
    false, false, false,
    [](uint32_t v, uint32_t e, uint32_t f) {
        // Enough space for one int32 per vertex.
        return v * sizeof(int32_t);
    });
```

Inside the kernel, allocate the matching amount **before** running queries that depend on `Query`'s own shared-memory staging, or **after** them, mirroring the order you expect `dealloc` to pop if you later release any of them.

---

## **Typical Kernel Pattern**

```cpp
template <uint32_t blockSize>
__global__ void my_kernel(Context context)
{
    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    int32_t* scratch =
        shrd_alloc.alloc<int32_t>(context.get_num_patches());

    Query<blockSize> query(context);
    query.dispatch<Op::FV>(block, shrd_alloc,
        [&] __device__(FaceHandle fh, VertexIterator& fv) {
            // ...
        });

    // If you want to pop `scratch`:
    // shrd_alloc.dealloc<int32_t>(context.get_num_patches());
}
```