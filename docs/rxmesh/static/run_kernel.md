# **`run_kernel`**

`run_kernel` is a low-level interface for launching a custom CUDA kernel where part of the logic uses neighborhood-based mesh queries. Unlike [`for_each`](for_each.md) which bundle one query `Op` and a single device lambda, `run_kernel` exposes the full kernel body. That is useful when you need work before or after the query, want to reuse shared memory across steps, or need multiple `Op` values in one kernel.

---

## **`run_kernel` overloads**

### 1) Automatic launch configuration

This overload performs `prepare_launch_box` internally. It is the most flexible and takes every configuration knob.

??? note "`run_kernel(kernel, op, oriented, with_vertex_valence, is_concurrent, user_shmem, stream, args…)`"
    ```cpp
    template <uint32_t blockSize, typename KernelT, typename... ArgsT>
    void run_kernel(
        KernelT kernel,
        const std::vector<Op> op,
        const bool oriented,
        const bool with_vertex_valence,
        const bool is_concurrent,
        std::function<size_t(uint32_t, uint32_t, uint32_t)> user_shmem,
        cudaStream_t stream,
        ArgsT... args)
    ```

    - **`op`**: One or more query operations used inside the kernel. See [Supported query types](for_each.md#supported-query-types).
    - **`oriented`**: Whether query results must be **oriented** where applicable (meaningful only for certain ops, e.g., `Op::VV` / `Op::VE`).
    - **`with_vertex_valence`**: Precompute vertex valences and store them in shared memory when needed.
    - **`is_concurrent`**: When **`op.size() > 1`**, whether multiple queries are accessed **at the same time** in the kernel body.
    - **`user_shmem`**: Returns **extra** dynamic shared memory (bytes) as a function of patch vertex, edge, and face counts; RXMesh adds its own requirements on top.
    - **`stream`**: CUDA stream for the launch.
    - **`args…`**: Arguments forwarded to **`kernel`** after RXMesh’s [`Context`](static.md) (see your kernel signature).

### 2) `LaunchBox` (explicit stream)

Use this when you already called [`prepare_launch_box`](../reference/launch_box.md) with the same `op`, kernel pointer, and shared-memory plan, and you want to **reuse** the resulting grid and shared-memory sizes across launches.

??? note "`run_kernel(lb, kernel, stream, args…)`"
    ```cpp
    template <uint32_t blockSize, typename KernelT, typename... ArgsT>
    void run_kernel(
        const LaunchBox<blockSize>& lb,
        const KernelT kernel,
        cudaStream_t stream,
        ArgsT... args)
    ```

    Launches **`kernel`** with **`lb.blocks`**, **`lb.num_threads`**, and **`lb.smem_bytes_dyn`** on **`stream`**.

### 3) `LaunchBox` (default stream)

Same as above when the **default** CUDA stream is acceptable.

??? note "`run_kernel(lb, kernel, args…)`"
    ```cpp
    template <uint32_t blockSize, typename KernelT, typename... ArgsT>
    void run_kernel(
        const LaunchBox<blockSize>& lb,
        const KernelT kernel,
        ArgsT... args)
    ```

### 4) Minimal convenience overload

Skip extra flags when your kernel only needs a simple query list and **default** orientation / concurrency / extra shared memory.

??? note "`run_kernel(op, kernel, args…)`"
    ```cpp
    template <uint32_t blockSize, typename KernelT, typename... ArgsT>
    void run_kernel(
        const std::vector<Op> op,
        KernelT kernel,
        ArgsT... args)
    ```

    Forwards to the full automatic path with **default** `oriented`, **`is_concurrent`**, and **no** extra user shared memory—only use this when that matches your kernel’s needs.

---

## **Writing a custom CUDA kernel**

With `run_kernel`, you author a **`__global__`** function that receives RXMesh’s execution **`Context`** and, inside the block, drives **`Query`** dispatches. The sections below follow one example: accumulate face normals into vertex normals using `Op::FV`.

### Example: computing vertex normal

```cpp
template <uint32_t blockSize>
__global__ void vertex_normal(Context context)
{
    auto compute_vn = [&](const FaceHandle face_id, const VertexIterator& fv) {
        const vec3<float> c0 = vertex_pos.to_glm<3>(fv[0]);
        const vec3<float> c1 = vertex_pos.to_glm<3>(fv[1]);
        const vec3<float> c2 = vertex_pos.to_glm<3>(fv[2]);

        glm::fvec3 n = cross(c1 - c0, c2 - c0);
        n = glm::normalize(n);

        for (uint32_t v = 0; v < 3; ++v)
            for (uint32_t i = 0; i < 3; ++i)
                atomicAdd(&normals(fv[v], i), n[i]);
    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockSize> query(context);
    ShmemAllocator shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, compute_vn);
}
```

The per-element work is device lambda passed into `dispatch`. For `Op::FV`, the lambda receives a `FaceHandle` and a `VertexIterator` over that face’s vertices:

```cpp
auto compute_vn = [&](const FaceHandle face_id, const VertexIterator& fv) { /* … */ };
```

Here the lambda builds the face normal, then uses `atomicAdd` so each incident vertex receives a contribution.

### Cooperative groups

Query dispatch must run **collectively** across the block. Capture the thread block group and pass it into `dispatch`:

```cpp
auto block = cooperative_groups::this_thread_block();
```

??? info "CUDA Cooperative Groups"
    Background: [CUDA C++ Programming Guide — Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups).

### `Query` object

Construct a **`Query<blockSize>`** (template argument matches your kernel’s `blockSize`) with the **`context`** argument RXMesh passes to your kernel:

```cpp
Query<blockSize> query(context);
```

This object manages loading neighborhood data into shared memory for the ops you dispatch.

### Shared memory allocator

Use **[`ShmemAllocator`](../reference/shmem_allocator.md)** for **your** dynamic shared-memory allocations so they do not collide with RXMesh’s internal buffers:

```cpp
ShmemAllocator shrd_alloc;
```

The total dynamic shared memory for the launch must match what you configured when calling **`run_kernel`** / **`prepare_launch_box`** (through **`user_shmem`** on the automatic overload).

### Dispatching the query

??? note "`query.dispatch<Op>(block, shrd_alloc, lambda)`"
    ```cpp
    query.dispatch<Op::FV>(block, shrd_alloc, compute_vn);
    ```

    - Runs the **`Op::FV`** query over faces.
    - Sets up internal shared memory for that query.
    - Invokes **`compute_vn`** for each face.

### Restricting to an active set

To limit work to a subset of mesh elements (e.g. only certain faces), define an **active-set** predicate and pass it into the **`dispatch`** overload that supports filtering:

```cpp
auto active_set = [&](FaceHandle fh) -> bool { /* … */ };
query.dispatch<Op::FV>(block, shrd_alloc, compute_vn, active_set);
```

Skipping inactive elements avoids unnecessary work when the predicate is cheap to evaluate.
