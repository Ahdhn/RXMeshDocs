# **`run_kernel` **

`run_kernel` is a low-level interface that allows you to launch a custom CUDA kernel where part of the logic involves neighborhood-based mesh queries. Unlike `run_query_kernel`, which encapsulates both the query and the computation in a single lambda, `run_kernel` gives you control over the full kernel body. This is useful when:

- You need to perform computation before or after the query operation.
- You want to cache or reuse data in shared memory.
- You want to use multiple queries within the same kernel.

There are two categories of `run_kernel` overloads: one where RXMesh computes the launch configuration internally, and one where you provide a pre-computed `LaunchBox`.

---

## Signature (Automatic Launch Configuration)

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

This overload performs all the setup work. It is the most flexible but also expects more inputs.

- `op`: A list of one or more query operations to be used inside the kernel. See [Supported Queries Types](#supported-queries-types).
- `oriented`: Whether query results need to be sorted.
- `with_vertex_valence`: Whether to precompute vertex valences and store them in shared memory.
- `is_concurrent`: Whether the queries will be accessed concurrently in the same kernel body.
- `user_shmem`: A lambda that returns the amount of extra shared memory (in bytes) required.
- `args...`: Inputs passed directly to the kernel.

---
## Signature (With `LaunchBox`)

```cpp
template <uint32_t blockSize, typename KernelT, typename... ArgsT>
void run_kernel(
    const LaunchBox<blockSize>& lb,
    const KernelT kernel,
    cudaStream_t stream,
    ArgsT... args)
```

This version assumes you have already initialized a `LaunchBox<blockSize>` using `prepare_launch_box`. This is useful when the same kernel is launched multiple times, as it avoids recomputing the launch configuration.

---
## Signature (`LaunchBox` + Default Stream)

```cpp
template <uint32_t blockSize, typename KernelT, typename... ArgsT>
void run_kernel(
    const LaunchBox<blockSize>& lb,
    const KernelT kernel,
    ArgsT... args)
```

Same as above, but runs on the default CUDA stream.

---

## Signature (Minimal Form)

```cpp
template <uint32_t blockSize, typename KernelT, typename... ArgsT>
void run_kernel(
    const std::vector<Op> op,
    KernelT kernel,
    ArgsT... args)
```

This form skips all optional flags and assumes defaults for `oriented` (i.e., false), `is_concurrent` (i.e., false), and shared memory. Use only if your kernel does not need anything special.


In the next section, we will discuss how to write your own query-enabled CUDA kernel to be used with `run_kernel`.

---

## Writing a Custom CUDA Kernel with Query Operations
When using `run_kernel`, you need to write your own CUDA kernel. This gives you full control over the execution and allows you to combine query operations with custom computation and shared memory logic. Below is a breakdown of the key components in such a kernel.

### Example: Computing Face Normals
This example demonstrates how to compute vertex normals by first computing face normals and then distributing them to the vertices of each face. 

```cpp
template<uint32_t blockSize>
__global__ void vertex_normal (Context context){      
    auto compute_vn = [&](const FaceHandle face_id, const VertexIterator& fv) {    	        
    	const vec3<float> c0 = vertex_pos.to_glm<3>(fv[0]);
        const vec3<float> c1 = vertex_pos.to_glm<3>(fv[1]);
        const vec3<float> c2 = vertex_pos.to_glm<3>(fv[2]);
        
        glm::fvec3 n = cross(c1 - c0, c2 - c0);

        n = glm::normalize(n);

        // add the face's normal to its vertices
    	for (uint32_t v = 0; v < 3; ++v)       // for every vertex in this face
            for (uint32_t i = 0; i < 3; ++i)   // for the vertex 3 coordinates
    		        atomicAdd(&normals(fv[v], i), n[i]); 
    };  
  auto block = cooperative_groups::this_thread_block();  
  Query<blockThreads> query(context);  
  ShmemAllocator shrd_alloc;
  query.dispatch<Op::FV>(block, shrd_alloc, compute_vn);
} 
```

** 1- User-defined computation**

Inside the kernel, the actual logic is defined via a lambda:

```cpp
auto compute_vn = [&] (const FaceHandle face_id, const VertexIterator& fv) { ... };
```

This lambda gets called on each face in the mesh. The second argument `fv` is a `VertexIterator` that provides access to the 3 vertices of the face (since it is an `Op::FV` query). In this example:

- It computes the face normal by forming two edge vectors and taking their cross product.

- Then, it loops over the 3 face vertices and adds the face normal to each vertex using `atomicAdd`.


** 2- Cooperative Group**

Query operations must be called collectively by all threads in the CUDA block. To ensure this, a cooperative thread group is created using:

```cpp
auto block = cooperative_groups::this_thread_block();
```
This group is passed into the query object to ensure synchronization and correctness. See [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) for more information about CUDA cooperative groups 

**3- Creating the Query Object**

You create a `Query<blockSize>` object, passing in the RXMesh execution context:

```cpp
Query<blockSize> query(context);
```

This object encapsulates all the logic required to load neighborhood information into shared memory and provide access to it during execution.


**4- Shared Memory Allocator**

RXMesh allows you to use additional shared memory inside the kernel. To manage that safely alongside the memory RXMesh uses, you should use the `ShmemAllocator`:

```cpp
ShmemAllocator shrd_alloc;
```
This allocator ensures proper alignment and avoids overlaps with internal buffers. You can then use `shrd_alloc`
to allocate shared memory for other purposes without worrying that they will overlap with RXMesh shared memory used to do the query operation. You have to make sure that the amount of shared memory is allocated properly you call `run_kernel`. See [here](shmem_allocator.md) for more information about `ShmemAllocator`.

**5- Dispatching the Query**
   
The actual work/query is launched using:

```cpp
query.dispatch<Op::FV>(block, shrd_alloc, compute_vn);
```

This:

- Executes the `Op::FV` query.
- Allocates and initializes internal shared memory.
- Runs the user-defined lambda (`compute_vn`) on each face.


#### Restricting to an Active Set

Sometimes you only want to run the query on a subset of elements (e.g., active faces). You can define an **active set predicate**:

```cpp
auto active_set = [&] (FaceHandle fh) -> bool { ... };
```

You then pass it to `dispatch`:

```cpp
query.dispatch<Op::FV, blockSize>(context, compute_vn, active_set);
```
This improves performance by skipping unnecessary computations.