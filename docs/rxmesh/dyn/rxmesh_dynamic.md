# **`RXMeshDynamic`**

`RXMeshDynamic` is the host-side mesh class for topology-changing algorithms. It derives from `RXMeshStatic`, so the static API remains available, e.g., attributes, `for_each`, custom kernels, reductions, indexing, and visualization all use the same programming model.

The dynamic work happens in four places:

- Construct the mesh with extra capacity.
- Prepare launches for kernels that use `CavityManager`.
- Drive the patch scheduler until the dynamic pass finishes.
- Repair and refresh the mesh after topology changes.

---

## **Construction**

The constructors mirror `RXMeshStatic`. The extra parameters control how much room each patch has to grow and how many new patches can be created by slicing.

```cpp
RXMeshDynamic rx("mesh.obj");
```

```cpp
std::vector<std::vector<uint32_t>> faces = ...;
RXMeshDynamic rx(faces);
```

??? note "`RXMeshDynamic(file_path, patcher_file, patch_size, capacity_factor, patch_alloc_factor, lp_hashtable_load_factor)`"
    Builds a dynamic mesh from an OBJ file.

    - `file_path`: input OBJ path.
    - `patcher_file`: optional saved patcher state for reproducing a partition.
    - `patch_size`: target number of faces per patch.
    - `capacity_factor`: extra per-patch storage for dynamic growth.
    - `patch_alloc_factor`: extra patch slots available for slicing.
    - `lp_hashtable_load_factor`: load factor for local patch lookup tables.

??? note "`RXMeshDynamic(fv, patcher_file, patch_size, capacity_factor, patch_alloc_factor, lp_hashtable_load_factor)`"
    Builds a dynamic mesh from an in-memory face list. `fv[i]` stores the vertex
    indices of face `i`.

The defaults are usually enough for first experiments. Increase `capacity_factor` when a pass adds many local elements before slicing, and increase `patch_alloc_factor` when the mesh may split into many more patches.

---

## **Dynamic Launch Boxes**

Dynamic kernels are normal CUDA kernels, but they need enough dynamic shared memory for both RXMesh queries and `CavityManager`.

```cpp
constexpr uint32_t blockThreads = 256;
LaunchBox<blockThreads> lb;

rx.prepare_launch_box(
    {Op::EVDiamond},
    lb,
    (void*)edge_split<blockThreads, float>,
    true,   // is_dyn: kernel uses CavityManager
    false,  // oriented
    false,  // with_vertex_valence
    false,  // is_concurrent
    [](uint32_t v, uint32_t e, uint32_t f) {
        return detail::mask_num_bytes(e) +
               ShmemAllocator::default_alignment;
    });
```

Use `is_dyn = true` for kernels that instantiate `CavityManager`. Use `is_dyn = false` for ordinary query kernels launched on an `RXMeshDynamic` instance, e.g., smoothing or cost computation between topology edits.

??? note "`prepare_launch_box(op, launch_box, kernel, is_dyn, oriented, with_vertex_valence, is_concurrent, user_shmem) const`"
    Populates `launch_box` and checks the kernel launch requirements. The  `op` list must include the RXMesh query operations used in the kernel. The `user_shmem` callback returns extra shared-memory bytes your kernel allocates
    with `ShmemAllocator`.

??? note "`update_launch_box(op, launch_box, kernel, is_dyn, oriented, with_vertex_valence, is_concurrent, user_shmem) const`"
    Same inputs as `prepare_launch_box`, but intended for repeated use inside loops. It updates launch dimensions without the extra setup chatter of the first preparation call.

Call `update_launch_box` again when the number of patches may have changed, for example after `slice_patches(...)`.

---

## **Patch Scheduler**

Dynamic kernels process one patch per CUDA block. Some patches cannot commit in a given launch because a neighboring patch is being edited or because the patch needs to be retried after slicing. The scheduler controls that retry loop.

```cpp
rx.reset_scheduler();

while (!rx.is_queue_empty()) {
    rx.update_launch_box(...);

    my_dynamic_kernel<<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
        rx.get_context(), ...);
   
    rx.slice_patches(...);
    rx.cleanup();
}
```

??? note "`void reset_scheduler()`"
    Re-fills the patch queue with the current live patches. Call this at the start of every dynamic pass.

??? note "`bool is_queue_empty(cudaStream_t stream = NULL)`"
    Returns `true` when there are no more queued patches for the current pass.

---

## **Repair After a Dynamic Kernel**

After a dynamic kernel, call `cleanup()` and `slice_patches(...)` before launching another topology edit.

```cpp
rx.slice_patches(*coords, *edge_status, *v_boundary);
rx.cleanup();
```

??? note "`void cleanup()`"
    Removes stale deleted elements and recalculates global element counts

??? note "`template <typename... AttributesT> void slice_patches(AttributesT... attributes)`"
    Splits patches that have grown too large. Pass every live attribute whose values must follow vertices, edges, or faces into newly created patches.

---

## **Host Refresh and Validation**

Topology changes happen on the device. Before using the modified mesh on the host, refresh the host-side representation.

```cpp
rx.update_host();
coords->move(DEVICE, HOST);
rx.update_polyscope();
```

??? note "`void update_host()`"
    Copies dynamic topology from the device to the host and resizes host-side buffers when the mesh has grown.

??? note "`void update_polyscope(std::string new_name = \"\")`"
    Re-registers the surface mesh in Polyscope after topology changed. Move vertex positions to the host before calling this.

??? note "`bool validate()`"
    Runs GPU-side consistency checks. This is useful while developing a new dynamic kernel or debugging a failing pass.