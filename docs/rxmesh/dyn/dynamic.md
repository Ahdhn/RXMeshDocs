# **Overview**

`RXMeshDynamic` is the RXMesh class for algorithms that change mesh connectivity on the GPU. Use it for local topology edits, e.g., edge splits, edge flips, edge collapses, and remeshing passes where the set of vertices, edges, or faces changes while the program is running.

It inherits `RXMeshStatic`. Attributes, neighborhood queries, reductions, custom kernels, and visualization still work the same way. The dynamic layer adds one extra idea, i.e., topology edits which are expressed as **cavities**. A cavity is a set of connected mesh elements that we removes and then fills back in with new mesh elements.

The high-level structure of the device kernel for any dynamic operation is:

1. Pick candidate seed elements with RXMesh query.
2. Create cavities around the candidates.
3. Let `CavityManager` keep a non-conflicting subset.
4. Fill each surviving cavity with new vertices, edges, and faces.
5. Clean up the mesh and, when patches grew, slice them.

The host loop usually looks like this:

```cpp
while (!done) {
    rx.reset_scheduler();

    while (!rx.is_queue_empty()) {
        rx.update_launch_box({Op::EVDiamond},
                             lb,
                             (void*)split_edges<blockThreads, float>,
                             true);

        split_edges<blockThreads>
            <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                rx.get_context(), *coords, *edge_status);
       
        rx.slice_patches(*coords, *edge_status);
        rx.cleanup();
    }

    done = check_convergence(...);
}
```

The two loops do different jobs. The inner loop drains the patch scheduler, i.e., a kernel launch processes the patches currently available, and patches that could not commit are requeued. The outer loop belongs to the algorithm, i.e., it keeps launching passes until there is no more work to do.

---

## **What `RXMeshDynamic` Adds**

Compared to `RXMeshStatic`, `RXMeshDynamic` adds:

- **Patch scheduling**: a queue of patches to be processed by dynamic kernels.
- **Cavity operations**: the device-side API for local topology changes.
- **Post-edit repair**: `cleanup()` and `slice_patches(...)`.
- **Dynamic launch sizing**: `prepare_launch_box(..., is_dyn = true, ...)`.
- **Host refresh**: `update_host()` and `update_polyscope()` after topology changes.

Everything else is still RXMesh. You continue to store data in attributes, write CUDA kernels over handles, and use `Query` objects for local stencils.

---

## **When to Use `RXMeshDynamic`**

Use `RXMeshDynamic` when your algorithm changes connectivity:

| Operation | Typical cavity op | What changes |
|-----------|-------------------|--------------|
| Edge split | `CavityOp::E` | One edge and two faces are replaced by a fan around a new vertex. |
| Edge flip | `CavityOp::E` | Two triangles are replaced by the other diagonal. |
| Edge collapse | `CavityOp::EV` | An edge and its endpoint one-rings are replaced by a merged vertex. |
| Local remeshing | `CavityOp::E` and `CavityOp::EV` | Split, collapse, and flip passes are composed. |