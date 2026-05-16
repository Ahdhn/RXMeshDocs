# **Cavity Concepts**

A **cavity** is the unit of topology change in RXMesh. Instead of exposing a
separate primitive for every edit, RXMesh asks you to describe the edit as:

1. Choose a seed element.
2. Delete a local neighborhood around that seed.
3. Fill the boundary loop with new vertices, edges, and faces.

That pattern covers edge splits, edge flips, edge collapses, and larger local
remeshing operations.

---

## **Interior and Boundary**

A cavity has two parts:

- **Interior**: the elements removed by the cavity.
- **Boundary**: the surviving loop that your fill-in reconnects.

The cavity operation decides what the interior contains. For example,
`CavityOp::E` starts from an edge and removes the edge plus its incident faces.
That is enough for both edge split and edge flip. `CavityOp::EV` starts from an
edge and removes the endpoint one-rings, which is the usual shape for edge
collapse.

Inside the fill-in code, `CavityManager` exposes the boundary:

```cpp
cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
    for (uint16_t i = 0; i < size; ++i) {
        VertexHandle v = cavity.get_cavity_vertex(c, i);
        DEdgeHandle  e = cavity.get_cavity_edge(c, i);
    }
});
```

The boundary is ordered. Consecutive boundary edges walk around the cavity, and
the directed edge orientation is the orientation you use when building new
faces.

---

## **The Dynamic Kernel Shape**

Every dynamic kernel has the same skeleton:

```cpp
template <uint32_t blockThreads>
__global__ void edit_kernel(Context context, VertexAttribute<float> coords)
{
    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, true);

    if (cavity.patch_id() == INVALID32) {
        return;
    }

    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, [&](EdgeHandle eh,
                                                         VertexIterator& ev) {
        if (should_edit(eh, ev, coords)) {
            cavity.create(eh);
        }
    });

    if (cavity.prologue(block, shrd_alloc, coords)) {
        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            fill_cavity(cavity, c, size, coords);
        });
    }

    cavity.epilogue(block);
}
```

The interesting algorithm-specific work is usually small: `should_edit(...)`
decides whether an element becomes a seed, and `fill_cavity(...)` creates the
replacement topology.

---

## **What RXMesh Handles**

A block may create many candidate cavities. RXMesh handles the difficult parts:

- Cavities that overlap are filtered down to a non-conflicting subset.
- Patches that cannot safely commit are retried by the scheduler.
- Boundary elements needed from neighboring patches are made available locally.
- Attribute values are kept with migrated elements when the attributes are
  passed to `prologue(...)`.
- Failed fill-in can be rolled back with `recover(...)` or by returning invalid
  handles from `add_*`.

The user-facing rule is simple: create candidate cavities freely, pass all live
attributes to `prologue(...)`, fill only the cavities RXMesh gives you through
`for_each_cavity(...)`, and always call `epilogue(...)`.

---

## **Preserving the Deleted Region**

The `preserve_cavity` constructor argument controls whether deleted interior
elements remain readable during fill-in.

```cpp
CavityManager<blockThreads, CavityOp::EV> cavity(
    block, context, shrd_alloc, true);
```

Set it to `true` when fill-in needs the geometry or attributes of deleted
elements. Edge collapse, for example, usually reads the two endpoint positions
of the collapsed edge to place the new vertex.

Set it to `false` when the fill-in only needs the boundary. Edge flip often uses
`false`, because it only connects the two opposite boundary vertices.

---

## **Where to Go Next**

- [`CavityOp`](cavity_op.md) explains how to choose the cavity type.
- [`CavityManager`](cavity_manager.md) documents the device-side API.
- [`DEdgeHandle`](dedge_handle.md) explains directed edges used in fill-in.
- [Host-Side Driver Loop](dynamic_loop.md) shows the host loop around dynamic
  kernels.
