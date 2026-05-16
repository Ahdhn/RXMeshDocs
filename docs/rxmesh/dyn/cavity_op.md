# **`CavityOp`**

`CavityOp` tells `CavityManager` what kind of cavity to create around a seed
element. It is a compile-time template argument:

```cpp
CavityManager<blockThreads, CavityOp::E> cavity(
    block, context, shrd_alloc, true);
```

The first letter is the seed element type. The second letter, when present, describes the neighboring element type pulled into the deleted region. For example, `EV` starts from an edge and removes the edge-vertex neighborhood. 

---

## **Enum**

From `include/rxmesh/types.h`:

```cpp
enum class CavityOp
{
    V  = 0,
    E  = 1,
    F  = 2,
    VV = 3,
    VE = V,
    VF = V,
    FV = 4,
    FE = 5,
    FF = FE,
    EV = 6,
    EE = 7,
    EF = E,
};
```

Some names are aliases because deleting one type already implies deleting another. For example, removing a vertex also removes the incident edges and faces, so `VE` aliases `V`.

---

## **Common Choices**

Most dynamic mesh kernels start with these choices:

| Edit | `CavityOp` | Seed | Fill-in example |
|------|------------|------|--------------|
| Edge split | `E` | `EdgeHandle` | Add one midpoint vertex, connect it to the four boundary vertices, and create four faces. |
| Edge flip | `E` | `EdgeHandle` | Add the opposite diagonal and create the two replacement faces. |
| Edge collapse | `EV` | `EdgeHandle` | Add one merged vertex, connect it to the boundary loop, and create a fan of faces. |
| Face-edge local edit | `FE` | `FaceHandle` | Remove a face-edge stencil and refill it with custom topology. |

The same cavity operation can support different edits. Edge split and edge flip both use `CavityOp::E` since they differ only in how the fill-in reconnects the four boundary vertices.

---

## **Seed Handle Type**

The seed passed to `create(...)` should match the source element of the operation:

| Source | Example ops | Seed handle |
|--------|-------------|-------------|
| Vertex | `V`, `VV`, `VE`, `VF` | `VertexHandle` |
| Edge | `E`, `EV`, `EE`, `EF` | `EdgeHandle` |
| Face | `F`, `FV`, `FE`, `FF` | `FaceHandle` |

```cpp
for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
    if (edge_is_too_long(eh)) {
        cavity.create(eh);  // CavityOp::E or CavityOp::EV
    }
});
```
