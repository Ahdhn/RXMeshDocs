# **`DEdgeHandle`**

`DEdgeHandle` is a directed edge handle. A normal `EdgeHandle` identifies an undirected edge. A `DEdgeHandle` identifies one of its two orientations.

Dynamic fill-in uses directed edges because `add_face(e0, e1, e2)` needs a consistent triangle winding.

```cpp
DEdgeHandle e0 = cavity.add_edge(v0, v1);
DEdgeHandle e1 = cavity.add_edge(v1, v2);
DEdgeHandle e2 = cavity.add_edge(v2, v0);

FaceHandle f = cavity.add_face(e0, e1, e2);
```

You usually get a `DEdgeHandle` from `CavityManager`:

```cpp
DEdgeHandle boundary = cavity.get_cavity_edge(c, i);
DEdgeHandle created  = cavity.add_edge(src, dst);
```

The boundary edges returned by `get_cavity_edge(...)` are ordered around the
cavity. The edges returned by `add_edge(...)` are oriented from the first vertex
argument to the second.

---

## **Flipping Direction**

Use `get_flip_dedge()` when the next face needs the same underlying edge in the
opposite direction.

```cpp
DEdgeHandle e0 = cavity.add_edge(new_v, boundary_v0);
DEdgeHandle e1 = cavity.add_edge(boundary_v1, new_v);

FaceHandle f0 = cavity.add_face(e0, boundary_edge, e1);

// The next triangle reuses the shared edge in the opposite direction.
DEdgeHandle next_e0 = e1.get_flip_dedge();
```

??? note "`DEdgeHandle get_flip_dedge() const`"
    Returns the same undirected edge with the direction bit flipped.

---

## **Using Edge Attributes**

Attributes live on undirected edges, so convert to `EdgeHandle` before indexing
an edge attribute.

```cpp
DEdgeHandle de = cavity.add_edge(v0, v1);

if (de.is_valid()) {
    EdgeHandle e = de.get_edge_handle();
    edge_status(e) = 1;
}
```

??? note "`EdgeHandle get_edge_handle() const`"
    Returns the undirected edge handle.

??? note "`bool is_valid() const`"
    Returns whether the handle points to a real edge.

??? note "`uint32_t patch_id() const`"
    Patch id of the underlying edge.

??? note "`uint16_t local_id() const`"
    Local id of the underlying undirected edge. A directed edge and its flipped
    version share the same `local_id()`.

??? note "`uint64_t unique_id() const`"
    Packed id including the direction bit.
