# **Handles**

Every mesh element in RXMesh, i.e., vertex, edge, or face, is identified by a **handle**. A handle is a 64-bit value that encodes the **patch** the element lives in together with its **local index** within that patch. Handles are the primary way to reference mesh elements in [`for_each`](../static/for_each.md), [`run_kernel`](../static/run_kernel.md), [attributes](../static/working_with_attributes.md), [dense](../matrix/dense_matrices.md) / [sparse](../matrix/sparse_matrices.md) matrices, and the [AD problem](../diff/ad.md) layer.

---

## **Handle Types**

`VertexHandle`, `EdgeHandle`, and `FaceHandle` are **distinct `struct` types** (not typedefs) that have the same API. Using distinct types keeps attribute access strongly typed, i.e., a `VertexAttribute` can only be indexed by a `VertexHandle`, a mistake like `face_attr(vh)` will result into a compile error.

```cpp
VertexHandle vh = ...;
EdgeHandle   eh = ...;
FaceHandle   fh = ...;
```

---

## **Memory Layout**

Each handle stores a single `uint64_t` internally:

```
+------------------------------+-------------------------------+
|         patch_id (32)        |   unused  |   local_id (16)   |
+------------------------------+-----------+-------------------+
 63                          32           15                   0
```

Each handle is **8 bytes**, trivially copyable, and safely captured by value in device lambdas. The upper 32 bits encode `patch_id`, and the lower 16 bits encode `local_id` within the patch. An invalid or default-constructed handle is `INVALID64` (all bits set). Since handles are not contiguous index space, code that needs a flat index should use [Indexing](../static/indexing.md) (`linear_id` / `map_to_global`).

---

## **Construction**

```cpp
VertexHandle vh1;                                 // invalid / default
VertexHandle vh2(patch_id, LocalVertexT(l_id));   // explicit
VertexHandle vh3(uint64_t_handle);                // explicit, from packed id
```

??? note "`VertexHandle()` / `EdgeHandle()` / `FaceHandle()`"
    Default constructor. Initializes the handle to `INVALID64`, i.e., `is_valid()` returns `false`.

??? note "`VertexHandle(uint32_t patch_id, LocalVertexT local)`"
    Builds a handle from a patch id and a strongly typed local index. The analogous overloads exist for `EdgeHandle(uint32_t, LocalEdgeT)` and `FaceHandle(uint32_t, LocalFaceT)`.

??? note "`explicit VertexHandle(uint64_t packed)`"
    Builds a handle from a pre-packed 64-bit id. Marked `explicit` so that arbitrary integers do not silently convert.

All constructors are `constexpr __device__ __host__`.

---

## **Accessors**

Given a handle, you can read its components, check validity, and compare it to another handle:

```cpp
if (vh.is_valid()) {
    uint32_t p = vh.patch_id();
    uint16_t l = vh.local_id();
    uint64_t uid = vh.unique_id();
}
```

??? note "`bool is_valid() const`"
    Returns `true` if the handle is not `INVALID64`, commonly used as guard inside iterator loops.

??? note "`uint32_t patch_id() const`"
    Upper 32 bits. Identifies which [patch](../../getting-started/concepts.md#patches) the element belongs to.

??? note "`uint16_t local_id() const`"
    Lower 16 bits. The element's index within its patch.

??? note "`uint64_t unique_id() const`"
    The raw packed 64-bit value. Convenient as a key in host-side maps or as an identifier for logging.

??? note "`std::pair<uint32_t, uint16_t> unpack() const`"
    Returns `{patch_id, local_id}` in one call.

??? note "`operator==(other) const` / `operator!=(other) const`"
    Handle equality is equality of the underlying 64-bit id, so two handles are equal iff they refer to the same element. Comparison only makes sense between handles of the same type.

All accessors are `constexpr __device__ __host__` and compile to a couple of bit operations.