# **`ReduceHandle`**

RXMesh allows running parallel reduction operations over data stored in [attributes](working_with_attributes.md), e.g., dot products and norms, max/min and argmax, or arbitrary binary combines via [CUB](https://docs.nvidia.com/cuda/cub/index.html). The **`ReduceHandle`** class owns temporary device storage and a two-stage reduction pipeline (per-patch GPU reduce, then CUB `DeviceReduce` across patches). Scalar results are returned on the **host**.

`ReduceHandle<T, HandleT>` is strongly typed with the data type the reduction operation will operate on and the type of mesh element. We define the following aliases for convenience: 

- **`VertexReduceHandle<T>`** → `ReduceHandle<T, VertexHandle>`
- **`EdgeReduceHandle<T>`** → `ReduceHandle<T, EdgeHandle>`
- **`FaceReduceHandle<T>`** → `ReduceHandle<T, FaceHandle>`

---

## **Construction**

To perform any reduction operation, you must first construct a `ReduceHandle`. This object encapsulates the temporary memory and configuration needed to execute efficient GPU-based reductions. You can use a single `ReduceHandle` for multiple reduction operations launched one after another. You can create a `ReduceHandle` in two ways:


??? note "`ReduceHandle(const Attribute<T, HandleT>& attr)`"
    Use when you already have an attribute on the mesh.

??? note "`ReduceHandle(const RXMesh& rx)`"
    Works with **`RXMeshStatic`** or **`RXMeshDynamic`**.

Copy construction is allowed (`ReduceHandle(const ReduceHandle&)` defaults). The destructor releases internal temporary allocations.

```cpp
RXMeshStatic rx("mesh.obj");
auto v = rx.add_vertex_attribute<float>("v", 3, DEVICE);

ReduceHandle reduce_handle(*v);  // C++17: class template argument deduction from Attribute<float, VertexHandle>
```

---

## Dot Product

Pairwise inner product over matching mesh elements.
```cpp
auto v1 = rx.add_vertex_attribute<float>("v1", 3, DEVICE);
auto v2 = rx.add_vertex_attribute<float>("v2", 3, DEVICE);
// ... fill v1 and v2 on DEVICE ...

ReduceHandle rh(*v1);
float inner = rh.dot(*v1, *v2);
```

??? note "`dot(attr1, attr2, attribute_id, stream)`"
    ```cpp
    T dot(const Attribute<T, HandleT>& attr1,
          const Attribute<T, HandleT>& attr2,
          uint32_t                     attribute_id = INVALID32,
          cudaStream_t                 stream       = NULL);
    ```

    - **`attribute_id`**: reduce a single component column, or **`INVALID32`** for all columns together.
    - **`stream`**: CUDA stream for the device work.



---

## L2 Norm

Compute the squared L2 norm of an attribute:

```cpp
auto v = rx.add_vertex_attribute<float>("v", 3, DEVICE);
// ... fill v on DEVICE ...
ReduceHandle reduce_handle(*v);
float n = reduce_handle.norm2(*v);
```

??? note "`norm2(attr, attribute_id, stream)`"
    ```cpp
    T norm2(const Attribute<T, HandleT>& attr,
            uint32_t                     attribute_id = INVALID32,
            cudaStream_t                 stream       = NULL);
    ```

    Same as in `dot()`



---

## ArgMin / ArgMax

Returns the value and the corresponding handle of min/max value of the attribute:

```cpp
auto kv = reduce_handle.arg_max(attr);
// kv.key   — handle of the element with maximum value
// kv.value — corresponding scalar / component value
```

??? note "`arg_max` / `arg_min`"
    ```cpp
    KeyValue arg_max(const Attribute<T, HandleT>& attr,
                     uint32_t                     attribute_id = INVALID32,
                     cudaStream_t                 stream       = NULL);

    KeyValue arg_min(const Attribute<T, HandleT>& attr,
                     uint32_t                     attribute_id = INVALID32,
                     cudaStream_t                 stream       = NULL);
    ```

    Same as in `dot()`



---

## Custom Reduction

Generic reduction with a binary associative operator and neutral initial value `init`. You can use CUB’s functors (**`cub::Sum()`**, **`cub::Max()`**, **`cub::Min()`**, …) or a small device functor with `T operator()(const T& a, const T& b) const`.

```cpp
auto e = rx.add_edge_attribute<uint32_t>("e", 3, DEVICE);
// ... fill e on DEVICE ...

ReduceHandle reduce_handle(*e);
uint32_t m = reduce_handle.reduce(*e, cub::Max(), uint32_t(0));
```

User-defined reduction operation functor:

```cpp
struct CustomMin
{
    template <typename U>
    __device__ __forceinline__ U operator()(const U& a, const U& b) const
    {
        return (b < a) ? b : a;
    }
};
```

??? note "`reduce(attr, reduction_op, init, attribute_id, stream)`"
    ```cpp
    template <typename ReductionOp>
    T reduce(const Attribute<T, HandleT>& attr,
             ReductionOp                  reduction_op,
             T                            init,
             uint32_t                     attribute_id = INVALID32,
             cudaStream_t                 stream       = NULL);
    ```
    
    - **`init`**: identity for the operator (e.g. **`0`** for sum, **`0`** for **`cub::Max()`** on unsigned types as in the tests).
    - **`attribute_id`**: reduce a single component column, or **`INVALID32`** for all columns together.
    - **`stream`**: CUDA stream for the device work.