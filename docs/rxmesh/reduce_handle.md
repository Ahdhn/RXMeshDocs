# **Reduction**

RXMesh provides a set of efficient [reduction operations](https://en.wikipedia.org/wiki/Reduction_operator) on mesh attributes. These operations are essential in many geometry processing and simulation pipelines—for example, to compute energy terms, find maximum curvature, or calculate norms. Reductions in RXMesh are built on top of [CUB](https://docs.nvidia.com/cuda/cub/index.html), NVIDIA’s high-performance library for parallel primitives. All reductions operate over [Attributes](attributes.md), and the results are returned on the host.

To perform reductions, use the `ReduceHandle` class. It encapsulates memory management and provides an interface for common operations like sum, dot product, L2 norm, and more.

An example of computing the dot product between two mesh attributes:

```cpp
RXMeshStatic rx("mesh.obj");
auto v1 = rx.add_vertex_attribute<float>("v1", 3);
auto v2 = rx.add_vertex_attribute<float>("v2", 3);

// Populate v1 and v2 
//....

//Reduction handle 
ReduceHandle reduce(v1);

//Dot product between two attributes. Results are returned on the host 
float dot_product = reduce.dot(v1, v2);
```

---

## **`ReduceHandle`**

To perform any reduction operation in RXMesh, you must first construct a `ReduceHandle`. This object encapsulates the temporary memory and configuration needed to execute efficient GPU-based reductions. You can create a `ReduceHandle` in two ways:

```cpp
ReduceHandle(const Attribute<T, HandleT>& attr) 
ReduceHandle(const RXMesh& rx) 
```
Internally, `ReduceHandle` manages the necessary memory and launch parameters to ensure reductions run efficiently across patches and elements.

---

## **Dot Product**

Compute the dot product between two attributes:

```cpp
T dot(const Attribute<T, HandleT>& attr1, 
      const Attribute<T, HandleT>& attr2, 
      uint32_t attribute_id = INVALID32)
```

`attribute_id` lets you reduce a specific column in the attribute (e.g., only the x-component of a 3D vector). If left as `INVALID32`, the full dot product across all components is computed.

---

## **L2 Norm**

Compute the squared L2 norm of an attribute:

```cpp
T norm2(const Attribute<T, HandleT>& attr, uint32_t attribute_id = INVALID32)
```

This is equivalent to `dot(attr, attr)` but often optimized.

---

## **ArgMin / ArgMax**

Returns the value and the corresponding handle that minimizes/maximizes the attribute:

```cpp
KeyValue arg_min(const Attribute<T, HandleT>& attr, uint32_t attribute_id = INVALID32) 
KeyValue arg_max(const Attribute<T, HandleT>& attr, uint32_t attribute_id = INVALID32)
```

Where: `using KeyValue = KeyValuePair<HandleT, T>`

---

## ** Custom Reduction** 

Use `reduce()` to perform user-defined reductions:

```cpp
T reduce(const Attribute<T, HandleT>& attr, 
         ReductionOp reduction_op, T init, 
         uint32_t attribute_id = INVALID32)
```

`ReductionOp` is a functor implementing:

```cpp
struct CustomMin {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return (b < a) ? b : a;
    }
};
```

`init` is the initial value (e.g., 0 for sum, 1 for product, large value for min).

You can also use built-in functors from CUB, e.g., `cub::Sum()`, `cub::Max()`, `cub::Min()`.