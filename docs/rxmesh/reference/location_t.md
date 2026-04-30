# **`locationT`**

`locationT` is the small bitmask type that RXMesh uses to say **where** memory is allocated or **which side** of a host/device pair to act on. Every API that touches storage or launches compute, e.g., [attribute construction](../static/managing_attributes.md), [`Attribute::move` / `reset` / `release`](../static/working_with_attributes.md), [`DenseMatrix` construction and `move`](../matrix/dense_matrices.md), and [`RXMeshStatic::for_each_*`](../static/for_each.md#element-wise-for_each), takes a `locationT` argument.

---

## **Definition**

From `include/rxmesh/types.h`:

```cpp
using locationT = uint32_t;

enum : locationT {
    LOCATION_NONE = 0x00,
    HOST          = 0x01,
    DEVICE        = 0x02,
    LOCATION_ALL  = 0x0F,
};
```

The tags are **unscoped**, so you write `HOST`, `DEVICE`, `LOCATION_ALL` directly, without a `locationT::` prefix.

---

## **Values**

| Tag | Bits | Meaning |
|-----|------|---------|
| `LOCATION_NONE` | `0x00` | No storage allocated / do nothing. |
| `HOST` | `0x01` | CPU-side memory. |
| `DEVICE` | `0x02` | GPU-side memory. |
| `LOCATION_ALL` | `0x0F` | Both `HOST` and `DEVICE` (and any future bits). |

There is **no** `NOT_ALLOCATED` constant, "nothing allocated" is `LOCATION_NONE`.

---

## **Bitmask Semantics**

`HOST` and `DEVICE` are distinct bits, so you can combine them with the bitwise OR:

```cpp
auto attr = rx.add_vertex_attribute<float>("x", 3, HOST | DEVICE);
```

Library code tests individual bits with AND, i.e., `(loc & HOST) == HOST` and `(loc & DEVICE) == DEVICE`. This lets a single call affect host only, device only, or both:

```cpp
attr->reset(0.f, DEVICE);           // zero only the device copy
attr->reset(1.f, HOST | DEVICE);    // zero both copies
```

??? note "`std::string location_to_string(locationT location)`"
    Returns `"NONE"`, `"HOST"`, `"DEVICE"`, or `"ALL"` for the four canonical values. Unknown bit combinations log an error and return an empty string, so this is only suitable for the named single-flag values, not arbitrary masks.