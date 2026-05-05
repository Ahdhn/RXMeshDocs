# **Logging**

RXMesh ships with a tiny logging and error-reporting layer built on top of [spdlog](https://github.com/gabime/spdlog). It is host-side only, used by the library internally to report errors and timings, and by you to instrument your own host code.

The logger is **not initialized automatically** unless you call `rx_init` (see [Initialization](#initialization)). After initialization you can emit messages with the `RXMESH_*` macros and wrap CUDA / cuSPARSE / cuSOLVER / cuBLAS calls with the `*_ERROR` macros to get uniform error messages.

---

## **Initialization**

The recommended entry point is `rx_init`, which initializes logging and picks a CUDA device in one call.

??? note "`void rx_init(int device_id, spdlog::level::level_enum level = spdlog::level::info)`"
    Initializes the logger at the requested level and, when `device_id >= 0`, selects and queries the CUDA device via `cuda_query(device_id)`. Call this once at the start of `main` before any other RXMesh API. See [Building your first app](../../getting-started/first-app.md) for the full bootstrapping pattern.

If you want just the logger without touching CUDA, use `Log::init` directly:

??? note "`static void Log::init(spdlog::level::level_enum level = spdlog::level::trace)`"
    Creates the internal `spdlog::logger` named `"RXMesh"` with two sinks (stdout-color and the file `RXMesh.log` in the working directory), sets its level, and registers it globally. Safe to call once. Subsequent calls would try to re-register the logger.

??? note "`static void Log::set_level(spdlog::level::level_enum level)`"
    Changes the verbosity at runtime and configures `flush_on` so every message at or above `level` is flushed immediately. Typical values from `spdlog::level`: `trace`, `debug`, `info`, `warn`, `err`, `critical`, `off`.

??? note "`static std::shared_ptr<spdlog::logger>& Log::get_logger()`"
    Returns the underlying logger. Useful if you want to call spdlog directly (e.g., add another sink) rather than through the `RXMESH_*` macros.

---

## **Message Macros**

All macros accept spdlog's `fmt`-style format string and arguments. They forward to the global `Log::get_logger()`, so `Log::init` (directly or through `rx_init`) must run first.

??? note "`RXMESH_TRACE(...)` / `RXMESH_INFO(...)`"
    Informational messages. `TRACE` for fine-grained instrumentation, `INFO` for user-visible progress.

??? note "`RXMESH_WARN(...)`"
    Emits two log entries at `warn` level: first the source location (`Line {} File {}`), then the formatted message. Use for recoverable issues the user should see.

??? note "`RXMESH_ERROR(...)`"
    Same two-entry pattern as `WARN`, but at `error` level. Does **not** call `exit`, so control flow continues. 

??? note "`RXMESH_CRITICAL(...)`"
    Same two-entry pattern at `critical` level. The caller is expected to handle termination if desired; the macro itself does not call `exit`.

Example:

```cpp
RXMESH_INFO("Loaded mesh with {} vertices and {} faces",
            rx.get_num_vertices(),
            rx.get_num_faces());
```

---

## **Error-Checking Wrappers**

These wrap one CUDA API call and, on failure, log the source location, the CUDA error string, and call `exit(EXIT_FAILURE)`. They are designed to be idiomatic one-liners:

??? note "`CUDA_ERROR(err)`"
    Wraps a `cudaError_t`. On error: logs `"CUDA ERROR: <cudaGetErrorString(err)>"` and exits.

Usage:

```cpp
CUDA_ERROR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
CUDA_ERROR(cudaStreamSynchronize(stream));
```

---

## **Resource Helpers**

??? note "`GPU_FREE(ptr)`"
    Frees a device pointer if it is non-null and resets it to `nullptr`, all under `CUDA_ERROR`. Writing raw `cudaFree` is error-prone because of the double-free. `ptr` must be an lvalue (the macro assigns to it).

```cpp
float* d_buf = nullptr;
CUDA_ERROR(cudaMalloc(&d_buf, n * sizeof(float)));
// ...
GPU_FREE(d_buf);  // safe even if called twice
```