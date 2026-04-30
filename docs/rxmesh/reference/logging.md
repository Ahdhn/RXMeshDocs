# **Logging**

RXMesh ships with a tiny logging and error-reporting layer built on top of [spdlog](https://github.com/gabime/spdlog). It is host-side only, used by the library internally to report errors and timings, and by you to instrument your own host code.

The logger is **not initialized automatically** unless you call `rx_init` (see [Initialization](#initialization)). After initialization you can emit messages with the `RXMESH_*` macros and wrap CUDA / cuSPARSE / cuSOLVER / cuBLAS calls with the `*_ERROR` macros to get uniform error messages and termination on failure.

---

## **Initialization**

The recommended entry point is `rx_init`, which initializes logging and picks a CUDA device in one call.

??? note "`void rx_init(int device_id, spdlog::level::level_enum level = spdlog::level::info)`"
    Initializes the logger at the requested level and, when `device_id >= 0`, selects and queries the CUDA device via `cuda_query(device_id)`. Call this once at the start of `main` before any other RXMesh API. See [Building your first app](../../getting-started/first-app.md) for the full bootstrapping pattern.

If you want just the logger without touching CUDA, use `Log::init` directly:

??? note "`static void Log::init(spdlog::level::level_enum level = spdlog::level::trace)`"
    Creates the internal `spdlog::logger` named `"RXMesh"` with two sinks (stdout-color and the file `RXMesh.log` in the working directory), sets its level, and registers it globally. The stdout sink uses the pattern `"%^[%T] %n: %v%$"`; the file sink uses `"[%T] [%l] %n: %v"`. Safe to call once; subsequent calls would try to re-register the logger.

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
    Same two-entry pattern as `WARN`, but at `error` level. Does **not** call `exit`, so control flow continues. Used extensively inside `prepare_launch_box` and the sparse/dense matrix code before returning.

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

These wrap one CUDA-ecosystem call each and, on failure, log the source location, the provider-specific error string, and call `exit(EXIT_FAILURE)`. They are designed to be idiomatic one-liners:

??? note "`CUDA_ERROR(err)`"
    Wraps a `cudaError_t`. On error: logs `"CUDA ERROR: <cudaGetErrorString(err)>"` and exits.

??? note "`CUSPARSE_ERROR(err)`"
    Wraps a `cusparseStatus_t`. On error: logs `"CUSPARSE ERROR: <cusparseGetErrorString(err)>"` and exits.

??? note "`CUSOLVER_ERROR(err)`"
    Wraps a `cusolverStatus_t`. Logs a human-readable cuSOLVER status name and exits.

??? note "`CUBLAS_ERROR(err)`"
    Wraps a `cublasStatus_t`. Logs `"CUBLAS ERROR: <cublasGetStatusString(err)>"` and exits.

??? note "`CUDSS_ERROR(err)`"
    Only available when RXMesh is built with `USE_CUDSS`. Wraps a `cudssStatus_t` and logs the matching error string.

Usage:

```cpp
CUDA_ERROR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
CUDA_ERROR(cudaStreamSynchronize(stream));
```

---

## **Device-Side Diagnostics**

??? note "`myAssert(condition)`"
    A CUDA-aware `assert` replacement. Inside device code, on failure it prints the condition, file, line, `blockIdx.x`, and `threadIdx.x`, and triggers a trap (`asm("trap;");`). On the host, it only prints the failure. Use inside `__device__` and `__host__ __device__` code where plain `assert` is inconvenient.

---

## **Resource Helpers**

??? note "`GPU_FREE(ptr)`"
    Frees a device pointer if it is non-null and resets it to `nullptr`, all under `CUDA_ERROR`. Writing raw `cudaFree` is error-prone because of the double-free / lingering-pointer pitfall; `GPU_FREE` eliminates both. `ptr` must be an lvalue (the macro assigns to it).

```cpp
float* d_buf = nullptr;
CUDA_ERROR(cudaMalloc(&d_buf, n * sizeof(float)));
// ...
GPU_FREE(d_buf);  // safe even if called twice
```