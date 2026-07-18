# TileGrad Roadmap

TileGrad is a thin, inspectable TileLang-inspired programming layer over tinygrad UOps.

The long-term direction is:

```
global tile -> shared tile -> register tile/fragment -> mma -> global tile store
```

TileGrad should own the explicit schedule and tile-level intent. tinygrad should own UOp lowering, backend codegen, compilation, and runtime execution.

## MVP Goal

Make TileGrad a usable, inspectable, slow-but-correct tiled kernel DSL over tinygrad.

MVP means:

- Builder-first API, not a Python AST/decorator frontend yet.
- Static tile shapes and mostly static kernel structure.
- Contiguous tinygrad tensors.
- `float32` correctness path first.
- Explicit `grid(...)`, `threads(...)`, `range(...)`, `shared(...)`, `register(...)`.
- First-class `TileCopy` and `TileMMA` intent in TileGrad IR.
- Scalar fallback lowering for correctness.
- Runtime execution through tinygrad `Tensor.custom_kernel`.
- Debug stages that show tile intent before scalar expansion.
- No performance promises.

## Current Baseline

Already implemented:

- Core `KernelBuilder`, IR, validation, and tinygrad UOp lowerer.
- Runtime wrapper via `run(...)`.
- `grid`, `blocks`, `threads`, `parallel`, serial ranges, reduce ranges, and unroll ranges.
- Shared and register allocations.
- Barriers.
- Guarded loads/stores.
- `TileView` and first-class `TileCopy`.
- `TileCopy` scalar fallback.
- 1D, 2D, and compact 3D copy support.
- Copy origins, strides, bounds, masks, zero-fill, and `coalesced_width` metadata.
- First-class `TileMMA`.
- `TileMMA` scalar fallback.
- `TileMMA` scope and `float32` dtype validation.
- Canonical tiled GEMM using `TileView -> TileCopy -> TileMMA -> guarded store`.
- M/N edge tile correctness coverage.
- K-tail correctness coverage.
- Debug stages: tile IR, expanded tile copies, expanded fragments/MMA, register unroll, scalar IR.
- Optional lowered tinygrad UOp inspection.
- Canonical GEMM example self-checks and prints inspect-stage intent.
- Benchmark harness comparing tinygrad matmul and TileGrad GEMM variants.

Known MVP gaps:

- `pipelined(...)` is syntax-only.
- `coalesced_width` is metadata-only.
- No vectorized copy lowering.
- No WMMA/intrinsic lowering.
- No real async copy, wait groups, double buffering, or software pipeline scheduling.
- No CI workflow yet.

## Phase 1: Declare And Stabilize The MVP

Goal: make the current slow path clearly usable and documented.

Work:

- Keep README canonical GEMM description matched to implementation.
- Keep benchmark launch labels reporting `threads=(1)` for current canonical GEMM.
- Explicitly state that current GEMM is correctness-first and scalar-expanded.
- Keep `k.gemm(...)` as the public tile-level GEMM API.
- Keep `k.copy(...)` as the single synchronous data movement primitive.
- Keep `k.pipelined(...)` documented as syntax-only.
- Keep `parallel(...)` as a compatibility alias for `threads(...)`, but recommend `threads(...)`.

Success criteria:

- README and roadmap agree.
- New user can identify the intended MVP path in under five minutes.
- Existing tests still pass.
- No new optimization-specific code is required.

## Phase 2: Improve MVP Examples

Goal: make one blessed path obvious.

Work:

- Treat `examples/builder_canonical_tiled_gemm.py` as the canonical MVP example.
- Keep the canonical GEMM example self-checking against a Python reference.
- Keep the canonical GEMM example printing inspect stages.
- Keep showing that `TileCopy` and `TileMMA` exist in `tile_ir`.
- Keep showing that `TileCopy` and `TileMMA` are gone in `scalar_ir`.
- Document the recommended smoke-test commands.

Success criteria:

- Users can run copy, canonical GEMM, and inspect examples.
- The inspect output demonstrates tile-level intent before fallback expansion.
- The examples match the implementation exactly.

## Phase 3: MVP Hardening

Goal: make the slow path reliable enough to optimize later.

Work:

- Keep dtype metadata tracking for buffers where needed.
- Keep `TileMMA` dtype validation.
- Decide the initial supported dtype contract, likely `float32` only for MVP.
- Improve user-facing validation errors for unsupported shapes, scopes, dtypes, layouts, and non-zero fills.
- Keep tests for invalid `TileMMA` dtype combinations.
- Keep tests for canonical GEMM inspection.
- Add focused regression tests around nested ranges plus tile ops.
- Document known register-tile limitations around dynamic range indexing.
- Add minimal CI running `python3 -m pytest tests/`.

Success criteria:

- Full test suite passes locally and in CI.
- Unsupported MVP cases fail clearly.
- Current API behavior is stable enough to build examples and docs around.

## Phase 4: MVP Release Criteria

TileGrad reaches MVP when:

- `python3 -m pytest tests/` passes.
- `python3 examples/builder_copy.py` runs.
- `python3 examples/builder_grid_threads_copy.py` runs.
- `python3 examples/builder_tilecopy_inspect.py` runs.
- `python3 examples/builder_canonical_tiled_gemm.py` runs and self-validates.
- README accurately explains the current slow lowering path.
- Roadmap accurately separates implemented work from deferred optimization work.

## Non-Goals Before MVP

Do not prioritize:

- Python AST/decorator frontend.
- TileLang-compatible `@jit` or `T.prim_func` syntax.
- Automatic scheduling.
- Autotuning.
- Async copy.
- Wait groups.
- Real software pipelining.
- Layout inference.
- Swizzled layouts.
- TMA.
- WGMMA.
- `Ops.WMMA`.
- Vectorized copy lowering.
- Backend-specific code injection.
- Custom tinygrad runtime/device backends.
- Autograd.

## Post-MVP Optimization Track

Only after the MVP is documented, tested, and stable:

1. Add `TileCopy` classification.
2. Add scalar, contiguous, coalesced-candidate, and vectorizable-candidate categories.
3. Add vectorized lowering for simple unmasked contiguous copies.
4. Add copy microbenchmarks.
5. Prototype one `Ops.WMMA` lowering path for a narrow CUDA fp16 case.
6. Compare scalar TileGrad GEMM, WMMA TileGrad GEMM, and tinygrad matmul.
7. Add real `pipelined(...)` semantics.
8. Add double buffering.
9. Add async copy and wait-group primitives.
10. Add autotuning for `BM`, `BN`, `BK`, thread layout, pipeline stages, and MMA variants.

## Design Principles

- Keep TileGrad thin.
- Preserve the schedule the user wrote.
- Make tile-level intent visible before lowering.
- Always keep a scalar fallback before adding optimized lowering.
- Prefer one stable abstraction over many speculative ones.
- Use tinygrad for UOps, codegen, compilation, and runtime.
- Avoid depending on tinygrad internals more than necessary.
- Copy TileLang's useful user model, not its backend complexity.
- Optimize only after the correctness/debug story is boring.
