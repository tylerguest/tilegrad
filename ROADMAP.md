# TileGrad Roadmap

TileGrad should become a TileLang-like explicit tile DSL for tinygrad UOps.

The core direction is:

```text
global tile -> shared tile -> register fragment -> mma -> global tile store
```

TileGrad should own the high-level schedule. tinygrad should provide UOp lowering, backend codegen, and runtime integration.

## Current Position

TileGrad is already a useful proof-of-concept:

- It has a small IR and validator.
- It lowers explicit `grid`, `threads`, `range`, `reduce`, `unroll`, `shared`, `register`, and `barrier` to tinygrad UOps.
- It has broad correctness coverage for copy, shared memory, guarded loads/stores, reductions, fragments, edge GEMM, and grid/thread GEMM.
- It defaults to `KernelInfo(opts_to_apply=())`, which is the right default for an explicit scheduling DSL.

The main gap is that TileGrad is still mostly a scalar kernel builder, not yet a first-class tile programming layer.

## Key Findings

- `KernelBuilder` is solid but low-level. `BufferRef` currently tracks only `name` and `shape`, so TileGrad lacks first-class tile/layout/stride/dtype/scope objects.
- `copy(...)` is useful but scalar-expands loops. It is not yet a tile movement primitive with coalescing, vectorization, async, or pipeline semantics.
- `pipelined(...)` currently validates syntax only and behaves like a normal range.
- `FragmentGemm` currently scalar-expands, and intrinsic fragment lowering is not implemented yet.
- tinygrad already has the backend pieces needed: explicit custom kernels, `KernelInfo(opts_to_apply)`, GPU dimension lowering, local/register address spaces, and `Ops.WMMA` support.
- tinygrad's tensor-core path is mostly optimizer-oriented today. TileGrad should eventually build explicit `Ops.WMMA` for scheduled fragments instead of relying on heuristic tensor-core discovery.

## Design Principles

- TileGrad owns explicit scheduling.
- tinygrad is the backend/codegen/runtime layer.
- The default mode should preserve the schedule the user wrote.
- tinygrad heuristic scheduling should be opt-in, not the default for explicitly scheduled TileGrad kernels.
- TileGrad should expose tile-level concepts rather than forcing users to manually write scalar load/store loops.
- Correct scalar fallback should exist before optimized lowering.
- Debuggability should be a core feature.

## Phase 1: Stabilize The Core

Goals:

- Keep `KernelBuilder` as the core frontend.
- Keep `opts_to_apply=()` as the default for TileGrad-lowered kernels.
- Add explicit `dtype`, `scope`, `shape`, and `stride` metadata to `BufferRef`.
- Add a public debug API to print TileGrad IR, expanded IR, lowered UOps, and generated source.
- Add compatibility wrappers around tinygrad UOp API calls so tinygrad churn is isolated in one file.

Deliverables:

- `BufferRef` metadata cleanup.
- `tilegrad.debug` helpers.
- One compatibility module for tinygrad API boundaries.
- Tests for debug/render paths that do not require exact backend source matching.

## Phase 2: Add Tile Views

Add first-class tile objects.

Target API:

```python
A = k.buffer("A", shape=(M, K), dtype="float16")
AS = k.shared("AS", shape=(BM, BK), dtype="float16")

a_tile = A.tile(origin=(bm * BM, ko * BK), shape=(BM, BK), bounds=(M, K))
k.copy(a_tile, AS)
```

Tile views should represent metadata, not immediately scalar loops.

Add:

- `TileView`
- `SharedTile`
- `RegisterTile`
- `FragmentTile`
- `Layout`
- `bounds`
- `stride`
- `mask`
- optional `swizzle`

Deliverables:

- Tile view classes.
- Tile view validation.
- `copy(...)` support for tile views.
- Examples rewritten to use tile views while preserving current behavior.

## Phase 3: Normalize Tile Ops To IR

Add tile-level IR operations before lowering to tinygrad:

```text
TileCopy
TileFill
TileStore
TileLoad
TileMMA
TileBarrier
Pipeline
```

Initially, lower these to the existing scalar IR so correctness remains easy.

The short-term path should be:

```text
User tile API -> Tile IR -> scalar fallback IR -> tinygrad UOps
```

The long-term path should be:

```text
User tile API -> Tile IR -> direct WMMA/vectorized UOps
```

Deliverables:

- Tile IR dataclasses.
- Tile IR validator.
- Scalar fallback lowering for tile operations.
- Tests proving tile ops match existing scalar behavior.

## Phase 4: Real Fragment / WMMA Path

This is the first performance-critical milestone.

tinygrad has `Ops.WMMA` and renderer support for CUDA, AMD, and Metal paths. TileGrad should eventually lower `TileMMA` or selected `FragmentGemm` cases directly to `Ops.WMMA`.

Roadmap:

- Add `TileMMA` IR.
- Keep scalar fallback.
- Add one supported WMMA case first, likely CUDA `float16 x float16 -> float32`.
- Make validation strict: exact shape, dtype, layout, warp/thread count.
- Add tests that inspect lowered UOps for `Ops.WMMA`.
- Add benchmark comparison against tinygrad matmul.

Deliverables:

- One end-to-end WMMA GEMM path.
- Fallback path for unsupported dtypes/shapes/devices.
- Tests for both intrinsic and fallback paths.
- Benchmark results in `benchmarks/bench_gemm.py`.

## Phase 5: Memory Movement

Improve tile movement after the basic WMMA path exists.

Focus areas:

- Vectorized global loads/stores.
- Coalesced copy policies.
- Shared memory layouts.
- Padding/swizzling to avoid bank conflicts.
- Masked edge tiles.
- `copy(..., fill=0)` as a tile primitive, not just scalar loop sugar.

Deliverables:

- Explicit copy layout/coalescing options.
- Shared tile swizzle support.
- Edge tile masking tests.
- Copy microbenchmarks.

## Phase 6: Real Pipeline

Turn `pipelined(...)` into an actual scheduling primitive.

Target API:

```python
with k.pipelined("ko", KTILES, stages=2) as ko:
  k.copy(...)
  k.copy(...)
  k.mma(...)
```

Milestones:

- Double-buffer shared memory.
- Separate producer/consumer phases.
- Insert barriers/waits.
- Later support async copy if tinygrad exposes a stable primitive for it.

Deliverables:

- Functional double-buffered shared-memory pipeline.
- Tests for correctness across tails/edges.
- Benchmarks against the non-pipelined GEMM path.

## Phase 7: Autotuning

Once tile parameters matter, add simple autotuning.

Search over TileGrad schedule parameters, not arbitrary UOp rewrites:

- `BM`, `BN`, `BK`
- thread layout
- number of pipeline stages
- layout/swizzle
- WMMA variant where applicable

Deliverables:

- Small autotuning API.
- Benchmark cache by device, dtype, and shape.
- GEMM tuning examples.
- Reproducible benchmark output.

## Priority Order

1. Debug/render tools.
2. `BufferRef` metadata cleanup.
3. `TileView` API.
4. Tile IR with scalar fallback.
5. Explicit WMMA lowering.
6. Synchronous tiled copy improvements.
7. Real pipelining.
8. Autotuning.

## Immediate Next Step

Implement `TileView` and make `copy(...)` operate on tile views while lowering to the current scalar loop IR.

This moves TileGrad from "scalar builder with tile-like examples" to an actual tile DSL without taking on WMMA complexity immediately.
