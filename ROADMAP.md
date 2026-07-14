# TileGrad Roadmap

TileGrad should be a thin, inspectable tile programming layer over tinygrad UOps.

The long-term direction is:

```text
global tile -> shared tile -> register fragment -> mma -> global tile store
```

TileGrad should own the explicit schedule. tinygrad should provide UOp lowering, backend codegen, and runtime integration.

## Current Position

TileGrad is already a useful proof-of-concept:

- It has a small IR and validator.
- It lowers explicit `grid`, `threads`, `range`, `reduce`, `unroll`, `shared`, `register`, and `barrier` to tinygrad UOps.
- It has correctness coverage for copy, shared memory, guarded loads/stores, reductions, fragments, edge GEMM, and grid/thread GEMM.
- It defaults to `KernelInfo(opts_to_apply=())`, which is the right default for an explicit scheduling DSL.
- It has a first `TileView` MVP: buffers carry `dtype`, `scope`, and `stride` metadata, and `copy(...)` accepts tile views while lowering through the existing scalar IR.

The main gap is that TileGrad tile operations still immediately scalar-expand into the existing IR. The next step is not a big compiler stack; it is a stable tile IR node for copy with scalar fallback.

## Status Snapshot

Completed:

- Core scalar `KernelBuilder`, IR, validator, and tinygrad UOp lowerer.
- Explicit schedule lowering for `grid`, `threads`, serial ranges, reduce ranges, unroll ranges, shared/register allocations, barriers, and guarded memory operations.
- Fragment GEMM scalar fallback and grid/thread GEMM coverage.
- Default TileGrad lowering opts out of tinygrad heuristic scheduling with `KernelInfo(opts_to_apply=())`.
- Current tinygrad UOp API compatibility fixes for dtype scalar access, pointer indexing, and explicit kernel info.
- `BufferRef` metadata for `dtype`, `scope`, and default 2D stride.
- `TileView` metadata object with origin, shape, stride, bounds, mask, and layout fields.
- `BufferRef.tile(...)`, `KernelBuilder.shared(...)`, and `KernelBuilder.register(...)` helpers.
- `copy(...)` accepts `TileView` inputs and outputs and lowers to the current scalar copy IR.
- A TileView shared-copy example and TileView copy tests.

Partially complete:

- Phase 1 core stability is mostly complete, but public debug/render helpers and a dedicated tinygrad compatibility adapter are still pending.
- Phase 2 TileView MVP has copy support, but still needs hardening around validation, edge cases, and a TileView-based GEMM example.

Not started:

- Tile-level IR nodes such as `TileCopy`.
- Layout/coalescing/vectorized copy policies.
- `TileMMA` contract and explicit `Ops.WMMA` lowering.
- Real pipeline semantics.
- Autotuning.

## Design Principles

- Keep TileGrad thin: `KernelBuilder / TileView -> TileGrad IR -> tinygrad UOps`.
- Preserve the schedule the user wrote.
- Use tinygrad as backend/codegen/runtime, not as the high-level scheduler for explicitly scheduled kernels.
- Prefer one good abstraction over many speculative abstractions.
- Keep scalar fallback paths before optimized lowering.
- Make every layer inspectable.
- Let users escape to lower-level IR/UOps when needed.

## Non-Goals For Now

- Do not replace tinygrad backend codegen.
- Do not infer tiled schedules from ordinary tinygrad Tensor programs.
- Do not build a decorator or Python AST frontend before the tile IR is stable.
- Do not build a custom optimizer before there is one fast kernel to optimize.
- Do not add many tile subclasses until one `TileView` abstraction proves insufficient.
- Do not rely on tinygrad heuristic scheduling for explicitly scheduled TileGrad kernels.

## Architecture Boundary

The key boundary should remain:

```text
TileGrad user API -> TileGrad IR -> tinygrad UOps -> tinygrad codegen/runtime
```

TileGrad should own:

- `grid` and `threads` structure
- shared/register/global memory intent
- barriers and ordering
- tile shapes and masks
- fragment/MMA intent
- pipeline structure once implemented

tinygrad should own:

- UOp simplification and validation
- GPU dimension lowering
- renderer/backend lowering
- compilation and execution
- low-level backend details

## Phase 1: Core Stability

Goal: make the existing scalar builder and lowerer easier to evolve without adding a large abstraction layer.

Status: mostly complete.

Work:

- Done: keep `KernelBuilder` as the core frontend.
- Done: keep `opts_to_apply=()` as the default for TileGrad-lowered kernels.
- Done: add explicit `dtype`, `scope`, `shape`, and `stride` metadata to `BufferRef`.
- Add public debug helpers for TileGrad IR, expanded IR, lowered UOps, and generated source.
- Add a small tinygrad UOp adapter only for APIs that have already shown churn: scalar dtype, index/load/store, placeholders, ranges, and `KernelInfo`.

Success criteria:

- Existing test suite passes.
- Raw UOp examples run.
- Debug helpers work on copy, shared copy, and canonical tiled GEMM.
- Compatibility code is small and does not hide normal tinygrad UOp usage.

## Phase 2: TileView MVP

Goal: introduce one tile abstraction without creating a large class hierarchy.

Status: copy path implemented, hardening still pending.

Start with a single `TileView` object that carries metadata:

- buffer
- origin
- shape
- stride
- bounds
- mask
- layout
- scope

Target API:

```python
A = k.buffer("A", shape=(M, K), dtype="float16")
AS = k.shared("AS", shape=(BM, BK), dtype="float16")

a_tile = A.tile(origin=(bm * BM, ko * BK), shape=(BM, BK), bounds=(M, K))
k.copy(a_tile, AS.tile())
```

Implementation notes:

- Done: do not add `SharedTile`, `RegisterTile`, or `FragmentTile` classes yet.
- Done: represent scope through the underlying buffer/allocation metadata.
- Done: lower tile copies to the existing scalar IR first.
- Done: keep current scalar indexing APIs working.

Success criteria:

- Done: `BufferRef.tile(...)` exists.
- Done: `k.copy(...)` accepts `TileView` inputs and outputs.
- Done: at least one copy example uses `TileView`.
- Pending: one GEMM example uses `TileView`.
- Pending: generated scalar IR is equivalent to the current handwritten scalar version across the full existing copy/GEMM suite.

## Phase 3: Tile IR With Scalar Fallback

Goal: give tile operations stable IR nodes while preserving correctness through scalar fallback.

Add tile-level IR incrementally:

```text
TileCopy
TileFill
TileStore
TileLoad
TileMMA
Pipeline
```

Do not add every node at once. Start with `TileCopy`, because it maps directly to existing `copy(...)` behavior.

Lowering path:

```text
User tile API -> Tile IR -> scalar fallback IR -> tinygrad UOps
```

Later optimized path:

```text
User tile API -> Tile IR -> direct vectorized/WMMA UOps
```

Success criteria:

- Tile IR validates shapes, bounds, dtypes, scopes, and layouts.
- `TileCopy` scalar fallback passes existing copy and GEMM tests.
- Tile IR can be printed independently from scalar-expanded IR.

## Phase 4: Layout And Memory Movement

Goal: make tile movement explicit and optimizable before relying on tensor cores.

Focus areas:

- Coalesced global loads/stores.
- Vectorized copy where legal.
- Shared-memory layout metadata.
- Padding and swizzle hooks.
- Masked edge tiles.
- `copy(..., fill=0)` as a tile primitive, not just scalar loop sugar.

Success criteria:

- `TileCopy` has explicit layout/coalescing policy hooks.
- Shared tile layout is represented in IR.
- Edge/tail copy tests cover masked loads and stores.
- Copy microbenchmarks exist.

## Phase 5: MMA And WMMA

Goal: turn fragment/MMA intent into a real performance path.

Step 1: define the `TileMMA` contract.

- Validate shape compatibility.
- Validate dtype compatibility.
- Validate layout requirements.
- Keep scalar fallback.

Step 2: lower one supported case to tinygrad `Ops.WMMA`.

- Start with one backend and dtype, likely CUDA `float16 x float16 -> float32`.
- Require exact supported tile shapes at first.
- Keep unsupported shapes/dtypes/devices on scalar fallback.
- Add tests that inspect lowered UOps for `Ops.WMMA`.

Success criteria:

- At least one GEMM path emits `Ops.WMMA`.
- Unsupported cases fall back cleanly.
- WMMA path beats scalar fragment GEMM.
- Benchmarks compare tinygrad matmul, scalar TileGrad GEMM, and WMMA TileGrad GEMM.

## Phase 6: Real Pipeline

Goal: turn `pipelined(...)` from syntax into scheduling semantics.

First target: synchronous double-buffered pipeline.

Later target: async copy/wait groups if tinygrad exposes a stable primitive.

Target API:

```python
with k.pipelined("ko", KTILES, stages=2) as ko:
  k.copy(...)
  k.copy(...)
  k.mma(...)
```

Work:

- Double-buffer shared memory.
- Separate producer and consumer phases.
- Insert barriers/waits.
- Validate stage counts and buffer lifetimes.

Success criteria:

- Functional double-buffered shared-memory pipeline.
- Correctness across K tails and M/N edges.
- Benchmarks against the non-pipelined GEMM path.

## Phase 7: Autotuning

Goal: search TileGrad schedule parameters, not arbitrary tinygrad heuristic rewrites.

Search space:

- `BM`, `BN`, `BK`
- thread layout
- number of pipeline stages
- layout/swizzle
- WMMA variant where applicable

Success criteria:

- Small autotuning API.
- Benchmark cache by device, dtype, and shape.
- GEMM tuning examples.
- Reproducible benchmark output.

## Priority Order

1. Finish TileView MVP hardening.
2. Add debug/render tools.
3. Add a small tinygrad compatibility adapter.
4. Add `TileCopy` IR with scalar fallback.
5. Add layout and memory movement policy.
6. Add `TileMMA` contract with scalar fallback.
7. Add explicit `Ops.WMMA` lowering for one case.
8. Add real pipelining.
9. Add autotuning.

## Immediate Next Milestone

Finish and harden the TileView MVP:

- Validate TileView source/destination shape mismatches.
- Preserve explicit empty-shape errors in `copy(...)`.
- Add TileView edge-case tests for shape mismatch, bounds, masks, and destination guards.
- Rewrite one GEMM example to use `TileView` while preserving current scalar fallback behavior.
- Run the full test suite and the TileView examples.

After that, add `TileCopy` as the first tile-level IR node and lower it back to the current scalar IR before attempting vectorized copies or WMMA.
