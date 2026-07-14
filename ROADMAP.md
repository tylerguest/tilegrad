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

The main gap is that TileGrad is still mostly a scalar kernel builder. The next step is not a big compiler stack; it is a small tile layer on top of the existing scalar IR.

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

Work:

- Keep `KernelBuilder` as the core frontend.
- Keep `opts_to_apply=()` as the default for TileGrad-lowered kernels.
- Add explicit `dtype`, `scope`, `shape`, and `stride` metadata to `BufferRef`.
- Add public debug helpers for TileGrad IR, expanded IR, lowered UOps, and generated source.
- Add a small tinygrad UOp adapter only for APIs that have already shown churn: scalar dtype, index/load/store, placeholders, ranges, and `KernelInfo`.

Success criteria:

- Existing test suite passes.
- Raw UOp examples run.
- Debug helpers work on copy, shared copy, and canonical tiled GEMM.
- Compatibility code is small and does not hide normal tinygrad UOp usage.

## Phase 2: TileView MVP

Goal: introduce one tile abstraction without creating a large class hierarchy.

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

- Do not add `SharedTile`, `RegisterTile`, or `FragmentTile` classes yet.
- Represent scope through the underlying buffer/allocation metadata.
- Lower tile copies to the existing scalar IR first.
- Keep current scalar indexing APIs working.

Success criteria:

- `BufferRef.tile(...)` exists.
- `k.copy(...)` accepts `TileView` inputs and outputs.
- At least one copy example and one GEMM example use `TileView`.
- Generated scalar IR is equivalent to the current handwritten scalar version.

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

1. Debug/render tools.
2. `BufferRef` metadata cleanup.
3. `TileView` MVP.
4. `TileCopy` IR with scalar fallback.
5. Layout and memory movement policy.
6. `TileMMA` contract with scalar fallback.
7. Explicit `Ops.WMMA` lowering for one case.
8. Real pipelining.
9. Autotuning.

## Immediate Next Milestone

Implement the TileView MVP:

- Add `dtype`, `scope`, and `stride` metadata to `BufferRef`.
- Add `BufferRef.tile(...)`.
- Make `k.copy(...)` accept `TileView`.
- Lower tile copies to the current scalar IR.
- Rewrite one copy example and one GEMM example to use `TileView`.

This moves TileGrad from "scalar builder with tile-like examples" to a real tile DSL without prematurely building a compiler tower.
