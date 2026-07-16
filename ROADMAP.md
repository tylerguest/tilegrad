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
- It has a hardened `TileView` MVP: buffers carry `dtype`, `scope`, and `stride` metadata, and `copy(...)` accepts tile views with shape, bounds, mask, and destination-guard coverage.
- It has a first tile-level IR node: `TileCopy` preserves copy intent in `k.build()` and expands through scalar fallback before tinygrad UOp lowering.

The main gap is no longer preserving copy intent. The next step is to add a thin TileGrad-owned debug artifact and named IR stage inspection before adding layout/coalescing/vectorized copy policies.

This should complement, not replace, tinygrad `DEBUG` and `VIZ`. TileGrad should expose the boundaries it owns: unexpanded TileGrad IR, named TileGrad transform stages, scalar fallback IR, and lowered tinygrad UOps. Backend source/codegen inspection should remain tinygrad-owned unless a stable no-compile source path is available.

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
- `copy(...)` accepts `TileView` inputs and outputs.
- TileView copy hardening for shape mismatches, explicit empty shapes, bounds, masks, and destination guards.
- A TileView shared-copy example, a TileView GEMM example, and TileView copy tests.
- `TileCopy` IR node for copy intent.
- `expand_tile_copies(...)` scalar fallback pass.
- `validate_tile_copy(...)` and a `validate_tile_copies(...)` prepass for unexpanded TileCopy IR before scalar fallback.
- Explicit 2D stride handling for `BufferRef` tuple indexing and TileView/TileCopy fallback indexing.
- A TileCopy inspection example showing tile IR, expanded scalar IR, and runtime output.

Partially complete:

- Phase 1 core stability is mostly complete, but public stage/debug artifacts and a dedicated tinygrad compatibility adapter are still pending.
- Phase 3 Tile IR has `TileCopy` with scalar fallback and validation, but public stage inspection still needs hardening.

Not started:

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

## Debug/Inspect Direction

TileGrad debug tooling should follow a small artifact/stage model similar in spirit to TileLang's exposed lower artifacts and pass inspection tools, but scaled to TileGrad's simpler compiler path.

Target API shape:

```python
from tilegrad.debug import inspect_kernel, ir_stages

dbg = inspect_kernel(k, out_uop, inp_uop)

print(dbg.tile_ir)
print(dbg.scalar_ir)
print(dbg.uops_text())
```

Core objects:

```python
@dataclass(frozen=True)
class IRStage:
  name: str
  kernel: Kernel

@dataclass(frozen=True)
class DebugArtifact:
  tile_ir: Kernel
  stages: tuple[IRStage, ...]
  scalar_ir: Kernel
  uops: UOp | None = None
```

Initial named stages:

- `tile_ir`: the unexpanded `KernelBuilder.build()` result, preserving `TileCopy`.
- `expand_tile_copies`: scalar fallback for tile copy intent.
- `expand_fragments`: scalar fallback for fragment operations.
- `unroll_register_tiles`: register-index unrolling before tinygrad lowering.
- `scalar_ir`: the final TileGrad scalar IR accepted by `lower_kernel(...)`.
- `lowered_uops`: the tinygrad `Ops.SINK` produced from scalar IR when placeholder UOps are supplied.

Non-goals for this milestone:

- Do not build a replacement for tinygrad `DEBUG` or `VIZ`.
- Do not promise generated backend source as a stable TileGrad API.
- Do not add a full pass manager before there are enough TileGrad transforms to justify one.
- Do not add runtime debug printing until there is a concrete TileGrad-side need distinct from normal tensor/output checks.

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
- Done: fix explicit 2D stride handling so `BufferRef` tuple indexing and TileView/TileCopy lowering consistently honor `BufferRef.stride` instead of silently falling back to compact shape-derived strides.
- Add public debug artifacts for unexpanded TileGrad IR, named TileGrad-owned IR stages, scalar fallback IR, and lowered tinygrad UOps.
- Add a small tinygrad UOp adapter only for APIs that have already shown churn: scalar dtype, index/load/store, placeholders, ranges, and `KernelInfo`.

Success criteria:

- Existing test suite passes.
- Raw UOp examples run.
- Debug artifacts work on copy, shared copy, TileView copy, and canonical tiled GEMM.
- tinygrad `DEBUG` and `VIZ` remain the recommended path for generated source/codegen/runtime inspection.
- Compatibility code is small and does not hide normal tinygrad UOp usage.

## Phase 2: TileView MVP

Goal: introduce one tile abstraction without creating a large class hierarchy.

Status: complete for the current MVP.

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
- Done: emit `TileCopy` for tile copies and lower through scalar fallback.
- Done: keep current scalar indexing APIs working.

Success criteria:

- Done: `BufferRef.tile(...)` exists.
- Done: `k.copy(...)` accepts `TileView` inputs and outputs.
- Done: at least one copy example uses `TileView`.
- Done: one GEMM example uses `TileView`.
- Done: TileView edge-case tests cover shape mismatch, bounds, masks, and destination guards.
- Done: generated scalar fallback preserves current copy/GEMM behavior across the existing test suite.

## Phase 3: Tile IR With Scalar Fallback

Goal: give tile operations stable IR nodes while preserving correctness through scalar fallback.

Status: started. `TileCopy` exists, validates before scalar fallback, and runs through scalar fallback; public stage inspection is next.

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

- Done: TileCopy IR validates source/destination buffers, shapes, origins, bounds, masks, guards, strides, fill, and layouts before expansion.
- Done: direct `TileCopy` IR rejects unsupported semantics, including nonzero fill and non-`None` layouts, until those policies have defined lowering behavior.
- Done: `TileCopy` scalar fallback passes existing copy and GEMM tests.
- Done: Tile IR can be printed independently from scalar-expanded IR through the TileCopy inspection example.
- Pending: public debug artifacts expose unexpanded TileGrad IR, named TileGrad transform stages, expanded scalar IR, and lowered UOps.
- Pending: docs explain that generated backend source belongs to tinygrad debug/codegen paths for now.

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

1. Add public debug artifacts and named IR stage inspection for TileGrad-owned lowering boundaries.
2. Add a small tinygrad compatibility adapter.
3. Add layout and memory movement policy hooks for `TileCopy`.
4. Add copy microbenchmarks.
5. Add `TileMMA` contract with scalar fallback.
6. Add explicit `Ops.WMMA` lowering for one case.
7. Add real pipelining.
8. Add autotuning.

## Immediate Next Milestone

Add public debug artifacts and named IR stage inspection:

- Add `tilegrad.debug.inspect_kernel(...)` returning a `DebugArtifact`.
- Add `tilegrad.debug.ir_stages(...)` returning named TileGrad-owned IR snapshots.
- Add convenience wrappers only if they stay thin: `tile_ir(...)`, `scalar_ir(...)`, and `lowered_uops(...)`.
- Add `DebugArtifact.uops_text()` using tinygrad's UOp text rendering.
- Do not expose generated source as a stable TileGrad helper in this milestone; document tinygrad `DEBUG=6` and `VIZ=1` instead.
- Keep `lower_kernel(...)` on the safe path: `TileCopy -> scalar fallback IR -> tinygrad UOps`.
- Run the full test suite plus `examples/builder_tilecopy_inspect.py`, the TileView GEMM example, and representative raw UOp examples.

After that, add layout/coalescing policy hooks to `TileCopy` before attempting vectorized copies, async copies, or WMMA.
