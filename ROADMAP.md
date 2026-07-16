# TileGrad Roadmap

TileGrad should be a thin, inspectable tile programming layer over tinygrad UOps.

The long-term direction is:

```text
global tile -> shared tile -> register fragment -> mma -> global tile store
```

TileGrad should own the explicit schedule. tinygrad should provide UOp lowering, backend codegen, and runtime integration.

## MVP Goal

Make TileGrad a complete, inspectable, slow-but-correct tiled kernel DSL over tinygrad before optimizing.

A user can write a canonical tiled GEMM using:

```text
grid -> threads -> TileCopy global/shared -> barrier -> TileMMA -> store
```

It should:
- Run correctly through scalar fallback.
- Handle edge tiles and K tails.
- Expose tile IR, scalar fallback IR, and lowered tinygrad UOps for every stage.
- Keep tinygrad responsible for backend codegen/runtime.
- Make no performance promises yet.

## Current Baseline (pre-MVP)

Already done:

- Core `KernelBuilder`, IR, validation, and tinygrad UOp lowerer.
- `grid`, `threads`, `range`, `reduce`, `unroll`, shared/register allocs, barriers.
- `TileView` and `TileCopy` with scalar fallback.
- Debug artifacts and named IR stages.
- tinygrad compatibility adapter.
- Existing scalar canonical GEMM (with raw buffer copies, not TileView/TileCopy).

Not yet done (MVP scope below):

- `TileMMA` IR node and scalar fallback for tile-level GEMM intent.
- Canonical GEMM expressed as `TileView -> TileCopy -> TileMMA` pipeline.
- End-to-end debug inspection showing tile intent through to lowered UOps.
- Edge-tile and K-tail correctness coverage for the tile-op GEMM.

## Phase 1: Stabilize MVP Surface

Goal: make the intended user model clear and stable.

Work:
- Decide the public names for tile-level GEMM (`k.gemm(...)` or `k.mma(...)`).
- Keep `k.copy(...)` as the single data movement primitive.
- Keep `k.pipelined(...)` syntax-only for now.
- Avoid adding copy policy/vectorization APIs until after MVP.
- Update examples to show the intended MVP style.
- Update README to clearly say MVP prioritizes correctness, not speed.

Success criteria:
- One "blessed" GEMM example shows the final API shape.
- README clearly states priorities.
- Existing scalar/builder examples still work.

## Phase 2: Add First-Class TileMMA

Goal: make GEMM intent explicit in TileGrad IR.

Work:
- Add a `TileMMA` IR node as the tile-level GEMM contract.
- Validate A/B/C shapes for tile GEMM compatibility.
- Validate dtype compatibility.
- Validate supported scopes (shared A/B, register C).
- Keep unsupported cases rejected clearly or routed to scalar fallback.
- Preserve `TileMMA` in unexpanded tile IR.
- Expand `TileMMA` to scalar operations before tinygrad lowering.

Success criteria:
- `k.build()` preserves `TileMMA` in the unexpanded kernel.
- `inspect_kernel(k).tile_ir` shows `TileMMA`.
- `inspect_kernel(k).scalar_ir` contains only scalar fallback ops.
- Correctness tests pass for small GEMM shapes.

## Phase 3: Canonical MVP GEMM

Goal: make one complete GEMM path feel finished.

Work:
- Rewrite or add canonical GEMM around `TileView`, `TileCopy`, `TileMMA`, barriers, and guarded stores.
- Cover M/N edge tiles (partial tiles at matrix boundaries).
- Cover K tails (partial tiles along the reduction dimension).
- Use explicit `grid(...)` and `threads(...)`.
- Keep scalar fallback as the only required lowering path.

Success criteria:
- A canonical GEMM example runs end-to-end.
- Tests compare against a Python/tinygrad reference.
- Edge-shape tests pass.
- Debug inspection works on the canonical GEMM.

## Phase 4: MVP Debug and Docs

Goal: make the MVP usable and explainable.

Work:
- Document the MVP lowering path:

```text
TileGrad API -> Tile IR -> scalar fallback IR -> tinygrad UOps -> tinygrad runtime
```

- Document non-goals: no vectorized copies, no WMMA, no real pipeline yet.
- Add one inspect example for `TileCopy`.
- Add one inspect example for `TileMMA`/canonical GEMM.
- Keep tinygrad `DEBUG`/`VIZ` as the backend inspection path.

Success criteria:
- New user can run copy, tiled GEMM, and inspect examples.
- README and ROADMAP agree on priorities.
- Debug output makes tile-level intent visible.

## Phase 5: MVP Hardening

Goal: make the slow path reliable enough to optimize later.

Work:
- Add focused validation errors for bad `TileMMA` shapes/dtypes/scopes.
- Add regression tests for nested ranges around tile ops.
- Add runtime tests for copy + MMA + guarded store combinations.
- Keep compatibility adapter small and only for tinygrad APIs that have churned.

Success criteria:
- Full `tests/` passes.
- MVP GEMM works across representative small and edge shapes.
- No optimization-specific code is required for correctness.

## Design Principles

- Keep TileGrad thin: `KernelBuilder / TileView -> TileGrad IR -> tinygrad UOps`.
- Preserve the schedule the user wrote.
- Use tinygrad as backend/codegen/runtime, not as the high-level scheduler for explicitly scheduled kernels.
- Prefer one good abstraction over many speculative abstractions.
- Keep scalar fallback paths before optimized lowering.
- Make every layer inspectable.
- Let users escape to lower-level IR/UOps when needed.
- MVP first: build the complete slow version before optimizing anything.

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

## Non-Goals For Now

- Do not replace tinygrad backend codegen.
- Do not infer tiled schedules from ordinary tinygrad Tensor programs.
- Do not build a decorator or Python AST frontend before the tile IR is stable.
- Do not build an optimizer before there is one fast kernel to optimize.
- Do not add many tile subclasses until one `TileView` abstraction proves insufficient.
- Do not rely on tinygrad heuristic scheduling for explicitly scheduled TileGrad kernels.
- Do not add copy policy/classification/vectorized copies before MVP.
- Do not add `Ops.WMMA` lowering before MVP.
- Do not add real pipelining before MVP.
- Do not add autotuning before MVP.

## Post-MVP Optimization Track

Only after the MVP above is stable:

1. Add `TileCopy` classification (scalar, contiguous, coalesced-candidate, vectorizable-candidate).
2. Add vectorized lowering for unmasked contiguous copies.
3. Add copy microbenchmarks.
4. Add `Ops.WMMA` lowering for one CUDA fp16 case.
5. Compare scalar TileGrad GEMM, WMMA TileGrad GEMM, and tinygrad matmul.
6. Add real `pipelined(...)` semantics and double buffering.
7. Add autotuning for `BM`, `BN`, `BK`, thread layout, pipeline stages, and WMMA variants.

## Priority Order

1. Stabilize MVP API shape.
2. Add first-class `TileMMA` with scalar fallback.
3. Build canonical MVP GEMM around tile ops.
4. Harden debug/docs/tests around the MVP.
5. Then optimize memory movement.
6. Then target `Ops.WMMA`.
7. Then pipeline and autotune.
