# TileGrad Next Roadmap

TileGrad now supports the first meaningful GPU execution model on top of tinygrad UOps:

- `grid(...)` lowers to `AxisType.GLOBAL` and GPU block ids.
- `threads(...)` / `parallel(...)` lowers to `AxisType.LOCAL` and GPU thread ids.
- `range(..., axis="unroll")` lowers to `AxisType.UNROLL` and can produce vectorized codegen.
- Shared memory, barriers, guarded edge loads/stores, and register accumulation work in grid/thread GEMM examples.

The next work should focus on making this model ergonomic, reusable, and measurable rather than adding unrelated syntax.

## Phase 1: Stabilize The Axis Model

Goal: make grid/thread/unroll semantics official and easy to understand.

Tasks:

- Keep `README.md` current with `grid`, `blocks`, `threads`, `parallel`, and `unroll` semantics.
- Add quick examples for the current flagship kernels.
- Rename or clarify old examples that use `parallel`, since `parallel()` now means local thread axes.
- Consider renaming `_parallel_counter` to `_axes_counter` in `KernelBuilder`.
- Add validation for empty axis contexts, rejecting `k.grid()` and `k.threads()` with no extents.
- Format `AXIS_TYPES` in `lowerer.py` as a normal multiline mapping.
- Add a test that `blocks()` aliases `grid()`.

Execution plan:

1. Keep the work local to `tilegrad`.
   `tinygrad` already exposes the needed `AxisType.GLOBAL`, `AxisType.LOCAL`, and `AxisType.UNROLL` values, and its GPU dimension lowering already consumes global/local ranges.

2. Rename the internal builder counter.
   Change `KernelBuilder._parallel_counter` to `_axes_counter`, since the counter is shared by `grid()` and `threads()` and `parallel()` is now only a local-thread alias.

3. Reject empty axis contexts at the builder API boundary.
   Make `k.grid()`, `k.blocks()`, `k.threads()`, and `k.parallel()` with no extents raise a clear `ValueError`, instead of silently producing an empty tuple and no ranges.

4. Add focused builder tests.
   Cover `blocks()` aliasing `grid()`, empty `grid()` rejection, empty `threads()` rejection, and optionally empty `parallel()` rejection through the alias.

5. Keep IR validation focused on axis names.
   Existing validation should continue accepting `loop`, `reduce`, `global`, `local`, and `unroll`; empty axis-context rejection belongs in the builder tests, not IR validation.

6. Format `AXIS_TYPES` in `lowerer.py` as a normal multiline mapping.
   This is readability-only and should not change behavior.

7. Clarify old `parallel` examples.
   Prefer minimal documentation updates first: explain that `parallel()` is retained as an alias for `threads()` / local axes. Rename old files only if they are actively misleading in README or examples lists.

8. Refresh README axis documentation and quick examples.
   Keep the execution-axis table current, explicitly describe `blocks()` as a `grid()` alias, and make the flagship grid/thread/unroll examples easy to find.

9. Verify with targeted tests first.
   Run `python3 -m pytest tests/test_builder.py tests/test_validate.py tests/test_lowerer.py`, then run the full `tests/` suite if those pass.

## Phase 2: Upgrade `copy()`

Goal: move toward a small, inspectable TileGrad equivalent of TileLang's `T.copy`.

Current limitations:

- `copy()` supports only 1D and 2D.
- No destination offsets.
- No guard/fill support.
- No threaded copy policy.
- No shape inference from buffer refs.
- No async or pipeline support.

Target API direction:

```python
k.copy(
  src,
  dst,
  shape=(BM, BK),
  src_origin=(row, col),
  dst_origin=(0, 0),
  src_stride=K,
  dst_stride=BK,
  guard=(row < M) & (col < K),
  fill=0,
)
```

Initial correctness targets:

- 1D, 2D, and 3D copies.
- Guarded global-to-shared copies.
- Shared-to-global copies.
- Edge tile zero-fill.
- Destination offsets.
- Thread-local copy patterns using `threads(...)`.

Execution plan:

1. Keep `copy()` as builder-only syntax expansion.
   Do not add new IR nodes, lowerer paths, or tinygrad changes for the first upgrade. Expand `copy()` into existing `Range`, `Load`, `LoadIf`, `Store`, and `StoreIf` nodes so the result remains inspectable.

2. Support the new API while preserving current call sites.
   Add `src_origin`, `dst_origin`, `src_stride`, `dst_stride`, `guard`, and `fill` parameters. Keep the existing `stride`, `src_row_off`, and `src_col_off` parameters as compatibility shims until examples and tests are migrated.

3. Normalize legacy arguments internally.
   Treat old `stride` as `src_stride` when `src_stride` is not provided. Treat `src_row_off` and `src_col_off` as the default 2D `src_origin` when `src_origin` is not provided. Default `dst_origin` to zero offsets.

4. Add small private helpers in `builder.py`.
   Add helpers for offset addition, origin lookup, and copy-index construction so the main `copy()` implementation does not grow separate ad hoc branches for every rank.

5. Rewrite `copy()` around generic rank handling.
   Validate that `shape` is a non-empty tuple. Generate loop names like `_c{n}_i0`, `_c{n}_i1`, and `_c{n}_i2`, build nested `Range`s from the inside out, and support 1D, 2D, and 3D copies.

6. Implement destination offsets and strides.
   Compute destination indices from `dst_origin` and `dst_stride`, instead of always storing into a zero-origin compact tile. Default 2D `dst_stride` to the destination tile width or destination buffer shape when available.

7. Implement source offsets and strides.
   Compute source indices from `src_origin` and `src_stride`. Infer 2D source stride from `src.shape[1]` when `src` is a shaped `BufferRef`; otherwise require an explicit stride for 2D source indexing.

8. Add shape inference from buffer refs.
   If `shape` is omitted, infer it from `dst.shape` when `dst` is a shaped `BufferRef`. Use shaped `BufferRef`s to infer contiguous flattening for 3D copies where possible.

9. Add guarded copy support.
   Use `LoadIf(guard, src, src_idx)` for guarded loads when zero-fill behavior is requested. Use `StoreIf(guard, dst, dst_idx, Load(src, src_idx))` when the caller wants to skip out-of-bounds stores rather than fill.

10. Start with `fill=0` only.
    Implement edge-tile zero-fill as the first fill mode. Reject non-zero fill values clearly until conditional value selection is added.

11. Add builder IR tests.
    Cover 1D copy with `dst_origin`, 2D copy with `src_origin` and `dst_origin`, explicit `src_stride` and `dst_stride`, 3D contiguous copy IR, shape inference from destination `BufferRef`, guarded copy IR, and zero-fill guarded load IR.

12. Add runtime tests.
    Cover 1D destination-offset copy, 2D global-to-shared copy with source origin, 2D shared-to-global copy with destination origin, 2D guarded edge-tile zero-fill, 3D copy correctness, and one thread-local copy pattern using `threads(...)`.

13. Update examples.
    Convert `examples/builder_copy_3d.py` to use `k.copy(...)`, add a guarded 2D copy example using `guard` and `fill=0`, and add a grid/thread copy example only if the semantics remain simple and testable.

14. Document `copy()` in README.
    Add a short section showing the upgraded signature and explaining that `copy()` is synchronous, expands into normal TileGrad loops, supports 1D/2D/3D, supports edge guards and zero-fill, and does not yet implement async copies, coalescing policies, or pipelines.

15. Verify in layers.
    Run `python3 -m pytest tests/test_builder.py tests/test_runtime.py`, then `python3 -m pytest tests/`, then run the updated copy examples.

## Phase 3: Canonical Tiled GEMM

Goal: create one flagship GEMM that represents the intended TileGrad programming model.

Start with:

- `M=3`, `N=3`, `K=5`
- `BM=2`, `BN=2`, `BK=3`
- `grid(ceildiv(M, BM), ceildiv(N, BN))`
- `threads(BM, BN)`
- one output element per local lane
- shared A/B tiles
- K-tail guards
- M/N edge guards
- barrier between copy and compute
- guarded output store

Then generalize into a helper/factory:

```python
def tiled_gemm(M, N, K, BM=2, BN=2, BK=3):
  ...
  return k
```

Test cases:

- exact tile sizes
- M/N edge tiles
- K tail
- non-square M/N
- transposed B later

## Phase 4: Benchmarks

Goal: measure before optimizing.

Add small benchmark scripts comparing:

- TileGrad scalar loop GEMM
- TileGrad shared/register GEMM
- TileGrad grid/thread output-lane GEMM
- TileGrad fragment GEMM
- tinygrad `Tensor.matmul`
- optional raw tinygrad custom UOp baselines

Measure:

- correctness
- compile time
- runtime
- generated launch dimensions
- effective GFLOPS for larger shapes

Initial benchmark sizes:

- `64x64x64`
- `128x128x128`
- `256x256x256`

## Phase 5: Fragment Direction

Goal: decide how TileGrad fragments should evolve.

Current state:

- `FragmentGemm` expands to scalar register operations.
- This is useful for correctness and inspectability.
- It is not a tensor-core path.

Recommended path:

- Keep scalar fragment expansion as the default.
- Add tests for fragment GEMM under `grid` and `threads`.
- Add a shape/dtype gate for intrinsic lowering later.
- Prototype one tinygrad `Ops.WMMA` lowering for a single supported shape and dtype.

Do not start with a general WMMA implementation. Start with one backend-supported case.

Execution plan:

1. Treat scalar fragment expansion as the default path.
   The Phase 4 benchmark shows that `FragmentGemm` is correctness-oriented and inspectable, not performance-oriented. Keep this behavior as the fallback for every unsupported shape, dtype, or backend.

2. Keep the grid/thread fragment coverage.
   Maintain tests and examples that run fragment GEMM under `grid(...)` and `threads(...)`, even if the local thread axis is minimal. This protects launch-axis compatibility before intrinsic lowering exists.

3. Replace the placeholder intrinsic gate with a real policy function.
   Evolve `can_lower_fragment_gemm_intrinsic(...)` from a hardcoded `False` into an explicit shape/dtype/backend gate. The function should be small, easy to audit, and conservative.

4. Choose exactly one prototype target.
   Prefer one CUDA-supported shape and dtype combination, such as a single `16x16x16` style WMMA case with `float16` inputs and `float32` accumulation, if that matches tinygrad's current `Ops.WMMA` support. If tinygrad supports a different canonical shape, match tinygrad rather than inventing a TileGrad-specific one.

5. Inspect tinygrad's WMMA support before lowering anything.
   Answer these questions in code comments or tests before implementing lowering:
   - What UOp signature does `Ops.WMMA` expect?
   - What shapes and dtypes are supported?
   - Which renderers implement it?
   - Is CUDA supported in the local tinygrad checkout?

6. Add gate-only tests first.
   Test that unsupported dtypes return `False`, unsupported shapes return `False`, the chosen supported shape returns `True`, and ordinary fragment GEMM still expands through the scalar fallback.

7. Prototype one intrinsic lowering path only after the gate is tested.
   Keep the prototype behind the gate. If the gate rejects the case, fragment GEMM must continue using scalar expansion.

8. Do not implement a general WMMA system yet.
   Avoid broad layout handling, transpose support, backend abstraction, mixed architecture policies, or auto-selection until one backend-supported case works and is tested.

## Phase 6: Shape And Symbolic Dimensions

Goal: avoid hardcoding every kernel shape.

Possible API direction:

```python
M = k.dim("M")
N = k.dim("N")
K = k.dim("K")
```

or:

```python
k.buffer("a", shape=("M", "K"))
```

First practical step:

- Support shape-derived dimensions such as `"a.shape.0"` and `"a.shape.1"`.
- Add a small `ceildiv` expression helper.
- Use these for grid dimensions in examples.

## Phase 7: Pipeline Later

Goal: eventually approach TileLang's `T.Pipelined` model.

Do not implement real pipelining yet. Prerequisites:

- upgraded `copy()`
- canonical GEMM
- benchmark harness
- stable shared/barrier dependency behavior

Future API direction:

```python
with k.pipelined("ko", KTILES, stages=2) as ko:
  ...
```

Initial implementation can be syntax-only or validation-only. Async copy, wait groups, double buffering, and stage scheduling should come later.

## Priority Order

1. Stabilize and document the axis model.
2. Upgrade `copy()` with guards, offsets, and threaded patterns.
3. Build one canonical grid/thread tiled GEMM.
4. Add benchmarks.
5. Extend fragment GEMM coverage under grid/thread axes.
6. Prototype one WMMA lowering path.
7. Add shape/dim ergonomics.
8. Explore pipeline syntax and double buffering.

## Immediate Recommendation

Focus next on `copy()`.

The execution-axis model is working. The biggest friction in every tiled kernel is now hand-written shared-memory movement. A small, correct, inspectable `copy()` upgrade will make GEMM and future attention-style kernels much easier to write and benchmark.
