# tilegrad Roadmap

## Goal

Build a TileLang-inspired frontend that lowers to tinygrad UOps while keeping tinygrad internals behind one backend boundary.

The intended path is:

```text
tilegrad DSL -> tilegrad IR -> tinygrad UOps -> tinygrad runtime/codegen -> VIZ
```

## Architecture

Use four layers:

1. Frontend API
2. tilegrad IR
3. tinygrad lowerer
4. tinygrad runtime path

Keep this boundary strict:

```text
frontend.py   no tinygrad imports
ir.py         no tinygrad imports
lowerer.py    imports tinygrad internals
runtime.py    tinygrad Tensor/custom_kernel integration
```

## Current Layout

```text
tilegrad/
  tilegrad/
    builder.py
    ir.py
    validate.py
    lowerer.py
  examples/
    builder_copy.py
    builder_shared_copy.py
    builder_transpose_2d.py
    ir_zero.py
    ir_copy.py
    ir_shared_copy.py
    ir_transpose_2d.py
    ir_sequential_overwrite.py
    ir_two_shared_allocs.py
    raw_uop_zero.py
    raw_shared_copy.py
  tests/
    test_builder.py
    test_ir.py
    test_validate.py
    test_lowerer.py
  README.md
  outline.md
  pyproject.toml
```

## Completed

- Raw tinygrad UOp zero kernel
- Minimal tilegrad IR
- IR zero kernel
- IR copy kernel
- Raw shared-memory copy kernel
- IR shared-memory copy kernel
- Expression trees for basic integer/elementwise math
- Flattened index expressions for 2D-style indexing
- Sequential effect ordering tests
- Multiple shared allocation slots
- IR validation layer with no tinygrad dependency
- Small builder API over the IR
- Builder examples for copy, shared copy, and transpose
- Basic unittest-style tests for IR, validation, lowering, and builder behavior
- Editable package setup

## Next Milestones

1. Refresh the public API surface.
   Export the builder and IR helpers from `tilegrad.__init__` once the names feel stable enough for examples to import from `tilegrad` directly.
2. Add index helper functions.
   Keep 2D indexing as flattened expressions for now, but add small helpers such as `idx2(row, col, stride)` to avoid hand-writing `Add(Mul(...), ...)` everywhere.
3. Add predicate expressions and guarded effects.
   Implement the smallest useful set first: subtraction, comparisons, and guarded stores or a simple `If` statement. This is needed for edge tiles and safe partial copies.
4. Add reduce ranges.
   Extend `Range` with a loop kind or add a separate reduce range that lowers to `AxisType.REDUCE`. Prove it with scalar sum before matmul.
5. Add register allocation support.
   Introduce a minimal register scalar or fragment allocation backed by tinygrad `AddrSpace.REG`. Use it for reductions before exposing larger fragment semantics.
6. Add a naive scalar GEMM.
   Express `C[i, j] = sum_k A[i, k] * B[k, j]` using loop ranges, reduce ranges, and register accumulation. This is the correctness bridge before shared-memory tiled matmul.
7. Add tile load/store helpers.
   Start with synchronous copy helpers that expand to ordinary ranges, loads, stores, and barriers. Do not add async copy, coalescing, or layout inference yet.
8. Add a thin runtime helper.
   Wrap tinygrad `Tensor.custom_kernel` so examples can run a TileGrad kernel without manually defining tinygrad callback functions every time.
9. Add shared-memory tiled matmul for one fixed shape.
   Keep it scalar and correctness-focused. Use explicit shared allocations, synchronous copies, barriers, and scalar accumulation.
10. Add WMMA/tensor-core lowering for one known backend/shape.
    Treat this as a separate backend experiment after the scalar tiled path is correct and inspectable in tinygrad VIZ.

## Guiding Rule

Do not jump straight to matmul. Keep proving one lowering concept at a time:

```text
UOp primitive -> tilegrad IR node -> lowerer rule -> example -> test -> VIZ
```
