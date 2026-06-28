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
    ir.py
    lowerer.py
  examples/
    raw_uop_zero.py
    ir_zero.py
    ir_copy.py
    raw_shared_copy.py
    ir_shared_copy.py
  tests/
    test_ir.py
    test_lowerer.py
  README.md
  pyproject.toml
```

## Completed

- Raw tinygrad UOp zero kernel
- Minimal tilegrad IR
- IR zero kernel
- IR copy kernel
- Raw shared-memory copy kernel
- IR shared-memory copy kernel
- Basic unittest-style tests
- Editable package setup

## Next Milestones

1. Add expression trees for elementwise math.
2. Add a small builder API over the IR.
3. Add index expressions and 2D indexing.
4. Add typed allocations and multiple shared allocation slots.
5. Add tile load/store helpers.
6. Add scalar tiled matmul.
7. Add WMMA/tensor-core lowering for one known backend/shape.

## Guiding Rule

Do not jump straight to matmul. Keep proving one lowering concept at a time:

```text
UOp primitive -> tilegrad IR node -> lowerer rule -> example -> test -> VIZ
```
