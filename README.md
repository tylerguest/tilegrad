<div align="center">

<img alt="tilegrad" src="tilegrad.png" width="50%">

</div>

# tilegrad

`tilegrad` is a TileLang-inspired kernel frontend targeting tinygrad UOps.

The goal is to make tiled kernels easier to write than raw tinygrad UOps while keeping the compiler path small, inspectable, and hackable.

```text
tilegrad frontend -> tilegrad IR -> tinygrad UOps -> tinygrad runtime/codegen/VIZ
```

`tilegrad` is early. The project is currently focused on defining a small, correct IR for tiled kernels before growing a larger frontend or chasing performance.

## Why

TileLang has a productive programming model for tiled GPU kernels. tinygrad has a compact compiler/runtime stack with visible IR, codegen, debugging, and VIZ.

`tilegrad` explores the overlap:

- TileLang-style kernel programming
- tinygrad as the lowering/runtime backend
- a small IR with explicit validation
- a tiny codebase that is easy to read, test, and modify

The long-term direction is a small tile-kernel DSL that can express shared-memory tiled programs, matmul-style kernels, and eventually hardware-specific lowering paths while still going through tinygrad.

## Architecture

`tilegrad` keeps its own semantics separate from tinygrad internals.

```text
tilegrad.ir        pure IR nodes
tilegrad.validate  IR validation and semantic checks
tilegrad.lowerer   tinygrad UOp lowering
examples/          runnable kernels and raw tinygrad references
tests/             semantic and lowering regression tests
```

The important boundary is:

```text
IR and validation do not import tinygrad.
Only the lowerer depends on tinygrad internals.
```

This keeps tinygrad churn isolated to the backend boundary.

## Status

This is an experiment in compiler construction and tiled-kernel programming on top of tinygrad.

The near-term goal is boring correctness:

```text
define IR semantics -> validate them -> lower them -> test them
```

Once the IR is stable, the next step is making kernels pleasant to write.

## Direction

The project is intentionally not jumping straight to matmul.

The intended path is:

1. stabilize the IR and validation rules
2. add a small builder/frontend API
3. add tile-shaped load/store helpers
4. build tiled copy and transpose examples
5. build scalar tiled matmul
6. explore shared-memory matmul
7. investigate WMMA/tensor-core lowering

## Install

The recommended setup is from source with a local tinygrad checkout.

```bash
python3 -m venv .venv
source .venv/bin/activate

git clone https://github.com/tinygrad/tinygrad ../tinygrad
pip install -e ../tinygrad
pip install -e ".[dev]"
```

## Tests

```bash
python -m pytest tests
```

## Examples

```bash
python examples/ir_zero.py
python examples/ir_copy.py
python examples/ir_shared_copy.py
```

Raw tinygrad UOp references:

```bash
python examples/raw_uop_zero.py
python examples/raw_shared_copy.py
```

## tinygrad VIZ

Because tilegrad lowers to normal tinygrad UOps, tinygrad VIZ can inspect generated graphs:

```bash
VIZ=1 python examples/ir_shared_copy.py
```
