# tilegrad

`tilegrad` is a small experiment in building a TileLang-style frontend that lowers to tinygrad UOps.

This is early and intentionally minimal. Right now it supports simple IR nodes for ranges, loads, stores, shared allocation, and barriers.

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install tinygrad from a local checkout:

```bash
git clone https://github.com/tinygrad/tinygrad ../tinygrad
pip install -e ../tinygrad
```

Install tilegrad:

```bash
pip install -e ".[dev]"
```

## Run Examples

```bash
python examples/raw_uop_zero.py
python examples/ir_zero.py
python examples/ir_copy.py
python examples/raw_shared_copy.py
python examples/ir_shared_copy.py
```

## Run Tests

```bash
python -m pytest tests
```

## tinygrad VIZ

Because tilegrad lowers to normal tinygrad UOps, tinygrad VIZ can inspect generated graphs:

```bash
VIZ=1 python examples/ir_shared_copy.py
```

## Current Status

Implemented:

- flat ranges
- loads and stores
- shared allocation
- barriers
- tinygrad UOp lowering

Not implemented yet:

- builder API
- expression trees
- 2D indexing
- tile load/store
- dot/matmul
- WMMA lowering

See `outline.md` for the roadmap.
