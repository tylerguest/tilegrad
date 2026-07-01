<div align="center">

<img alt="tilegrad" src="tilegrad.png" width="50%">

</div>

# tilegrad

`tilegrad` is a small TileLang-inspired kernel frontend that lowers to tinygrad UOps.

The goal is to make tiled GPU kernels easier to write than raw tinygrad custom UOps while keeping the compiler path tiny, inspectable, and hackable.

```text
KernelBuilder -> tilegrad IR -> tinygrad UOps -> tinygrad runtime/codegen
```

`tilegrad` is early. It currently focuses on proving a small set of tiled-kernel semantics:

- explicit IR and validation
- shared memory allocations
- register accumulators
- tiled copies
- barriers
- reduce loops
- small GEMM kernels

## Why

`tilegrad` is trying to be to tinygrad what TileLang is to TVM.

TileLang gives TVM users a productive way to write tiled GPU kernels without manually working at the lowest compiler-IR level. `tilegrad` aims to do the same for tinygrad: provide a small tiled-kernel frontend while still lowering into tinygrad's UOps, runtime, codegen, and debugging tools.

The goal is not to replace tinygrad. The goal is to make custom tiled kernels easier to author while preserving tinygrad's inspectable compiler path.

## Setup

Clone tilegrad and install it with a local tinygrad checkout:

```bash
git clone https://github.com/tylerguest/tilegrad.git
cd tilegrad

python3 -m venv .venv
source .venv/bin/activate

git clone https://github.com/tinygrad/tinygrad ../tinygrad
pip install -e ../tinygrad
pip install -e ".[dev]"
```

Run tests:

```bash
python3 -m pytest tests/
```

## Quick Examples

Run a simple IR copy:

```bash
python3 examples/ir_copy.py
```

Run a shared-memory copy:

```bash
python3 examples/ir_shared_copy.py
```

Run a builder-authored tiled GEMM:

```bash
python3 examples/builder_tiled_gemm.py
```

Expected output:

```text
[413.0, 434.0, 1061.0, 1118.0]
```

## Example Kernel

A small tiled GEMM in tilegrad uses shared tiles, a register accumulator, tiled copies, a barrier, and a reduce loop:

```python
k = KernelBuilder("builder_tiled_gemm", ("out", "a", "b"))
k.alloc("as", 3, "float32")
k.alloc("bs", 3, "float32")
k.alloc("acc", 1, "float32", "register")

with k.range("i", 2):
  with k.range("j", 2):
    k.set("acc", 0, 0)
    with k.range("ko", 2):
      k.copy("a", "as", shape=(1, 3), stride=6, src_row_off="i", src_col_off=Mul("ko", 3))
      k.copy("b", "bs", shape=(3,), stride=2, src_row_off=Mul("ko", 3), src_col_off="j")
      k.barrier()
      with k.range("kk", 3, axis="reduce"):
        k.set("acc", 0, Add(k.load("acc", 0), Mul(k.load("as", "kk"), k.load("bs", "kk"))))
    k.set("out", Index2D("i", "j", 2), k.load("acc", 0))
```

See `examples/builder_tiled_gemm.py` for the runnable version.

## Debugging

Because tilegrad lowers to tinygrad UOps, tinygrad debugging tools work normally:

```bash
DEBUG=6 python3 examples/builder_tiled_gemm.py
VIZ=1 python3 examples/ir_shared_copy.py
```

## Status

This is experimental. The current priority is correctness and inspectability over performance.
