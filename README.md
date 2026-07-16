<div align="center">

<img alt="tilegrad" src="tilegrad.png" width="50%">

</div>

# tilegrad

`tilegrad` is a small TileLang-inspired kernel frontend that lowers to tinygrad UOps.

The goal is to make custom tiled GPU kernels easier to write than raw tinygrad custom UOps while keeping the compiler path small, inspectable, and hackable.

```text
KernelBuilder -> tilegrad IR -> tinygrad UOps -> tinygrad runtime/codegen
```

`tilegrad` is early and experimental. Current priorities are correctness, readability, and learning the right programming model before optimizing.

## Features

- Explicit kernel builder API
- Validation over a small TileGrad IR
- GPU grid axes with `grid(...)` / `blocks(...)`
- GPU local thread axes with `threads(...)`
- `parallel(...)` as a compatibility alias for `threads(...)`
- Serial, reduce, and unroll ranges
- Shared memory allocations
- Register accumulators
- Barriers
- Guarded loads and stores
- Tiled `copy(...)`
- Fragment GEMM scalar expansion
- Canonical grid/thread tiled GEMM
- Simple benchmark harness

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

Run a simple copy:

```bash
python3 examples/builder_copy.py
```

Run a shape-derived copy:

```bash
python3 examples/builder_shape_dims_copy.py
```

Run a grid/thread copy:

```bash
python3 examples/builder_grid_threads_copy.py
```

Run the canonical tiled GEMM:

```bash
python3 examples/builder_canonical_tiled_gemm.py
```

Expected output:

```text
[360.0, 375.0, 390.0, 910.0, 950.0, 990.0, 1460.0, 1525.0, 1590.0]
```

Run fragment GEMM under grid/thread axes:

```bash
python3 examples/builder_grid_thread_fragment_gemm.py
```

Run syntax-only pipelined copy:

```bash
python3 examples/builder_pipelined_copy.py
```

Run GEMM benchmarks:

```bash
python3 benchmarks/bench_gemm.py
```

## Execution Axes

TileGrad ranges map directly to tinygrad axis types.

| TileGrad API | IR axis | tinygrad axis | Purpose |
| --- | --- | --- | --- |
| `k.range(...)` | `"loop"` | `AxisType.LOOP` | serial/logical loop |
| `k.range(..., axis="reduce")` | `"reduce"` | `AxisType.REDUCE` | recurrence/reduction loop |
| `k.grid(...)` / `k.blocks(...)` | `"global"` | `AxisType.GLOBAL` | GPU block/grid ids |
| `k.threads(...)` | `"local"` | `AxisType.LOCAL` | GPU local/thread ids |
| `k.parallel(...)` | `"local"` | `AxisType.LOCAL` | alias for `k.threads(...)` |
| `k.range(..., axis="unroll")` | `"unroll"` | `AxisType.UNROLL` | unrolled/vectorized loop |

Prefer `threads(...)` in new code. `parallel(...)` is kept as an alias, but it means local GPU thread axes, not generic parallel iteration.

Example:

```python
from tilegrad import KernelBuilder

k = KernelBuilder("grid_threads_copy", ("out", "inp"))
out = k.buffer("out")
inp = k.buffer("inp")

with k.grid(2) as block:
  with k.threads(4) as tid:
    i = block * 4 + tid
    out[i] = inp[i]
```

With tinygrad debugging enabled, this lowers to GPU block and thread indices such as `%ctaid.x` and `%tid.x` on CUDA.

## Copy

`copy(...)` is a synchronous helper that expands into normal TileGrad loops.

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

Current support:

- 1D, 2D, and compact 3D copies
- source and destination offsets
- source and destination strides
- shape inference from shaped `BufferRef`s
- guarded edge loads
- zero-fill with `fill=0`

Not supported yet:

- async copy
- wait groups
- software pipelines
- copy coalescing policies
- non-zero fill values

## Canonical GEMM

The main GEMM path is `tiled_gemm(...)`:

```python
from tilegrad import run
from tilegrad.kernels import tiled_gemm

M = 3
N = 3
K = 5
BM = 2
BN = 2
BK = 3

k = tiled_gemm(M, N, K, BM, BN, BK)
```

It uses:

- `grid(ceildiv(M, BM), ceildiv(N, BN))`
- `threads(BM, BN)`
- one output element per local lane
- shared A/B tiles
- K-tail guards
- M/N edge guards
- guarded output stores

See:

```bash
python3 examples/builder_canonical_tiled_gemm.py
```

## Fragments

TileGrad has a `FragmentGemm` path, but it currently expands to scalar register operations. It is useful for correctness and inspectability, not performance.

See:

```bash
python3 examples/builder_grid_thread_fragment_gemm.py
```

The benchmark currently shows fragment GEMM is much slower than the canonical grid/thread GEMM. This is expected until a real intrinsic path, such as a single supported `Ops.WMMA` lowering, is prototyped.

## Shape Helpers

TileGrad supports simple shape-derived dimensions:

```python
with k.range("i", "inp.shape.0") as i:
  out[i] = inp[i]
```

It also has a small integer helper:

```python
from tilegrad import ceildiv, ceildiv_expr
```

`ceildiv(...)` is for Python integers. `ceildiv_expr(...)` builds a TileGrad expression for static integer divisors.

## Pipelined Syntax

`pipelined(...)` exists as syntax only:

```python
with k.pipelined("ko", KTILES, stages=2) as ko:
  ...
```

Today it behaves exactly like:

```python
with k.range("ko", KTILES) as ko:
  ...
```

It validates `stages`, but does not implement async copy, wait groups, double buffering, or real software pipeline scheduling yet.

## Benchmarks

Run:

```bash
python3 benchmarks/bench_gemm.py
```

The benchmark compares:

- tinygrad `Tensor.matmul`
- TileGrad canonical tiled GEMM
- scalar fragment GEMM baseline
- grid/thread fragment GEMM baseline

The fragment baselines are intentionally limited to small sizes because fragment GEMM currently expands to scalar operations.

## Debugging

TileGrad exposes the IR stages it owns:

```python
from tilegrad import KernelBuilder
from tilegrad.debug import inspect_kernel

k = KernelBuilder("copy", ("out", "inp"))
out = k.buffer("out", shape=(4,), dtype="float32")
inp = k.buffer("inp", shape=(4,), dtype="float32")
k.copy(inp.tile(), out.tile())

dbg = inspect_kernel(k)

print(dbg.tile_ir)
print(dbg.scalar_ir)
print([stage.name for stage in dbg.stages])
```

Because TileGrad lowers to tinygrad UOps, tinygrad debugging tools remain the recommended path for backend codegen and runtime inspection:

```bash
DEBUG=6 python3 examples/builder_canonical_tiled_gemm.py
DEBUG=6 python3 examples/builder_grid_threads_copy.py
VIZ=1 python3 examples/ir_shared_copy.py
```

