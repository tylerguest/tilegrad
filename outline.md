# tinytile Roadmap

## Goal

Build a TileLang-inspired frontend that lowers to tinygrad UOps while keeping tinygrad internals behind one backend boundary.

The intended path is:

```text
tinytile DSL -> tinytile IR -> tinygrad UOps -> tinygrad runtime/codegen -> VIZ
```

## Architecture

Use four layers:

1. Frontend API
2. Tinytile IR
3. tinygrad lowerer
4. tinygrad runtime path

Keep this boundary strict:

```text
frontend.py        no tinygrad imports
ir.py              no tinygrad imports
lower_tinygrad.py  imports tinygrad internals
runtime.py         tinygrad Tensor/custom_kernel integration
```

## Suggested Layout

```text
tinytile/
  tinytile/
    __init__.py
    ir.py
    frontend.py
    lower_tinygrad.py
    runtime.py
    dtypes.py
  examples/
    00_raw_uop_zero.py
    01_ir_zero.py
    02_builder_zero.py
    03_tiled_matmul_ir.py
  tests/
    test_ir.py
    test_lower_tinygrad.py
  README.md
  pyproject.toml
```

## Phase 1: Raw tinygrad UOp Kernel

Goal: understand tinygrad custom kernels before adding abstractions.

Write a handwritten tinygrad kernel directly, no tinytile IR yet.

Example target:

```python
def zero_kernel(out):
    i = UOp.range(out.size, 0, AxisType.LOOP)
    return out.flatten().index(i, ptr=True).store(0).end(i).sink(
        arg=KernelInfo(name="zero_kernel")
    )
```

Run through `Tensor.custom_kernel`, then use:

```bash
VIZ=1 PYTHONPATH=/home/ubulap/repos/tinygrad:/home/ubulap/repos/tinytile python3 examples/00_raw_uop_zero.py
```

Success criteria:

- tinygrad runs the custom kernel.
- VIZ shows the UOp graph.
- You understand `UOp.range`, `.index`, `.store`, `.end`, and `.sink`.

## Phase 2: Minimal Tinytile IR

Goal: create a tiny IR for simple scalar kernels.

Start with dataclasses for:

- `Kernel`
- `Arg`
- `Range`
- `Store`
- `Const`
- `Index`

Target shape:

```python
Kernel(
    name="zero",
    args=[Arg("out", shape, dtype)],
    body=[
        Range("i", 0, N),
        Store("out", index="i", value=0),
    ],
)
```

Lower this to the same UOps from Phase 1.

Success criteria:

- Your IR produces a tinygrad `SINK`.
- The generated tinygrad kernel runs.
- VIZ still works.

## Phase 3: Builder API

Goal: make IR easier to write by hand.

Example target:

```python
T = KernelBuilder("zero")
out = T.arg("out", (N,), "float32")

with T.range("i", 0, N) as i:
    T.store(out, i, 0)

kernel_ir = T.finish()
```

This should still only create tinytile IR, not tinygrad UOps.

Success criteria:

- Kernels are pleasant to type.
- You can print/debug tinytile IR before lowering.
- The lowerer remains separate.

## Phase 4: Tensor Integration

Goal: execute through tinygrad normally.

Create an API like:

```python
tinytile.call(out, kernel_ir)
```

or:

```python
out = tinytile.zero(out)
```

Internally, use `Tensor.custom_kernel` so tinygrad owns scheduling, codegen, profiling, and VIZ.

Success criteria:

- User-facing tinytile functions return tinygrad `Tensor`s.
- `.realize()` triggers tinygrad compilation.
- `VIZ=1` shows the generated UOp kernel.

## Phase 5: Shared Memory

Goal: add `T.shared`, `T.local`, and `T.barrier`.

Add IR nodes:

- `SharedAlloc`
- `LocalAlloc`
- `Barrier`
- `Load`
- `Store`

Lowering targets:

```python
T.shared((N,), "float32")
```

to:

```python
UOp.placeholder((N,), dtypes.float, slot, AddrSpace.LOCAL)
```

and:

```python
T.local(...)
```

to:

```python
UOp.placeholder(..., AddrSpace.REG)
```

Barrier lowering must preserve dependencies:

```python
st = shared.index(..., ptr=True).store(...)
bar = st.barrier()
shared_ready = shared.after(bar)
```

Success criteria:

- You can write global-to-shared-to-global copy.
- Generated UOps include `BUFFER`, `STORE`, `BARRIER`, and `LOAD`.
- VIZ shows the barrier dependency.

## Phase 6: Tile Helpers

Goal: move from scalar operations to tile operations.

Add IR nodes:

- `LoadTile`
- `StoreTile`

Frontend target:

```python
T.load_tile(As, A, mapping="coalesced")
T.store_tile(C, acc)
```

Lowering should decide:

- Which thread loads which elements.
- Whether to vectorize.
- What predicates are needed for boundary handling.
- Whether to insert a barrier.

Start with perfectly divisible shapes. Add boundary masks later.

## Phase 7: Scalar Dot

Goal: implement tiled matmul without tensor cores.

Frontend target:

```python
T.dot(acc, As, Bs, mode="scalar")
```

Lower to explicit reduction loops:

```python
for k in range(BK):
    acc[m, n] += As[m, k] * Bs[k, n]
```

Use `AddrSpace.REG` for accumulators.

Success criteria:

- A small tiled matmul works.
- It does not need to be fast yet.
- VIZ is readable enough to inspect the graph.

## Phase 8: WMMA Dot

Goal: lower specific dot shapes to tinygrad `Ops.WMMA` or `Ops.SHAPED_WMMA`.

Start with one backend and one shape.

Reference files in tinygrad:

- `extra/gemm/amd_copy_matmul.py`
- `extra/gemm/amd_flash_attention.py`
- `extra/thunder/tiny/tk/group.py`

Frontend target:

```python
T.dot(acc, As, Bs, mode="wmma")
```

Lowering should:

- Validate tile shape.
- Validate dtype.
- Build fragments.
- Emit `Ops.SHAPED_WMMA` or `Ops.WMMA`.
- Store results back into accumulator registers.

Success criteria:

- One known matmul shape lowers to tensor-core UOps.
- VIZ shows `WMMA` or `SHAPED_WMMA`.

## Best First Files

Write files in this order:

1. `examples/00_raw_uop_zero.py`
2. `tinytile/ir.py`
3. `tinytile/lower_tinygrad.py`
4. `examples/01_ir_zero.py`
5. `tinytile/frontend.py`
6. `examples/02_builder_zero.py`

## Guiding Rule

Do not start with matmul. First prove the smallest path:

```text
raw UOp zero kernel -> IR zero kernel -> builder zero kernel -> tinygrad VIZ
```
