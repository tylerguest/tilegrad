# tinygrad Integration Contract

TileGrad delegates scalar semantics, legalization, code generation, compilation,
caching, and execution to tinygrad. This document describes the live tinygrad
API boundary used by TileGrad.

TileGrad intentionally does not pin this contract to a tinygrad version or Git
revision. Compatibility is checked from the APIs TileGrad uses and by running
the integration tests against the active tinygrad checkout.

## Required APIs

Production access to tinygrad internals is isolated in
`tilegrad/tinygrad_compat.py`. TileGrad currently depends on:

- `Tensor.custom_kernel`
- caller-owned `TinyJit`
- `@function(precompile=True)`
- `UOp` construction and traversal
- `AxisType`
- `AddrSpace`
- `KernelInfo`
- tinygrad dtype definitions
- `tinygrad.uop.render.print_uops`

If a required API is absent, importing TileGrad raises
`TinygradCompatibilityError` naming the missing API.

## Kernel Boundary

`tilegrad.runtime.run` passes tensors to `Tensor.custom_kernel` in TileGrad
kernel argument order. tinygrad supplies placeholder UOps in the same order.
TileGrad returns one `Ops.SINK` and currently exposes the first output tensor.

Multiple outputs, aliases, workspaces, and gradient functions are not yet part
of this contract.

## UOps

TileGrad currently constructs these UOp categories:

- Parameters and storage: `PARAM`, `BUFFER`, and `CONST`.
- Indexing and memory: `INDEX`, `LOAD`, and `STORE`.
- Execution axes: `RANGE` and `END`.
- Dependencies: `AFTER`, `GROUP`, and `BARRIER`.
- Scalar arithmetic, comparisons, boolean operations, `WHERE`, and `CAST`.
- A `SINK` kernel root.

TileGrad does not construct renderer source or renderer-specific native matrix
arguments.

## Axes And Storage

| TileGrad | tinygrad |
| --- | --- |
| `loop` | `AxisType.LOOP` |
| `reduce` | `AxisType.REDUCE` |
| `global` | `AxisType.GLOBAL` |
| `local` | `AxisType.LOCAL` |
| `unroll` | `AxisType.UNROLL` |
| shared allocation | `AddrSpace.LOCAL` |
| register allocation | `AddrSpace.REG` |
| kernel argument | `AddrSpace.GLOBAL` |

Axis identities must remain visible in lowered UOps. Barriers must preserve the
dependency between shared-memory stores and subsequent loads.

## KernelInfo

TileGrad constructs kernel metadata with:

```python
KernelInfo(name=kernel_name, opts_to_apply=())
```

The active tinygrad checkout must accept and preserve the empty
`opts_to_apply` tuple. A complete rendered schedule-preservation contract is
still future work.

## Composition

TileGrad kernels must execute inside a caller-owned `TinyJit`. Compatibility
tests exercise initial execution, capture, and replay with different inputs of
the same shape and dtype.

TileGrad kernels must also compose inside `@function(precompile=True)`. The
precompiled function constructs its output with `Tensor.invalids` and calls
`run(..., realize=False)` so tinygrad owns realization of the resulting graph.

TileGrad does not create an internal `TinyJit`.

## Validation

`tests/test_tinygrad_compat.py` checks the required API surface, UOp axis and
storage metadata, barrier preservation, kernel metadata, and execution through
`Tensor.custom_kernel`, an enclosing `TinyJit`, and
`@function(precompile=True)`.

The contract does not yet certify GPU renderer compatibility, native matrix
selection, vector load/store preservation, or performance.
