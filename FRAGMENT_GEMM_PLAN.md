# Fragment GEMM Lowering Plan

## Goal

Implement a TileLang-style accumulator fragment path for tiled GEMM instead of expressing BMxBN accumulators as ordinary dynamically indexed register arrays.

The target user API should look like:

```python
acc = k.fragment("acc", shape=(BM, BN), dtype="float32")
k.clear(acc)

with k.range("ko", ...):
  ...
  k.gemm(as_tile, bs_tile, acc)

k.copy(acc, out_tile)
```

This avoids `acc[ii, jj]` dynamic register indexing and gives the lowerer a semantic GEMM primitive it can lower safely.

## Why This Is Needed

Tinygrad PTX currently requires register indices to be constant. The scalar IR pattern:

```python
acc[ii, jj] = acc[ii, jj] + as_tile[ii, kk] * bs_tile[kk, jj]
```

lowers to dynamic register indexing unless fully unrolled. Attempts to unroll enough of the BMxBN kernel create invalid or oversized tinygrad UOp graphs.

TileLang avoids this by modeling accumulator tiles as fragments:

```python
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
T.clear(C_local)
T.gemm(A_shared, B_shared, C_local)
T.copy(C_local, C[...])
```

TileLang fragments are not normal dynamic register arrays; they are lowered by tile-op-specific passes.

## Design Principle

Add a semantic layer above scalar `Set`/`Load` for tiled accumulators.

Do not try to make arbitrary `register[dynamic_index]` work. Instead:

- Register scalar access remains supported.
- Small static register tile unroll remains supported.
- BMxBN GEMM uses fragment/tile-op IR.

## Phase 1: IR Additions

Add new IR nodes in `tilegrad/ir.py`:

```python
@dataclass(frozen=True)
class FragmentAlloc(KernelOp):
  name: str
  shape: tuple[int, int]
  dtype: str

@dataclass(frozen=True)
class FragmentClear(Stmt):
  buffer: str

@dataclass(frozen=True)
class FragmentGemm(Stmt):
  a: str
  b: str
  c: str
  a_shape: tuple[int, int]
  b_shape: tuple[int, int]
  c_shape: tuple[int, int]
  trans_a: bool = False
  trans_b: bool = False

@dataclass(frozen=True)
class FragmentStore(Stmt):
  src: str
  dst: str
  dst_row: object
  dst_col: object
  dst_stride: object
  guard: object | None = None
```

Notes:

- Keep fragment operations explicit.
- Do not make fragments addressable through normal `Load`/`Set` initially.
- `FragmentStore` can lower to scalar stores for now.

## Phase 2: Builder API

Add to `KernelBuilder`:

```python
def fragment(self, name, shape, dtype):
  self._body.append(FragmentAlloc(name, shape, dtype))
  return FragmentRef(self, name, shape)

def clear(self, fragment):
  self._current_body().append(FragmentClear(fragment.name))

def gemm(self, a, b, c, trans_a=False, trans_b=False):
  self._current_body().append(FragmentGemm(...))

def store_fragment(self, src, dst, dst_origin, guard=None):
  self._current_body().append(FragmentStore(...))
```

Example intended user code:

```python
acc = k.fragment("acc", (2, 2), "float32")

with k.range("bi", 2) as bi:
  with k.range("bj", 2) as bj:
    k.clear(acc)
    with k.range("ko", 2):
      ...
      k.gemm(as_tile, bs_tile, acc)
    k.store_fragment(acc, out, (bi * 2, bj * 2), guard_bounds=(3, 3))
```

## Phase 3: Validation

Update `validate.py`:

- Track fragment names separately from shared/register buffers.
- `FragmentAlloc` must be top-level.
- `FragmentClear`, `FragmentGemm`, and `FragmentStore` are allowed inside ranges.
- `FragmentGemm` constraints:
- `a` and `b` must be known shared or fragment buffers.
- `c` must be a fragment.
- Shape compatibility: `A[M,K] * B[K,N] -> C[M,N]`.
- `c.dtype` is accumulator dtype.
- Reject normal `Load`/`Set` against fragments for now.

## Phase 4: Lowering Strategy V1: Scalar Expansion

First implementation should be correctness-only.

Lower `FragmentAlloc("acc", (BM, BN), dtype)` as a register buffer of size `BM * BN`, but fragment operations must lower with constant indices.

Lower:

```python
FragmentClear(acc)
```

to:

```python
Set(acc, 0, 0)
Set(acc, 1, 0)
...
```

Lower:

```python
FragmentGemm(as, bs, acc)
```

to statically emitted scalar reductions:

```python
for ii in static range(BM):
  for jj in static range(BN):
    for kk in dynamic/reduce range(BK):
      acc_const = ii * BN + jj
      acc[acc_const] = acc[acc_const] + as[ii, kk] * bs[kk, jj]
```

Important:

- `ii` and `jj` must be compile-time Python loops in the lowerer, not IR `Range`s.
- Only `kk` remains an actual reduce `Range`.
- Register indices are always integer constants.
- This avoids dynamic PTX register indexing.

Lower:

```python
FragmentStore(acc, out, row, col, stride, guard)
```

to static scalar `StoreIf`s:

```python
for ii in static range(BM):
  for jj in static range(BN):
    store_if(
      guard(row + ii, col + jj),
      out,
      Index2D(row + ii, col + jj, stride),
      Load(acc, ii * BN + jj),
    )
```

## Phase 5: Avoid Tinygrad CFG Issues

The current lowerer hit tinygrad CFG assertions around repeated same-range `END`s.

For fragment V1:

- Emit static scalar ops inside the existing outer loop scope.
- Keep only reduce `Range`s for `kk`.
- Avoid unrolling parent loops like `bi`, `bj`, `ko`.
- Group independent final stores when they share the same active ranges.
- Do not chain independent global stores through `env[out].after(...)` unless required for same-index overwrites.

Regression tests must include:

- Multiple independent stores to different output indices.
- Repeated stores to the same output index still overwrite in order.
- Fragment GEMM output stores do not produce tinygrad CFG assertion.

## Phase 6: Tests

Add IR tests:

- `test_fragment_alloc_ir`
- `test_fragment_clear_ir`
- `test_fragment_gemm_ir`
- `test_fragment_store_ir`

Add validation tests:

- Fragment must be allocated top-level.
- `FragmentGemm` shape mismatch fails.
- `FragmentGemm` unknown buffer fails.
- Normal `Load(fragment, ...)` fails.
- Normal `Set(fragment, ...)` fails.

Add lowerer structural tests:

- Fragment clear emits constant register indices.
- Fragment GEMM emits constant register indices.
- Fragment store emits expected global stores.
- No register `INDEX` uses non-constant index.

Add runtime tests:

- `test_run_fragment_clear_store_2x2`
- `test_run_fragment_gemm_2x2x3`
- `test_run_tiled_gemm_bm_bn_accum_fragment`

The current skipped test should be rewritten to use fragments and then unskipped.

## Phase 7: Example

Replace `examples/builder_tiled_gemm_bm_bn_accum_tile.py` with the fragment version.

Expected behavior:

```bash
python3 examples/builder_tiled_gemm_bm_bn_accum_tile.py
```

prints:

```python
[360.0, 375.0, 390.0, 910.0, 950.0, 990.0, 1460.0, 1525.0, 1590.0]
```

## Phase 8: Later Optimization Path

Once scalar fragment correctness is stable:

1. Add `FragmentLayout` metadata.
2. Add vectorized fragment copy.
3. Add tinygrad WMMA/WGMMA lowering if available.
4. Add backend-specific fragment lowering:
   - CUDA WMMA/WGMMA
   - ROCm MFMA
   - CPU fallback scalar expansion
5. Keep scalar expansion as universal fallback.

## Non-Goals For First Pass

- Arbitrary dynamic indexing into fragments.
- Full TileLang layout inference.
- Tensor-core performance.
- Automatic conversion of existing scalar `acc[ii, jj]` patterns into fragments.
- General-purpose register array dynamic indexing.

## Implementation Order

1. Add IR nodes.
2. Add builder API.
3. Add validation.
4. Add a fragment expansion pass before validation/lowering.
5. Lower expanded scalar ops with constant register indices.
6. Add tests.
7. Rewrite BMxBN example and test to use fragments.
8. Remove skip from BMxBN runtime test.
9. Run full suite.

## Success Criteria

- Full test suite passes with no BMxBN skip.
- BMxBN fragment GEMM runtime test passes.
- BMxBN example runs successfully.
- No tinygrad PTX dynamic register-index failure.
- No tinygrad CFG assertion.
