# tilegrad hardening pass — next steps

## Context

After skipping the unsupported 2D-register-accumulator tile GEMM test
(`test_run_tiled_gemm_bm_bn_accum_tile`), two hardening items remain
to make the lowerer and validator robust before pursuing perf work
(WMMA primitive, parallel-coverage, etc.).

## Root cause (confirmed by reproductions)

- **Lowerer bug (scalar repro, KeyError):** A 1-element register acc
  indexed with `acc[0]` (constant), wrapped in nested `ii`/`jj`/`ko`
  ranges with a `set_if((ii<1)&(jj<1), acc, 0, 0)` sibling to a reduce
  range, also produces the KeyError. Isolates the bug from the
  2D-reg-index issue.
- **Root cause** — `tilegrad/lowerer.py:147`:
  ```python
  next_buf = target.set(val, end=(recurrence_range, *register_end_ranges)) if axis == "reduce" else target.set(val)
  ```
  The `loop` branch passes no `end=`. When `register_end_ranges` is
  non-empty (the set is at a deeper nesting than the buffer's current
  scope), `buf.after(*register_end_ranges)` chains the new ranges
  but `target.set(val)` doesn't tell tinygrad's UOp graph those ranges
  own this set. Loop-exit cleanup at lines 171–181 checks
  `if i in buf.ranges` — which is empty → no `.end(i)` emitted →
  register buffer stays in inconsistent `AFTER(...)` state →
  CFG analysis in tinygrad's `add control flow` rewrite deadlocks →
  `unified_rewrite` raises `KeyError[root]`.
- **Validate gap:** `acc[ii, jj]` with `ii`/`jj` being Range vars
  sails through `validate_kernel` and only fails deep in
  `tinygrad/renderer/ptx.py:199`
  (`r[u.src[0]][u.src[1].arg]` with `.arg is None`).

## Task A — Validate-side: reject register access with non-constant index

**Files:** `tilegrad/validate.py`, `tests/test_validate.py`

1. Track which buffer names are registers. Pass a `register_buffers: set[str]`
   alongside the existing `buffers: set[str]` (populated when `Alloc` with
   `space=="register"` is processed at top level in `validate_kernel`).
2. Add helper `validate_reg_index(index, indices)` that walks an
   `Index2D`/`Var`/str index expression and returns True iff it
   transitively references any Range var name currently in `indices`.
   Stops at `Const`/int/float (constant — safe).
3. In `validate_store`, when `stmt.buffer in register_buffers`:
   If `validate_reg_index(stmt.index, indices)` is True, raise
   `ValueError(f"register buffer '{stmt.buffer}' cannot be indexed by a range variable: {stmt.index!r} - tinygrad requires constant register indices")`.
   Apply to both `Set` and `SetIf` paths (shared `validate_store`
   call covers both).
4. Update `validate_kernel`, `validate_range`, `validate_store`
   signatures so `register_buffers` propagates like `buffers` already does.
5. In `tests/test_validate.py` add:
   - `test_register_indexed_by_range_var_fails` — `Alloc("acc", 4, "float32", "register")` + `Set("acc", Index2D("ii", "jj", 2), 0)` inside `Range("ii")`/`Range("jj")`. Assert `ValueError` with new message.
   - `test_register_constant_index_ok` — same shape but `Set("acc", 0, 0)` (constant) inside the ranges — must pass.
   - `test_register_1d_const_in_loop_ok` — `Set("acc", 0, 0)` inside `Range("i")` — confirms constant 0 always allowed.

## Task B — Lowerer-side: fix register scope-chain on `set`/`set_if` at `axis="loop"`

**Files:** `tilegrad/lowerer.py`, `tests/test_lowerer.py`, `tests/test_runtime.py`, `examples/builder_tiled_gemm_bm_bn_accum_tile.py`

1. Replace the asymmetric ternary at `lowerer.py:147` with a narrower form
   that only passes the register-end ranges through `set(end=...)` for
   conditional loop-axis register sets (`SetIf`) and leaves plain loop-axis
   `Set` behavior unchanged:
   ```python
   if axis == "reduce":
     ends = (recurrence_range, *register_end_ranges)
   elif cond is not None:
     ends = register_end_ranges
   else:
     ends = ()
   next_buf = target.set(val, end=ends) if ends else target.set(val)
   ```
   The earlier broad loop-axis change regressed existing GEMM tests, so the
   fix must be scoped only to the failing `SetIf` shape.
2. Rename `RuntimeError("register scope mismatch")` at line 132 to
   include context: `f"register scope mismatch for {stmt.buffer}: current={current_scope} desired={desired_scope}"`.
3. In `tests/test_lowerer.py` add `test_lower_set_if_with_nested_ranges_register`:
   lower a minimal scalar `SetIf` kernel and assert the register `STORE`
   containing the `WHERE` is wrapped in `END(i)`. This targets the actual
   failing shape without invoking unrelated tinygrad CFG behavior.
4. Do not add a runtime test yet. The previous runtime/codegen attempt was too
   broad and surfaced separate tinygrad CFG limitations. First land the
   structural lowerer regression.
5. In `examples/builder_tiled_gemm_bm_bn_accum_tile.py`: the current
   early-exit guard stays. Update the skip message to reference the
   validate-time error name so users know where to look.

## Task C — Verification

1. `python -m pytest tests/` → expect 132+ passed, 1 skipped (the
   BMxBN 2D-reg example test stays skipped), 0 failed.
2. All 23 non-skipped examples still produce identical output.
3. Manually try a kernel with `acc[ii, jj]` (2D reg index) — should
   now fail at `validate_kernel` with a clear `ValueError`, not deep
   in tinygrad PTX.

## Out-of-scope (follow-up passes)

- Internal lowerer-side unrolling of 2D register tiles.
- WMMA tile primitive (`k.mma(a, b, acc, shape=...)` lowering to `Ops.WMMA`).
- General `parallel()` coverage (`builder_parallel_gemm.py`).
- Port tinygrad `test_custom_kernel.py` patterns (QKV attention, slice_sum, contract).
