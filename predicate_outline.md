# Predicate And Guarded Store Outline

This outline tracks the next implementation milestone: predicates and guarded effects.

## Goal

Unlock safe edge tiles and partial stores for symbolic builder kernels.

Current `parallel` and 2D buffer indexing work cleanly for static, perfectly divisible shapes. Predicates make kernels usable when a tile extends past the logical problem bounds.

Target first syntax:

```python
with k.parallel(BM, BN) as (i, j):
  k.store_if((i < M) & (j < N), C, (i, j), A[i, j] + B[i, j])
```

Possible later syntax:

```python
with k.if_((i < M) & (j < N)):
  C[i, j] = A[i, j] + B[i, j]
```

## Implementation Steps

1. Add predicate IR nodes in `tilegrad/ir.py`.

   Add comparison expressions:

   ```python
   Lt(lhs, rhs)
   Le(lhs, rhs)
   Gt(lhs, rhs)
   Ge(lhs, rhs)
   Eq(lhs, rhs)
   Ne(lhs, rhs)
   ```

   Add boolean expressions:

   ```python
   And(lhs, rhs)
   Or(lhs, rhs)
   Not(x)
   ```

2. Add helper constructors.

   Prefer helpers first so structural dataclass equality remains easy to reason about while the API settles:

   ```python
   lt(a, b)
   le(a, b)
   gt(a, b)
   ge(a, b)
   eq(a, b)
   ne(a, b)
   and_(a, b)
   or_(a, b)
   not_(x)
   ```

3. Add safe operator overloads on `Expr`.

   Add these first:

   ```python
   <
   <=
   >
   >=
   &
   |
   ```

   Defer `==` and `!=` until the equality tradeoff is explicit, because dataclass equality is currently useful in tests.

4. Add guarded statement IR.

   Start smaller than a full `If` block. Add one guarded effect node:

   ```python
   Guard(cond, stmt)
   ```

   Or, if simpler for lowering, add explicit guarded set/store nodes:

   ```python
   SetIf(cond, buffer, index, value)
   StoreIf(cond, buffer, index, value)
   ```

   Recommendation: start with `SetIf`, because public `BufferRef.__setitem__` currently emits `Set` and output assignment uses `Set` most often.

5. Add builder API.

   First API:

   ```python
   k.set_if(cond, buffer, index, value)
   k.store_if(cond, buffer, index, value)
   ```

   `buffer` should accept either raw string names or `BufferRef` objects. `index` should accept flat indices or tuple indices for shaped `BufferRef` objects.

6. Validate predicates and guarded effects.

   Update `validate_expr` to accept comparison and boolean expressions.

   Update range/kernel statement validation to accept the guarded statement node.

   Validation should ensure:

   - Predicate expressions only reference known loop vars and valid expressions.
   - Guarded statements reference known buffers.
   - Guarded statement indices and values validate like ordinary `Set` or `Store`.

7. Lower predicates.

   Update `lower_expr` to lower comparison and boolean nodes to tinygrad UOps.

   Expected lowering shape:

   ```python
   Lt -> lhs < rhs
   Le -> lhs <= rhs if supported, otherwise lhs < rhs + 1 for integer predicates or use not/gt form
   Gt -> lhs > rhs if supported, otherwise rhs < lhs
   Ge -> lhs >= rhs if supported, otherwise rhs <= lhs
   And -> lhs & rhs
   Or -> lhs | rhs
   Not -> logical not if available, otherwise cond == False if supported
   ```

8. Lower guarded effects.

   Implement the smallest working semantics.

   Possible lowering strategies:

   - For stores, use tinygrad invalid/gated index semantics if available.
   - For `SetIf`, lower as conditional value update if tinygrad supports `where`, otherwise start with guarded global `StoreIf` only.

   Keep this step narrow. The goal is correctness for masked edge writes, not a general control-flow system.

9. Add tests.

   Add IR tests for predicate nodes and helpers.

   Add validation tests:

   - valid guarded set/store
   - unknown index in predicate fails
   - unknown buffer in guarded effect fails

   Add builder tests:

   - `k.store_if((i < 3), out, i, inp[i])` creates expected IR
   - 2D shaped `BufferRef` tuple index works in `store_if`

   Add lowerer/runtime tests:

   - masked 1D copy where only first N elements are written
   - masked 2D edge-tile vecadd

10. Add one example.

   Suggested example:

   ```text
   examples/builder_guarded_vecadd_2d.py
   ```

   It should demonstrate a tile larger than the logical shape and guard the out-of-bounds stores.

11. Update public exports.

   Export stable predicate helpers from `tilegrad.__init__` once the names settle.

12. Update docs.

   Add a short README section showing guarded stores as the recommended way to handle edge tiles.

## Non-Goals For This Milestone

- Do not implement full `if` block syntax first.
- Do not add AST parsing.
- Do not attempt `for i, j in T.Parallel(...)` yet.
- Do not add async copies, tensor cores, or autotuning.
- Do not generalize to arbitrary N-D indexing unless needed by the guarded-store tests.

## Design Note: Equality Operators

Be careful with `Expr.__eq__` and `Expr.__ne__`.

Dataclass equality currently makes tests straightforward:

```python
Add(Var("i"), 1) == Add(Var("i"), 1)
```

If `Expr.__eq__` becomes an IR comparison operator, that structural equality behavior changes. Start with helper functions such as `eq(a, b)` and `ne(a, b)`, then decide later whether operator syntax is worth the tradeoff.
