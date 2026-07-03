from tilegrad.ir import Alloc, Barrier, BinaryExpr, Const, FragmentAlloc, FragmentClear, FragmentGemm, FragmentStore, Kernel, Load, LoadIf, Not, Range, Store, StoreIf, Index2D, Set, SetIf, Var

VALID_RANGE_AXES = ("loop", "reduce", "global", "local", "unroll")

def validate_shape(shape, buffers):
  if isinstance(shape, int):
    if shape <= 0: raise ValueError(f"shape must be positive: {shape}")
    return
  if isinstance(shape, str):
    if shape.endswith(".numel"):
      name = shape[:-6]
      if name not in buffers: raise ValueError(f"unknown buffer: {name}")
      return 
    if ".shape." in shape:
      name, dim = shape.rsplit(".shape.", 1)
      if name not in buffers: raise ValueError(f"unknown buffer: {name}")
      if not dim.isdigit(): raise ValueError(f"invalid shape dimension: {shape}")
      return
    raise ValueError(f"unknown shape expression: {shape}")
  raise TypeError(f"unsupported shape type: {type(shape).__name__}")

def validate_fragment_shape(shape):
  if not isinstance(shape, tuple) or len(shape) != 2:
    raise ValueError("fragment shape must be a 2D tuple")
  if not all(isinstance(dim, int) and dim > 0 for dim in shape):
    raise ValueError(f"fragment shape must contain positive integers: {shape}")

def validate_expr(expr, buffers, indices, register_buffers=None, fragments=None):
  if isinstance(expr, (int, float)): return
  if isinstance(expr, str):
    if expr not in indices: raise ValueError(f"unknown index variable: {expr}")
    return
  if isinstance(expr, Var):
    if expr.name not in indices: raise ValueError(f"unknown index variable: {expr.name}")
    return
  if isinstance(expr, Const):
    if not isinstance(expr.value, (int, float)): raise TypeError(f"const value must be int or float, got {type(expr.value).__name__}")
    return
  if isinstance(expr, Not):
    validate_expr(expr.x, buffers, indices, register_buffers, fragments)
    return
  if isinstance(expr, BinaryExpr):
    validate_expr(expr.lhs, buffers, indices, register_buffers, fragments)
    validate_expr(expr.rhs, buffers, indices, register_buffers, fragments)
    return
  if isinstance(expr, Load):
    if expr.buffer not in buffers: raise ValueError(f"unknown buffer: {expr.buffer}")
    if fragments is not None and expr.buffer in fragments:
      raise ValueError(f"fragment buffer '{expr.buffer}' cannot be used with scalar Load/Set/Store")
    validate_expr(expr.index, buffers, indices, register_buffers, fragments)
    if register_buffers is not None and expr.buffer in register_buffers and _index_references_range_var(expr.index, indices):
      raise ValueError(
        f"register buffer '{expr.buffer}' cannot be indexed by a range variable: "
        f"{expr.index!r} - tinygrad requires constant register indices")
    return
  if isinstance(expr, LoadIf):
    validate_expr(expr.cond, buffers, indices, register_buffers, fragments)
    if expr.buffer not in buffers: raise ValueError(f"unknown buffer: {expr.buffer}")
    if fragments is not None and expr.buffer in fragments:
      raise ValueError(f"fragment buffer '{expr.buffer}' cannot be used with scalar Load/Set/Store")
    validate_expr(expr.index, buffers, indices, register_buffers, fragments)
    if register_buffers is not None and expr.buffer in register_buffers and _index_references_range_var(expr.index, indices):
      raise ValueError(
        f"register buffer '{expr.buffer}' cannot be indexed by a range variable: "
        f"{expr.index!r} - tinygrad requires constant register indices")
    return
  if isinstance(expr, Index2D):
    validate_expr(expr.row, buffers, indices, register_buffers, fragments)
    validate_expr(expr.col, buffers, indices, register_buffers, fragments)
    validate_expr(expr.stride, buffers, indices, register_buffers, fragments)
    return
  raise TypeError(f"unsupported expression: {type(expr).__name__}")

def _index_references_range_var(index, indices):
  # Applied to the *index* of a Set/SetIf/Store against a register buffer.
  # Register indices must be constant; any Range var reference makes them non-constant.
  if isinstance(index, (int, float)): return False
  if isinstance(index, str): return index in indices
  if isinstance(index, Var): return index.name in indices
  if isinstance(index, Const): return False
  if isinstance(index, Index2D):
    return _index_references_range_var(index.row, indices) or _index_references_range_var(index.col, indices) or \
           _index_references_range_var(index.stride, indices)
  if isinstance(index, BinaryExpr):
    return _index_references_range_var(index.lhs, indices) or _index_references_range_var(index.rhs, indices)
  if isinstance(index, Not): return _index_references_range_var(index.x, indices)
  # Loads are values, not indices. A reg index like Load("smem", "i") is itself
  # a runtime value — non-constant — so treat as forbidden for register indexing.
  if isinstance(index, (Load, LoadIf)): return True
  return False

def validate_store(stmt, buffers, indices, register_buffers=None, fragments=None):
  if stmt.buffer not in buffers: raise ValueError(f"unknown buffer: {stmt.buffer}")
  if fragments is not None and stmt.buffer in fragments:
    raise ValueError(f"fragment buffer '{stmt.buffer}' cannot be used with scalar Load/Set/Store")
  validate_expr(stmt.index, buffers, indices, register_buffers, fragments)
  validate_expr(stmt.value, buffers, indices, register_buffers, fragments)
  if register_buffers is not None and stmt.buffer in register_buffers:
    if _index_references_range_var(stmt.index, indices):
      raise ValueError(
        f"register buffer '{stmt.buffer}' cannot be indexed by a range variable: "
        f"{stmt.index!r} - tinygrad requires constant register indices")

def _fragment_gemm_effective_shapes(stmt):
  a_m, a_k = (stmt.a_shape[1], stmt.a_shape[0]) if stmt.trans_a else stmt.a_shape
  b_k, b_n = (stmt.b_shape[1], stmt.b_shape[0]) if stmt.trans_b else stmt.b_shape
  return (a_m, a_k), (b_k, b_n), stmt.c_shape

def validate_fragment_gemm(stmt, buffers, fragments):
  if stmt.a not in buffers: raise ValueError(f"unknown buffer: {stmt.a}")
  if stmt.b not in buffers: raise ValueError(f"unknown buffer: {stmt.b}")
  if stmt.c not in buffers: raise ValueError(f"unknown buffer: {stmt.c}")
  if stmt.c not in fragments: raise ValueError(f"fragment gemm C must be a fragment: {stmt.c}")
  for shape in (stmt.a_shape, stmt.b_shape, stmt.c_shape): validate_fragment_shape(shape)
  (a_m, a_k), (b_k, b_n), c_shape = _fragment_gemm_effective_shapes(stmt)
  if a_k != b_k or c_shape != (a_m, b_n):
    raise ValueError(f"fragment gemm shape mismatch: A{stmt.a_shape} B{stmt.b_shape} C{stmt.c_shape}")

def validate_fragment_store(stmt, buffers, indices, fragments, register_buffers):
  if stmt.src not in buffers: raise ValueError(f"unknown buffer: {stmt.src}")
  if stmt.src not in fragments: raise ValueError(f"fragment store src must be a fragment: {stmt.src}")
  if stmt.dst not in buffers: raise ValueError(f"unknown buffer: {stmt.dst}")
  if stmt.dst in fragments: raise ValueError(f"fragment store dst cannot be a fragment: {stmt.dst}")
  validate_expr(stmt.dst_row, buffers, indices, register_buffers, fragments)
  validate_expr(stmt.dst_col, buffers, indices, register_buffers, fragments)
  validate_expr(stmt.dst_stride, buffers, indices, register_buffers, fragments)
  if isinstance(stmt.dst_stride, int) and stmt.dst_stride <= 0:
    raise ValueError(f"fragment store dst_stride must be positive: {stmt.dst_stride}")
  if stmt.guard is not None: validate_expr(stmt.guard, buffers, indices, register_buffers, fragments)
  if stmt.bounds is not None:
    if not isinstance(stmt.bounds, tuple) or len(stmt.bounds) != 2:
      raise ValueError("fragment store bounds must be a 2D tuple")
    validate_expr(stmt.bounds[0], buffers, indices, register_buffers, fragments)
    validate_expr(stmt.bounds[1], buffers, indices, register_buffers, fragments)

def validate_fragment_stmt(stmt, buffers, indices, saw_effect, register_buffers, fragments):
  if isinstance(stmt, FragmentClear):
    if stmt.buffer not in buffers: raise ValueError(f"unknown buffer: {stmt.buffer}")
    if stmt.buffer not in fragments: raise ValueError(f"fragment clear buffer must be a fragment: {stmt.buffer}")
  elif isinstance(stmt, FragmentGemm):
    validate_fragment_gemm(stmt, buffers, fragments)
  elif isinstance(stmt, FragmentStore):
    validate_fragment_store(stmt, buffers, indices, fragments, register_buffers)
  else:
    raise TypeError(f"unsupported fragment statement: {type(stmt).__name__}")
  saw_effect[0] = True

def validate_range(op, buffers, indices, saw_effect, register_buffers=None, fragments=None):
  validate_shape(op.extent, buffers)
  if op.name in indices: raise ValueError(f"duplicate range variable: {op.name}")
  if op.axis not in VALID_RANGE_AXES: raise ValueError(f"unknown range axis: {op.axis}")
  indices = indices | {op.name}
  for stmt in op.body:
    if isinstance(stmt, (StoreIf, SetIf)):
      validate_expr(stmt.cond, buffers, indices, register_buffers, fragments)
      validate_store(stmt, buffers, indices, register_buffers, fragments)
      saw_effect[0] = True
    elif isinstance(stmt, (Store, Set)):
      validate_store(stmt, buffers, indices, register_buffers, fragments)
      saw_effect[0] = True
    elif isinstance(stmt, Range): validate_range(stmt, buffers, indices, saw_effect, register_buffers, fragments)
    elif isinstance(stmt, (FragmentClear, FragmentGemm, FragmentStore)):
      validate_fragment_stmt(stmt, buffers, indices, saw_effect, register_buffers, fragments)
    elif isinstance(stmt, Barrier):
      if not saw_effect[0]: raise ValueError("barrier requires a previous effect")
      saw_effect[0] = True
    else: raise TypeError(f"unsupported range statement: {type(stmt).__name__}")

def validate_kernel(kernel):
  if not isinstance(kernel, Kernel): raise TypeError(f"expected Kernel, got {type(kernel).__name__}")
  buffers = set()
  register_buffers = set()
  fragments = {}
  for arg in kernel.args:
    if arg.name in buffers: raise ValueError(f"duplicate arg name: {arg.name}")
    buffers.add(arg.name)
  saw_effect = [False]
  for op in kernel.body:
    if isinstance(op, Alloc):
      if op.name in buffers: raise ValueError(f"duplicate buffer name: {op.name}")
      if op.space not in ("shared", "register"): raise NotImplementedError(op.space)
      validate_shape(op.shape, buffers)
      buffers.add(op.name)
      if op.space == "register": register_buffers.add(op.name)
    elif isinstance(op, FragmentAlloc):
      if op.name in buffers: raise ValueError(f"duplicate buffer name: {op.name}")
      validate_fragment_shape(op.shape)
      buffers.add(op.name)
      fragments[op.name] = op
    elif isinstance(op, SetIf):
      validate_expr(op.cond, buffers, set(), register_buffers, fragments)
      validate_store(op, buffers, set(), register_buffers, fragments)
      saw_effect[0] = True
    elif isinstance(op, Set):
      validate_store(op, buffers, set(), register_buffers, fragments)
      saw_effect[0] = True
    elif isinstance(op, StoreIf):
      validate_expr(op.cond, buffers, set(), register_buffers, fragments)
      validate_store(op, buffers, set(), register_buffers, fragments)
      saw_effect[0] = True
    elif isinstance(op, Store):
      validate_store(op, buffers, set(), register_buffers, fragments)
      saw_effect[0] = True
    elif isinstance(op, Range): validate_range(op, buffers, set(), saw_effect, register_buffers, fragments)
    elif isinstance(op, (FragmentClear, FragmentGemm, FragmentStore)):
      validate_fragment_stmt(op, buffers, set(), saw_effect, register_buffers, fragments)
    elif isinstance(op, Barrier):
      if not saw_effect[0]: raise ValueError("barrier requires a previous effect")
      saw_effect[0] = True
    else: raise TypeError(f"unsupported kernel statement: {type(op).__name__}")
  if not saw_effect[0]: raise ValueError("kernel must produce at least one effect")
