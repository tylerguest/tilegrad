from tilegrad.ir import Add, Alloc, Barrier, Const, FloorDiv, Kernel, Load, Mod, Mul, Range, Store

def validate_shape(shape, buffers):
  if isinstance(shape, int):
    if shape <= 0: raise ValueError(f"shape must be positive: {shape}")
    return
  if isinstance(shape, str):
    if shape.endswith(".numel"):
      name = shape[:-6]
      if name not in buffers: raise ValueError(f"unknown buffer: {name}")
      return 
    raise ValueError(f"unknown shape expression: {shape}")
  raise TypeError(f"unsupported shape type: {type(shape).__name__}")

def validate_expr(expr, buffers, indices):
  if isinstance(expr, (int, float)): return
  if isinstance(expr, str):
    if expr not in indices: raise ValueError(f"unknown index variable: {expr}")
    return
  if isinstance(expr, Const):
    validate_expr(expr.value, buffers, indices)
    return
  if isinstance(expr, (Add, Mul, FloorDiv, Mod)):
    validate_expr(expr.lhs, buffers, indices)
    validate_expr(expr.rhs, buffers, indices)
    return
  if isinstance(expr, Load):
    if expr.buffer not in buffers: raise ValueError(f"unknown buffer: {expr.buffer}")
    validate_expr(expr.index, buffers, indices)
    return
  raise TypeError(f"unsupported expression: {type(expr).__name__}")

def validate_store(stmt, buffers, indices):
  if stmt.buffer not in buffers: raise ValueError(f"unknown buffer: {stmt.buffer}")
  validate_expr(stmt.index, buffers, indices)
  validate_expr(stmt.value, buffers, indices)

def validate_range(op, buffers, indices, saw_effect):
  validate_shape(op.extent, buffers)
  if op.name in indices: raise ValueError(f"duplicate range variable: {op.name}")
  indices = indices | {op.name}
  for stmt in op.body:
    if isinstance(stmt, Store):
      validate_store(stmt, buffers, indices)
      saw_effect[0] = True 
    elif isinstance(stmt, Range): validate_range(stmt, buffers, indices, saw_effect)
    else: raise TypeError(f"unsupported range statement: {type(stmt).__name__}")

def validate_kernel(kernel):
  if not isinstance(kernel, Kernel): raise TypeError(f"expected Kernel, got {type(kernel).__name__}")
  buffers = set()
  for arg in kernel.args:
    if arg.name in buffers: raise ValueError(f"duplicate arg name: {arg.name}")
    buffers.add(arg.name)
  saw_effect = [False]
  for op in kernel.body:
    if isinstance(op, Alloc):
      if op.name in buffers: raise ValueError(f"duplicate buffer name: {op.name}")
      validate_shape(op.shape, buffers)
      if op.space != "shared": raise NotImplementedError(op.space)
      buffers.add(op.name)
    elif isinstance(op, Range): validate_range(op, buffers, set(), saw_effect)
    elif isinstance(op, Barrier):
      if not saw_effect[0]: raise ValueError("barrier requires a previous effect")
      saw_effect[0] = True
    else: raise TypeError(f"unsupported kernel statement: {type(op).__name__}")
  if not saw_effect[0]: raise ValueError("kernel must produce at least one effect")