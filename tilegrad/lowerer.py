from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from tilegrad.ir import Add, Alloc, Barrier, Const, FloorDiv, Load, Mod, Mul, Range, Store, Index2D, Set, Sub
from tilegrad.validate import validate_kernel

def lower_shape(shape, env):
  if isinstance(shape, int): return shape
  if shape.endswith(".numel"): return env[shape[:-6]].max_numel()
  raise NotImplementedError(shape)

def lower_dtype(dtype):
  if not isinstance(dtype, str): return dtype
  if not hasattr(dtypes, dtype): raise NotImplementedError(dtype)
  return getattr(dtypes, dtype)

def lower_index(idx):
  return idx if isinstance(idx, UOp) else UOp.const(dtypes.weakint, idx)

def lower_expr(expr, env, indices, range_uop=None, axis=None, value_mode=False, current_set_buffer=None):
  if isinstance(expr, (int, float)): return expr
  if isinstance(expr, str):
    if expr in indices: return indices[expr]
    raise NotImplementedError(expr)
  if isinstance(expr, Const): return expr.value
  if isinstance(expr, Add): return lower_expr(expr.lhs, env, indices, range_uop, axis, value_mode, current_set_buffer) + lower_expr(expr.rhs, env, indices, range_uop, axis, value_mode, current_set_buffer)
  if isinstance(expr, Sub): return lower_expr(expr.lhs, env, indices, range_uop, axis, value_mode, current_set_buffer) - lower_expr(expr.rhs, env, indices, range_uop, axis, value_mode, current_set_buffer)
  if isinstance(expr, Mul): return lower_expr(expr.lhs, env, indices, range_uop, axis, value_mode, current_set_buffer) * lower_expr(expr.rhs, env, indices, range_uop, axis, value_mode, current_set_buffer)
  if isinstance(expr, FloorDiv): return lower_expr(expr.lhs, env, indices, range_uop, axis, value_mode, current_set_buffer) // lower_expr(expr.rhs, env, indices, range_uop, axis, value_mode, current_set_buffer)
  if isinstance(expr, Mod): return lower_expr(expr.lhs, env, indices, range_uop, axis, value_mode, current_set_buffer) % lower_expr(expr.rhs, env, indices, range_uop, axis, value_mode, current_set_buffer)
  if isinstance(expr, Index2D):
    row = lower_expr(expr.row, env, indices, range_uop, axis, value_mode, current_set_buffer)
    col = lower_expr(expr.col, env, indices, range_uop, axis, value_mode, current_set_buffer)
    stride = lower_expr(expr.stride, env, indices, range_uop, axis, value_mode, current_set_buffer)
    return row * stride + col
  if isinstance(expr, Load):
    idx = lower_expr(expr.index, env, indices, range_uop, axis, value_mode, current_set_buffer)
    buf = env[expr.buffer]
    if axis == "reduce" and expr.buffer == current_set_buffer and range_uop is not None: buf = buf.after(range_uop)
    buf = buf.flatten()
    if value_mode: return buf[idx]
    return buf.index(lower_index(idx)).load()
  raise NotImplementedError(type(expr).__name__)

def lower_stmt(stmt, env, effects, updated_buffers, indices, range_uop=None, axis="loop"):
  idx = lower_expr(stmt.index, env, indices)
  base = env[stmt.buffer]
  buf = base.flatten()
  if effects: buf = buf.after(effects[-1])
  if isinstance(stmt, Store):
    val = lower_expr(stmt.value, env, indices)
    effect = buf.index(lower_index(idx), ptr=True).store(val).end(range_uop)
    effects.append(effect)
    env[stmt.buffer] = base.after(effect)
  elif isinstance(stmt, Set):
    val = lower_expr(stmt.value, env, indices, range_uop, axis, value_mode=True, current_set_buffer=stmt.buffer)
    target = buf[idx]
    env[stmt.buffer] = target.set(val, end=range_uop) if axis == "reduce" else target.set(val)
    updated_buffers.add(stmt.buffer)
  else: raise NotImplementedError(type(stmt).__name__)

def lower_range(op, env, effects, updated_buffers, indices, range_slots):
  axis_type = AxisType.REDUCE if op.axis == "reduce" else AxisType.LOOP
  i = UOp.range(lower_shape(op.extent, env), range_slots[0], axis_type)
  range_slots[0] += 1
  indices = indices | {op.name: i}
  for stmt in op.body:
    if not isinstance(stmt, (Store, Set)): raise NotImplementedError(type(stmt).__name__)
    lower_stmt(stmt, env, effects, updated_buffers, indices, i, op.axis)

def lower_alloc(op, env, shared_slots, register_slots):
  if op.name in env: raise ValueError(f"duplicate buffer name: {op.name}")
  if op.space == "shared":
    slot = shared_slots[0]
    shared_slots[0] += 1
    addrspace = AddrSpace.LOCAL
  elif op.space == "register":
    slot = register_slots[0]
    register_slots[0] += 1
    addrspace = AddrSpace.REG
  else: raise NotImplementedError(op.space)
  env[op.name] = UOp.placeholder(
    (lower_shape(op.shape, env),),
    lower_dtype(op.dtype),
    slot=slot,
    addrspace=addrspace,
  )

def lower_barrier(env, effects):
  if not effects: raise ValueError("barrier requires a previous effect")
  bar = effects[-1].barrier(*effects[:-1])
  effects.append(bar)
  for name, buf in tuple(env.items()): 
    if buf.addrspace is AddrSpace.LOCAL: env[name] = buf.after(bar)

def lower_kernel(kernel, *args: UOp) -> UOp:
  validate_kernel(kernel)
  if len(args) != len(kernel.args): raise ValueError(f"expected {len(kernel.args)} args, got {len(args)}")
  env = {arg.name: uop for arg, uop in zip(kernel.args, args)}
  effects = []
  updated_buffers = set()
  indices = {}
  shared_slots = [0]
  register_slots = [0]
  range_slots = [0]
  for op in kernel.body:
    if isinstance(op, Alloc): lower_alloc(op, env, shared_slots, register_slots)
    elif isinstance(op, Set): lower_stmt(op, env, effects, updated_buffers, indices)
    elif isinstance(op, Range): lower_range(op, env, effects, updated_buffers, indices, range_slots)
    elif isinstance(op, Barrier): lower_barrier(env, effects)
    else: raise NotImplementedError(type(op).__name__)
  sinks = list(effects)
  sinks += [env[arg.name] for arg in kernel.args if arg.name in updated_buffers]
  if not sinks: raise ValueError("kernel must produce at least one effect")
  info = KernelInfo(name=kernel.name, opts_to_apply=()) if not effects else KernelInfo(name=kernel.name)
  return UOp.sink(*sinks, arg=info)
