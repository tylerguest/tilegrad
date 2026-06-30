from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from tilegrad.ir import Add, Alloc, Barrier, Const, FloorDiv, Index2D, Load, Mod, Mul, Range, Set, Store, Sub
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

def lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, value_mode, op):
  lhs = lower_expr(expr.lhs, env, indices, recurrence_buffer, recurrence_range, value_mode)
  rhs = lower_expr(expr.rhs, env, indices, recurrence_buffer, recurrence_range, value_mode)
  return op(lhs, rhs)

def lower_load(expr, env, indices, recurrence_buffer, recurrence_range, value_mode):
  idx = lower_expr(expr.index, env, indices, recurrence_buffer, recurrence_range, value_mode)
  buf = env[expr.buffer]
  if expr.buffer == recurrence_buffer and recurrence_range is not None:
    buf = buf.after(recurrence_range)
  buf = buf.flatten()
  if value_mode: return buf[idx]
  return buf.index(lower_index(idx)).load()

def lower_expr(expr, env, indices, recurrence_buffer=None, recurrence_range=None, value_mode=False):
  if isinstance(expr, (int, float)): return expr
  if isinstance(expr, str):
    if expr in indices: return indices[expr]
    raise NotImplementedError(expr)
  if isinstance(expr, Const): return expr.value
  if isinstance(expr, Add): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, value_mode, lambda lhs, rhs: lhs + rhs)
  if isinstance(expr, Sub): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, value_mode, lambda lhs, rhs: lhs - rhs)
  if isinstance(expr, Mul): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, value_mode, lambda lhs, rhs: lhs * rhs)
  if isinstance(expr, FloorDiv): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, value_mode, lambda lhs, rhs: lhs // rhs)
  if isinstance(expr, Mod): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, value_mode, lambda lhs, rhs: lhs % rhs)
  if isinstance(expr, Index2D):
    row = lower_expr(expr.row, env, indices, recurrence_buffer, recurrence_range, value_mode)
    col = lower_expr(expr.col, env, indices, recurrence_buffer, recurrence_range, value_mode)
    stride = lower_expr(expr.stride, env, indices, recurrence_buffer, recurrence_range, value_mode)
    return row * stride + col
  if isinstance(expr, Load): return lower_load(expr, env, indices, recurrence_buffer, recurrence_range, value_mode)
  raise NotImplementedError(type(expr).__name__)

def lower_store(stmt, env, effects, indices, active_ranges):
  idx = lower_expr(stmt.index, env, indices)
  val = lower_expr(stmt.value, env, indices)
  base = env[stmt.buffer]
  buf = base.flatten()
  if effects: buf = buf.after(effects[-1])
  if isinstance(val, UOp) and val.dtype != base.dtype.base: val = val.cast(base.dtype.base)
  effect = buf.index(lower_index(idx), ptr=True).store(val).end(*active_ranges)
  effects.append(effect)
  env[stmt.buffer] = base.after(effect)

def lower_set(stmt, env, updated_buffers, local_updated, indices, active_ranges, axis):
  idx = lower_expr(stmt.index, env, indices)
  recurrence_range = active_ranges[-1] if axis == "reduce" and active_ranges else None
  val = lower_expr(stmt.value, env, indices, recurrence_buffer=stmt.buffer, recurrence_range=recurrence_range, value_mode=True,)
  buf = env[stmt.buffer]
  if axis != "reduce" and active_ranges and buf.addrspace is AddrSpace.REG: buf = buf.after(*active_ranges)
  target = buf.flatten()[idx]
  env[stmt.buffer] = target.set(val, end=recurrence_range) if axis == "reduce" else target.set(val)
  updated_buffers.add(stmt.buffer)
  local_updated.add(stmt.buffer)

def lower_range(op, env, effects, updated_buffers, indices, range_slots, active_ranges=()):
  axis_type = AxisType.REDUCE if op.axis == "reduce" else AxisType.LOOP
  i = UOp.range(lower_shape(op.extent, env), range_slots[0], axis_type)
  range_slots[0] += 1
  indices = indices | {op.name: i}
  active_ranges = active_ranges + (i,)
  local_updated = set()
  for stmt in op.body:
    if isinstance(stmt, Range): local_updated |= lower_range(stmt, env, effects, updated_buffers, indices, range_slots, active_ranges)
    elif isinstance(stmt, Store): lower_store(stmt, env, effects, indices, active_ranges)
    elif isinstance(stmt, Set): lower_set(stmt, env, updated_buffers, local_updated, indices, active_ranges, op.axis)
    else: raise NotImplementedError(type(stmt).__name__)
  if op.axis == "loop":
    for name in local_updated:
      if name in env: env[name] = env[name].end(i)
  return local_updated

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
  env[op.name] = UOp.placeholder((lower_shape(op.shape, env),), lower_dtype(op.dtype), slot=slot, addrspace=addrspace,)

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
    elif isinstance(op, Set): lower_set(op, env, updated_buffers, set(), indices, (), "loop")
    elif isinstance(op, Range): lower_range(op, env, effects, updated_buffers, indices, range_slots)
    elif isinstance(op, Barrier): lower_barrier(env, effects)
    else: raise NotImplementedError(type(op).__name__)
  sinks = list(effects)
  sinks += [env[arg.name] for arg in kernel.args if arg.name in updated_buffers]
  if not sinks: raise ValueError("kernel must produce at least one effect")
  info = KernelInfo(name=kernel.name, opts_to_apply=()) if not effects else KernelInfo(name=kernel.name)
  return UOp.sink(*sinks, arg=info)
