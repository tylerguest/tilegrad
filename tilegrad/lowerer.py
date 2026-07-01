from tinygrad.dtype import AddrSpace, Invalid, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from tilegrad.ir import Add, Alloc, And, Barrier, Const, Eq, FloorDiv, Ge, Gt, Index2D, Le, Load, LoadIf, Lt, Mod, Mul, Ne, Not, Or, Range, Set, SetIf, Store, StoreIf, Sub, Var
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

def lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, op):
  lhs = lower_expr(expr.lhs, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode)
  rhs = lower_expr(expr.rhs, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode)
  return op(lhs, rhs)

def lower_load(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode):
  idx = lower_expr(expr.index, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode)
  buf = env[expr.buffer]
  if expr.buffer == recurrence_buffer and recurrence_range is not None:
    if recurrence_uop is not None: buf = recurrence_uop
    buf = buf.after(recurrence_range)
  buf = buf.flatten()
  if value_mode: return buf[idx]
  return buf.index(lower_index(idx)).load()

def lower_load_if(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode):
  cond = lower_expr(expr.cond, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode)
  idx = lower_expr(expr.index, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode)
  buf = env[expr.buffer]
  if expr.buffer == recurrence_buffer and recurrence_range is not None:
    if recurrence_uop is not None: buf = recurrence_uop
    buf = buf.after(recurrence_range)
  buf = buf.flatten()
  if value_mode:
    val = buf[idx]
    return cond.where(val, val.const_like(0)) if isinstance(cond, UOp) else val if cond else val.const_like(0)
  idx = lower_index(idx)
  guarded_idx = idx.valid(cond) if isinstance(cond, UOp) else idx if cond else idx.const_like(Invalid)
  return buf.index(guarded_idx).load()

def lower_expr(expr, env, indices, recurrence_buffer=None, recurrence_range=None, recurrence_uop=None, value_mode=False):
  if isinstance(expr, (int, float)): return expr
  if isinstance(expr, str):
    if expr in indices: return indices[expr]
    raise NotImplementedError(expr)
  if isinstance(expr, Var): 
    if expr.name in indices: return indices[expr.name]
    raise NotImplementedError(expr.name)
  if isinstance(expr, Const): return expr.value
  if isinstance(expr, Add): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: lhs + rhs)
  if isinstance(expr, Sub): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: lhs - rhs)
  if isinstance(expr, Mul): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: lhs * rhs)
  if isinstance(expr, FloorDiv): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: lhs // rhs)
  if isinstance(expr, Mod): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: lhs % rhs)
  if isinstance(expr, Lt): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: lhs < rhs)
  if isinstance(expr, Le): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: (rhs < lhs).logical_not())
  if isinstance(expr, Gt): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: rhs < lhs)
  if isinstance(expr, Ge): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: (lhs < rhs).logical_not())
  if isinstance(expr, Eq): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: lhs.eq(rhs))
  if isinstance(expr, Ne): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: lhs.ne(rhs))
  if isinstance(expr, And): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: lhs & rhs)
  if isinstance(expr, Or): return lower_binary(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode, lambda lhs, rhs: lhs | rhs)
  if isinstance(expr, Not):
    x = lower_expr(expr.x, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode)
    return x.logical_not()

  if isinstance(expr, Index2D):
    row = lower_expr(expr.row, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode)
    col = lower_expr(expr.col, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode)
    stride = lower_expr(expr.stride, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode)
    return row * stride + col
  if isinstance(expr, LoadIf): return lower_load_if(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode)
  if isinstance(expr, Load): return lower_load(expr, env, indices, recurrence_buffer, recurrence_range, recurrence_uop, value_mode)
  raise NotImplementedError(type(expr).__name__)

def lower_store_if(stmt, env, effects, sink_effects, buffer_effects, pending_shared, indices, active_ranges):
  cond = lower_expr(stmt.cond, env, indices)
  idx = lower_expr(stmt.index, env, indices)
  val = lower_expr(stmt.value, env, indices)
  base = env[stmt.buffer]
  buf = base.flatten()
  if stmt.buffer in buffer_effects: buf = buf.after(buffer_effects[stmt.buffer])
  if isinstance(val, UOp) and val.dtype != base.dtype.base: val = val.cast(base.dtype.base)
  idx = lower_index(idx)
  guarded_idx = cond.where(idx, idx.const_like(Invalid))
  effect = buf.index(guarded_idx, ptr=True).store(val)
  if base.addrspace is AddrSpace.LOCAL:
    buffer_effects[stmt.buffer] = effect
    pending_shared.append((effect, active_ranges))
  else:
    ended = effect.end(*active_ranges)
    buffer_effects[stmt.buffer] = ended
    effects.append(ended)
    sink_effects.append(ended)
    env[stmt.buffer] = base.after(ended)

def lower_store(stmt, env, effects, sink_effects, buffer_effects, pending_shared, indices, active_ranges):
  idx = lower_expr(stmt.index, env, indices)
  val = lower_expr(stmt.value, env, indices)
  base = env[stmt.buffer]
  buf = base.flatten()
  if stmt.buffer in buffer_effects: buf = buf.after(buffer_effects[stmt.buffer])
  if isinstance(val, UOp) and val.dtype != base.dtype.base: val = val.cast(base.dtype.base)
  effect = buf.index(lower_index(idx), ptr=True).store(val)
  if base.addrspace is AddrSpace.LOCAL:
    buffer_effects[stmt.buffer] = effect
    pending_shared.append((effect, active_ranges))
  else:
    ended = effect.end(*active_ranges)
    buffer_effects[stmt.buffer] = ended
    effects.append(ended)
    sink_effects.append(ended)
    env[stmt.buffer] = base.after(ended)

def lower_set(stmt, env, updated_buffers, local_updated, indices, active_ranges, axis, register_scopes, cond=None):
  idx = lower_expr(stmt.index, env, indices)
  recurrence_range = active_ranges[-1] if axis == "reduce" and active_ranges else None
  buf = env[stmt.buffer]
  recurrence_uop = None
  register_end_ranges = ()
  if buf.addrspace is AddrSpace.REG:
    desired_scope = active_ranges[:-1] if axis == "reduce" else active_ranges
    current_scope = register_scopes.get(stmt.buffer, ())
    if desired_scope[:len(current_scope)] != current_scope: raise RuntimeError("register scope mismatch")
    register_end_ranges = desired_scope[len(current_scope):]
    if register_end_ranges: buf = buf.after(*register_end_ranges)
    register_scopes[stmt.buffer] = desired_scope
    recurrence_uop = buf
  val = lower_expr(
    stmt.value, env, indices, recurrence_buffer=stmt.buffer,
    recurrence_range=recurrence_range, recurrence_uop=recurrence_uop, value_mode=True,)
  target = buf.flatten()[idx]
  if cond is not None:
    cond_uop = lower_expr(
      cond, env, indices, recurrence_buffer=stmt.buffer,
      recurrence_range=recurrence_range, recurrence_uop=recurrence_uop, value_mode=True,
    )
    val = cond_uop.where(val, target) if isinstance(cond_uop, UOp) else val if cond_uop else target
  next_buf = target.set(val, end=(recurrence_range, *register_end_ranges)) if axis == "reduce" else target.set(val)
  if buf.addrspace is not AddrSpace.REG and isinstance(val, UOp):
    leaked_ranges = [r for r in val.ranges if r not in active_ranges]
    if leaked_ranges: next_buf = next_buf.end(*leaked_ranges)
  env[stmt.buffer] = next_buf
  updated_buffers.add(stmt.buffer)
  local_updated.add(stmt.buffer)

def lower_range(op, env, effects, sink_effects, buffer_effects, pending_shared, updated_buffers, indices, range_slots, register_scopes, active_ranges=()):
  axis_type = AxisType.REDUCE if op.axis == "reduce" else AxisType.LOOP
  i = UOp.range(lower_shape(op.extent, env), range_slots[0], axis_type)
  range_slots[0] += 1
  indices = indices | {op.name: i}
  active_ranges = active_ranges + (i,)
  local_updated = set()
  for stmt in op.body:
    if isinstance(stmt, Range): local_updated |= lower_range(stmt, env, effects, sink_effects, buffer_effects, pending_shared, 
                                                             updated_buffers, indices, range_slots, register_scopes, active_ranges)
    elif isinstance(stmt, Store): lower_store(stmt, env, effects, sink_effects, buffer_effects, pending_shared, indices, active_ranges)
    elif isinstance(stmt, StoreIf): lower_store_if(stmt, env, effects, sink_effects, buffer_effects, pending_shared, indices, active_ranges)
    elif isinstance(stmt, SetIf): lower_set(stmt, env, updated_buffers, local_updated, indices, active_ranges, op.axis, register_scopes, cond=stmt.cond)
    elif isinstance(stmt, Set): lower_set(stmt, env, updated_buffers, local_updated, indices, active_ranges, op.axis, register_scopes)
    elif isinstance(stmt, Barrier): lower_barrier(env, effects, buffer_effects, pending_shared, active_ranges)
    else: raise NotImplementedError(type(stmt).__name__)
  if op.axis == "loop":
    for name in local_updated:
      if name in env and env[name].addrspace is not AddrSpace.REG: env[name] = env[name].end(i)
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

def lower_barrier(env, effects, buffer_effects, pending_shared, active_ranges=()):
  if not effects and not pending_shared: raise ValueError("barrier requires a previous effect")
  if pending_shared:
    stores = [s for s, _ in pending_shared]
    rngs = []
    for _, r in pending_shared:
      for x in r:
        if x not in rngs and x not in active_ranges: rngs.append(x)
    grouped = UOp.group(*stores) if len(stores) > 1 else stores[0]
    if rngs: grouped = grouped.end(*rngs)
    barrier_srcs = [grouped] + list(effects)
  else:
    barrier_srcs = list(effects)
  bar = UOp.barrier(*barrier_srcs)
  effects.clear()
  effects.append(bar)
  for name, buf in tuple(env.items()):
    if buf.addrspace is AddrSpace.LOCAL:
      env[name] = buf.after(bar)
      buffer_effects[name] = bar
  pending_shared.clear()

def lower_kernel(kernel, *args: UOp) -> UOp:
  validate_kernel(kernel)
  if len(args) != len(kernel.args): raise ValueError(f"expected {len(kernel.args)} args, got {len(args)}")
  env = {arg.name: uop for arg, uop in zip(kernel.args, args)}
  effects = []
  sink_effects = []
  buffer_effects = {}
  pending_shared = []
  updated_buffers = set()
  indices = {}
  shared_slots = [0]
  register_slots = [0]
  range_slots = [0]
  register_scopes = {}
  for op in kernel.body:
    if isinstance(op, Alloc): lower_alloc(op, env, shared_slots, register_slots)
    elif isinstance(op, SetIf): lower_set(op, env, updated_buffers, set(), indices, (), "loop", register_scopes, cond=op.cond)
    elif isinstance(op, Set): lower_set(op, env, updated_buffers, set(), indices, (), "loop", register_scopes)
    elif isinstance(op, Range): lower_range(op, env, effects, sink_effects, buffer_effects, pending_shared, updated_buffers, indices, range_slots, register_scopes)
    elif isinstance(op, Barrier): lower_barrier(env, effects, buffer_effects, pending_shared)
    else: raise NotImplementedError(type(op).__name__)
  if pending_shared:
    stores = [s for s, _ in pending_shared]
    rngs = []
    for _, r in pending_shared:
      for x in r:
        if x not in rngs: rngs.append(x)
    grouped = UOp.group(*stores) if len(stores) > 1 else stores[0]
    sink_effects.append(grouped.end(*rngs))
  sinks = list(sink_effects)
  sinks += [env[arg.name] for arg in kernel.args if arg.name in updated_buffers]
  if not sinks: raise ValueError("kernel must produce at least one effect")
  info = KernelInfo(name=kernel.name, opts_to_apply=()) if updated_buffers else KernelInfo(name=kernel.name)
  return UOp.sink(*sinks, arg=info)
