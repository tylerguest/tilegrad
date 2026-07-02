from tinygrad.dtype import AddrSpace, Invalid, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from tilegrad.fragments import expand_fragments
from tilegrad.ir import Add, Alloc, And, Barrier, BinaryExpr, Const, Eq, FloorDiv, Ge, Gt, Index2D, Le, Load, LoadIf, Lt, Mod, Mul, Ne, Not, Or, Range, Set, SetIf, Store, StoreIf, Sub, Var
from tilegrad.unroll import unroll_register_tiles
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

def _dedup_ranges(*ranges):
  out = []
  for r in ranges:
    if r is not None and r not in out: out.append(r)
  return tuple(out)

def _expr_load_buffers(expr):
  if isinstance(expr, Load): return {expr.buffer}
  if isinstance(expr, LoadIf): return {expr.buffer} | _expr_load_buffers(expr.cond) | _expr_load_buffers(expr.index)
  if isinstance(expr, Index2D): return _expr_load_buffers(expr.row) | _expr_load_buffers(expr.col) | _expr_load_buffers(expr.stride)
  if isinstance(expr, BinaryExpr): return _expr_load_buffers(expr.lhs) | _expr_load_buffers(expr.rhs)
  if isinstance(expr, Not): return _expr_load_buffers(expr.x)
  return set()

def _after_deps(buf): return buf.src[1:] if isinstance(buf, UOp) and buf.op.name == "AFTER" else ()

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
  safe_idx = cond.where(idx, idx.const_like(0)) if isinstance(cond, UOp) else idx if cond else idx.const_like(0)
  val = buf.index(safe_idx).load()
  return cond.where(val, val.const_like(0)) if isinstance(cond, UOp) else val if cond else val.const_like(0)

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

def _store_key(stmt):
  return repr(stmt.index)

def _needs_store_order(stmt, active_ranges, store_state):
  prev = store_state.get(stmt.buffer)
  if prev is None: return False
  prev_ranges, prev_key = prev
  return prev_ranges != active_ranges or prev_key == _store_key(stmt)

def _strip_after(buf):
  while isinstance(buf, UOp) and buf.op.name == "AFTER": buf = buf.src[0]
  return buf

def lower_store_if(stmt, env, effects, sink_effects, buffer_effects, pending_shared, store_state, shared_read_effects, indices, active_ranges):
  cond = lower_expr(stmt.cond, env, indices)
  idx = lower_expr(stmt.index, env, indices)
  val = lower_expr(stmt.value, env, indices)
  base = env[stmt.buffer]
  needs_order = _needs_store_order(stmt, active_ranges, store_state)
  if base.addrspace is AddrSpace.LOCAL: 
    buf = base.flatten()
    for dep in shared_read_effects.get(stmt.buffer, ()): buf = buf.after(dep)
  else: buf = (base if needs_order else _strip_after(base)).flatten()
  if needs_order: buf = buf.after(buffer_effects[stmt.buffer])
  if isinstance(val, UOp) and val.dtype != base.dtype.base: val = val.cast(base.dtype.base)
  idx = lower_index(idx)
  guarded_idx = cond.where(idx, idx.const_like(Invalid)) if isinstance(cond, UOp) else idx if cond else idx.const_like(Invalid)
  effect = buf.index(guarded_idx, ptr=True).store(val)
  if base.addrspace is AddrSpace.LOCAL:
    buffer_effects[stmt.buffer] = effect
    pending_shared.append((effect, active_ranges))
  else:
    ended = effect.end(*active_ranges)
    buffer_effects[stmt.buffer] = ended
    store_state[stmt.buffer] = (active_ranges, _store_key(stmt))
    effects.append(ended)
    sink_effects.append(ended)
    env[stmt.buffer] = base.after(ended)

def lower_store(stmt, env, effects, sink_effects, buffer_effects, pending_shared, store_state, shared_read_effects, indices, active_ranges):
  idx = lower_expr(stmt.index, env, indices)
  val = lower_expr(stmt.value, env, indices)
  base = env[stmt.buffer]
  needs_order = _needs_store_order(stmt, active_ranges, store_state)
  if base.addrspace is AddrSpace.LOCAL: 
    buf = base.flatten()
    for dep in shared_read_effects.get(stmt.buffer, ()): buf = buf.after(dep)
  else: buf = (base if needs_order else _strip_after(base)).flatten()
  if needs_order: buf = buf.after(buffer_effects[stmt.buffer])
  if isinstance(val, UOp) and val.dtype != base.dtype.base: val = val.cast(base.dtype.base)
  effect = buf.index(lower_index(idx), ptr=True).store(val)
  if base.addrspace is AddrSpace.LOCAL:
    buffer_effects[stmt.buffer] = effect
    pending_shared.append((effect, active_ranges))
  else:
    ended = effect.end(*active_ranges)
    buffer_effects[stmt.buffer] = ended
    store_state[stmt.buffer] = (active_ranges, _store_key(stmt))
    effects.append(ended)
    sink_effects.append(ended)
    env[stmt.buffer] = base.after(ended)

def lower_set(stmt, env, updated_buffers, local_updated, shared_read_effects, indices, active_ranges, axis, register_scopes, cond=None):
  idx = lower_expr(stmt.index, env, indices)
  recurrence_range = active_ranges[-1] if axis == "reduce" and active_ranges else None
  buf = env[stmt.buffer]
  is_register = buf.addrspace is AddrSpace.REG
  recurrence_uop = None
  register_end_ranges = ()
  if is_register:
    desired_scope = active_ranges[:-1] if axis == "reduce" else active_ranges
    current_scope = register_scopes.get(stmt.buffer, ())
    if desired_scope[:len(current_scope)] != current_scope:
      raise RuntimeError(f"register scope mismatch for {stmt.buffer}: current={current_scope} desired={desired_scope}")
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
  if axis == "reduce":
    ends = _dedup_ranges(recurrence_range, *register_end_ranges)
  elif cond is not None:
    ends = register_end_ranges
  else:
    ends = ()
  next_buf = target.set(val, end=ends) if ends else target.set(val)
  if is_register:
    deps = _after_deps(next_buf)
    if deps:
      for name in _expr_load_buffers(stmt.value):
        if name in env and env[name].addrspace is AddrSpace.LOCAL:
          shared_read_effects[name] = _dedup_ranges(*shared_read_effects.get(name, ()), *deps)
  if buf.addrspace is not AddrSpace.REG and isinstance(val, UOp):
    leaked_ranges = [r for r in val.ranges if r not in active_ranges]
    if leaked_ranges: next_buf = next_buf.end(*leaked_ranges)
  env[stmt.buffer] = next_buf
  updated_buffers.add(stmt.buffer)
  local_updated.add(stmt.buffer)

def lower_range(op, env, effects, sink_effects, buffer_effects, pending_shared, store_state, shared_read_effects, updated_buffers, indices, range_slots, register_scopes, active_ranges=()):
  axis_type = AxisType.REDUCE if op.axis == "reduce" else AxisType.LOOP
  i = UOp.range(lower_shape(op.extent, env), range_slots[0], axis_type)
  range_slots[0] += 1
  indices = indices | {op.name: i}
  active_ranges = active_ranges + (i,)
  local_updated = set()
  for stmt in op.body:
    if isinstance(stmt, Range): local_updated |= lower_range(stmt, env, effects, sink_effects, buffer_effects, pending_shared, 
                                                             store_state, shared_read_effects, updated_buffers, indices, range_slots, register_scopes, active_ranges)
    elif isinstance(stmt, Store): lower_store(stmt, env, effects, sink_effects, buffer_effects, pending_shared, store_state, shared_read_effects, indices, active_ranges)
    elif isinstance(stmt, StoreIf): lower_store_if(stmt, env, effects, sink_effects, buffer_effects, pending_shared, store_state, shared_read_effects, indices, active_ranges)
    elif isinstance(stmt, SetIf): lower_set(stmt, env, updated_buffers, local_updated, shared_read_effects, indices, active_ranges, op.axis, register_scopes, cond=stmt.cond)
    elif isinstance(stmt, Set): lower_set(stmt, env, updated_buffers, local_updated, shared_read_effects, indices, active_ranges, op.axis, register_scopes)
    elif isinstance(stmt, Barrier): lower_barrier(env, effects, buffer_effects, pending_shared, active_ranges)
    else: raise NotImplementedError(type(stmt).__name__)
  if op.axis == "loop":
    for name in local_updated:
      if name not in env: continue
      buf = env[name]
      if buf.addrspace is AddrSpace.REG:
        scope = register_scopes.get(name, ())
        if scope and scope[-1] is i:
          if i in buf.ranges:
            target = buf.flatten()[0]
            env[name] = target.set(target, end=i)
          register_scopes[name] = scope[:-1]
      else:
        env[name] = buf.end(i)
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

def _group_independent_sink_effects(sink_effects):
  out = []
  used = set()
  for i, effect in enumerate(sink_effects):
    if i in used: continue
    if effect.op.name != "END":
      out.append(effect)
      continue
    ranges = effect.src[1:]
    group = [effect]
    for j in range(i + 1, len(sink_effects)):
      other = sink_effects[j]
      if other.op.name != "END" or other.src[1:] != ranges: continue
      if any(other in x.backward_slice_with_self or x in other.backward_slice_with_self for x in group): continue
      group.append(other)
      used.add(j)
    if len(group) == 1:
      out.append(effect)
    else:
      out.append(UOp.group(*(x.src[0] for x in group)).end(*ranges))
  return out

def lower_kernel(kernel, *args: UOp) -> UOp:
  kernel = expand_fragments(kernel)
  kernel = unroll_register_tiles(kernel)
  validate_kernel(kernel)
  if len(args) != len(kernel.args): raise ValueError(f"expected {len(kernel.args)} args, got {len(args)}")
  env = {arg.name: uop for arg, uop in zip(kernel.args, args)}
  effects = []
  sink_effects = []
  buffer_effects = {}
  store_state = {}
  shared_read_effects = {}
  pending_shared = []
  updated_buffers = set()
  indices = {}
  shared_slots = [0]
  register_slots = [0]
  range_slots = [0]
  register_scopes = {}
  for op in kernel.body:
    if isinstance(op, Alloc): lower_alloc(op, env, shared_slots, register_slots)
    elif isinstance(op, SetIf): lower_set(op, env, updated_buffers, set(), shared_read_effects, indices, (), "loop", register_scopes, cond=op.cond)
    elif isinstance(op, Set): lower_set(op, env, updated_buffers, set(), shared_read_effects, indices, (), "loop", register_scopes)
    elif isinstance(op, StoreIf): lower_store_if(op, env, effects, sink_effects, buffer_effects, pending_shared, store_state, shared_read_effects,indices, ())
    elif isinstance(op, Store): lower_store(op, env, effects, sink_effects, buffer_effects, pending_shared, store_state, shared_read_effects, indices, ())
    elif isinstance(op, Range): lower_range(op, env, effects, sink_effects, buffer_effects, pending_shared, store_state, shared_read_effects, updated_buffers, indices, range_slots, register_scopes)
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
  sinks = _group_independent_sink_effects(sink_effects)
  sinks += [env[arg.name] for arg in kernel.args if arg.name in updated_buffers]
  if not sinks: raise ValueError("kernel must produce at least one effect")
  info = KernelInfo(name=kernel.name, opts_to_apply=()) if updated_buffers else KernelInfo(name=kernel.name)
  return UOp.sink(*sinks, arg=info)
