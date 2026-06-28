from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from tilegrad.ir import Add, Alloc, Barrier, Const, Load, Mul, Range, Store

def lower_shape(shape, env):
  if isinstance(shape, int): return shape
  if shape.endswith(".numel"): return env[shape[:-6]].max_numel()
  raise NotImplementedError(shape)

def lower_expr(expr, env, indices):
  if isinstance(expr, (int, float)): return expr
  if isinstance(expr, str):
    if expr in indices: return indices[expr]
    raise NotImplementedError(expr)
  if isinstance(expr, Const): return expr.value
  if isinstance(expr, Add): return lower_expr(expr.lhs, env, indices) + lower_expr(expr.rhs, env, indices)
  if isinstance(expr, Mul): return lower_expr(expr.lhs, env, indices) * lower_expr(expr.rhs, env, indices)
  if isinstance(expr, Load):
    idx = lower_expr(expr.index, env, indices)
    return env[expr.buffer].flatten().index(idx).load()
  raise NotImplementedError(type(expr).__name__)

def lower_range(op, env, effects, indices):
  i = UOp.range(lower_shape(op.extent, env), len(effects), AxisType.LOOP)
  indices = indices | {op.name: i}
  for stmt in op.body:
    if not isinstance(stmt, Store): raise NotImplementedError(type(stmt).__name__)
    idx = lower_expr(stmt.index, env, indices)
    val = lower_expr(stmt.value, env, indices)
    effects.append(env[stmt.buffer].flatten().index(idx, ptr=True).store(val).end(i))

def lower_alloc(op, env):
  if op.space != "shared": raise NotImplementedError(op.space)
  ref = next(iter(env.values()))
  env[op.name] = UOp.placeholder((lower_shape(op.shape, env),), ref.dtype.base, slot=0, addrspace=AddrSpace.LOCAL)

def lower_barrier(env, effects):
  bar = effects[-1].barrier()
  effects.append(bar)
  for name, buf in tuple(env.items()):
    if buf.addrspace is AddrSpace.LOCAL: env[name] = buf.after(bar)

def lower_kernel(kernel, *args: UOp) -> UOp:
  env = {arg.name: uop for arg, uop in zip(kernel.args, args)}
  effects = []
  indices = {}
  for op in kernel.body:
    if isinstance(op, Alloc): lower_alloc(op, env)
    elif isinstance(op, Range): lower_range(op, env, effects, indices)
    elif isinstance(op, Barrier): lower_barrier(env, effects)
    else: raise NotImplementedError(type(op).__name__)
  return UOp.sink(*effects, arg=KernelInfo(name=kernel.name))
