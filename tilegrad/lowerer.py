from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

from tilegrad.ir import Alloc, Barrier, Load, Range


def lower_shape(shape, env):
  if isinstance(shape, int):
    return shape
  if shape.endswith(".numel"):
    return env[shape[:-6]].max_numel()
  raise NotImplementedError(shape)


def lower_value(v, env, i):
  return env[v.buffer].flatten().index(i).load() if isinstance(v, Load) else v


def lower_range(op, env, effects):
  s = op.body[0]
  i = UOp.range(lower_shape(op.extent, env), len(effects), AxisType.LOOP)
  val = lower_value(s.value, env, i)
  effects.append(env[s.buffer].flatten().index(i, ptr=True).store(val).end(i))


def lower_alloc(op, env):
  if op.space != "shared":
    raise NotImplementedError(op.space)

  ref = next(iter(env.values()))
  env[op.name] = UOp.placeholder((lower_shape(op.shape, env),), ref.dtype.base, slot=0, addrspace=AddrSpace.LOCAL)


def lower_barrier(env, effects):
  bar = effects[-1].barrier()
  effects.append(bar)

  for name, buf in tuple(env.items()):
    if buf.addrspace is AddrSpace.LOCAL:
      env[name] = buf.after(bar)


def lower_kernel(kernel, *args: UOp) -> UOp:
  env = {arg.name: uop for arg, uop in zip(kernel.args, args)}
  effects = []

  for op in kernel.body:
    if isinstance(op, Alloc):
      lower_alloc(op, env)
    elif isinstance(op, Range):
      lower_range(op, env, effects)
    elif isinstance(op, Barrier):
      lower_barrier(env, effects)
    else:
      raise NotImplementedError(type(op).__name__)

  return effects[-1].sink(arg=KernelInfo(name=kernel.name))
