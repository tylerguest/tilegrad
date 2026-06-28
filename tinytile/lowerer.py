from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from tinytile.ir import Load

def lower_value(v, env, i):
  return env[v.buffer].flatten().index(i).load() if isinstance(v, Load) else v

def lower_kernel(kernel, *args: UOp) -> UOp:
  range_op = kernel.body[0]
  store_op = range_op.body[0]

  env = {arg.name: uop for arg, uop in zip(kernel.args, args)}
  extent = env[store_op.buffer].max_numel() if range_op.extent == f"{store_op.buffer}.numel" else range_op.extent

  i = UOp.range(extent, 0, AxisType.LOOP)
  store = env[store_op.buffer].flatten().index(i, ptr=True).store(lower_value(store_op.value, env, i))

  return store.end(i).sink(arg=KernelInfo(name=kernel.name))
