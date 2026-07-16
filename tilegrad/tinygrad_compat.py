from tinygrad.dtype import AddrSpace, Invalid, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp


AXIS_TYPES = {
  "loop": AxisType.LOOP,
  "reduce": AxisType.REDUCE,
  "global": AxisType.GLOBAL,
  "local": AxisType.LOCAL,
  "unroll": AxisType.UNROLL,
}

def dtype_from_name(dtype):
  if not isinstance(dtype, str): return dtype
  if not hasattr(dtypes, dtype): raise NotImplementedError(dtype)
  return getattr(dtypes, dtype)

def scalar_dtype(dtype): return dtype.scalar()

def index_const(idx): return idx if isinstance(idx, UOp) else UOp.const(dtypes.weakint, idx)

def placeholder(shape, dtype, slot, addrspace=AddrSpace.GLOBAL):
  return UOp.placeholder(shape, dtype, slot=slot, addrspace=addrspace)

def range_uop(extent, slot, axis_type): return UOp.range(extent, slot, axis_type)

def index_uop(buf, idx): return buf.index(index_const(idx))

def load_uop(buf, idx): return index_uop(buf, idx).load()

def group_uops(*uops): return UOp.group(*uops)

def barrier_uops(*uops):
  if not uops: raise ValueError("barrier requires at least one source")
  return uops[0].barrier(*uops[1:])

def kernel_info(name): return KernelInfo(name=name, opts_to_apply=())

def sink_uops(*uops, name): return UOp.sink(*uops, arg=kernel_info(name))
