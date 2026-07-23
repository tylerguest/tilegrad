class TinygradCompatibilityError(RuntimeError):
  pass


try:
  from tinygrad import Tensor, TinyJit, function
  from tinygrad.dtype import AddrSpace, Invalid, dtypes
  from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
  from tinygrad.uop.render import print_uops
except ImportError as exc:
  raise TinygradCompatibilityError(f"tinygrad is missing a TileGrad-required API: {exc}") from exc


def check_tinygrad_compatibility():
  required = (
    (Tensor, "custom_kernel", "Tensor.custom_kernel"),
    (AxisType, "GLOBAL", "AxisType.GLOBAL"),
    (AxisType, "LOCAL", "AxisType.LOCAL"),
    (AxisType, "LOOP", "AxisType.LOOP"),
    (AxisType, "REDUCE", "AxisType.REDUCE"),
    (AxisType, "UNROLL", "AxisType.UNROLL"),
    (AddrSpace, "GLOBAL", "AddrSpace.GLOBAL"),
    (AddrSpace, "LOCAL", "AddrSpace.LOCAL"),
    (AddrSpace, "REG", "AddrSpace.REG"),
    (KernelInfo, "__dataclass_fields__", "KernelInfo dataclass fields"),
    (UOp, "placeholder", "UOp.placeholder"),
    (UOp, "range", "UOp.range"),
    (UOp, "group", "UOp.group"),
    (UOp, "sink", "UOp.sink"),
  )
  missing = [label for owner, name, label in required if not hasattr(owner, name)]
  if not callable(getattr(Tensor, "custom_kernel", None)) and "Tensor.custom_kernel" not in missing:
    missing.append("Tensor.custom_kernel")
  if not callable(TinyJit):
    missing.append("TinyJit")
  if not callable(function):
    missing.append("function")

  fields = getattr(KernelInfo, "__dataclass_fields__", {})
  for name in ("name", "opts_to_apply"):
    if name not in fields: missing.append(f"KernelInfo.{name}")

  if missing:
    raise TinygradCompatibilityError("tinygrad is missing TileGrad-required APIs: " + ", ".join(missing))

  try:
    info = KernelInfo(name="tilegrad_compatibility_probe", opts_to_apply=())
  except (TypeError, ValueError) as exc:
    raise TinygradCompatibilityError(f"tinygrad KernelInfo is incompatible with TileGrad: {exc}") from exc
  if info.opts_to_apply != ():
    raise TinygradCompatibilityError("tinygrad KernelInfo does not preserve opts_to_apply=()")


check_tinygrad_compatibility()


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
