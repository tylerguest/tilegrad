from tinygrad import Tensor
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, UOp


def shared_copy_kernel(out: UOp, inp: UOp) -> UOp:
  n = out.max_numel()
  smem = UOp.placeholder((n,), out.dtype.base, slot=0, addrspace=AddrSpace.LOCAL)

  i = UOp.range(n, 0, AxisType.LOOP)
  src = inp.flatten().index(i, ptr=True).load()
  smem_store = smem.index(i, ptr=True).store(src).end(i)

  bar = smem_store.barrier()

  j = UOp.range(n, 1, AxisType.LOOP)
  val = smem.after(bar).index(j, ptr=True).load()
  out_store = out.flatten().index(j, ptr=True).store(val).end(j)

  return out_store.sink(arg=KernelInfo(name="tilegrad_raw_shared_copy"))


if __name__ == "__main__":
  inp = Tensor([1.0, 2.0, 3.0, 4.0])
  out = Tensor.empty(4)
  out = out.custom_kernel(inp, fxn=shared_copy_kernel)[0].realize()
  print(out.tolist())
