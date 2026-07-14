from tinygrad import Tensor
from tinygrad.uop.ops import UOp, AxisType, KernelInfo

def zero_kernel(out: UOp) -> UOp:
  i = UOp.range(out.max_numel(), 0, AxisType.LOOP)
  store = out.flatten().index(i).store(0)
  return store.end(i).sink(arg=KernelInfo(name="tilegrad_zero", opts_to_apply=()))

if __name__ == "__main__":
  out = Tensor.empty(16)
  out = out.custom_kernel(fxn=zero_kernel)[0].realize()
  print(out.tolist())
