from tinygrad import Tensor
from tilegrad.ir import Arg, Kernel, Range, Store
from tilegrad.lowerer import lower_kernel

def zero_kernel(out):
  ir = Kernel("tilegrad_ir_zero", (Arg("out"),), (Range("i", "out.numel", (Store("out", "i", 0),)),))
  return lower_kernel(ir, out)

if __name__ == "__main__":
  out = Tensor.empty(16)
  out = out.custom_kernel(fxn=zero_kernel)[0].realize()
  print(out.tolist())
