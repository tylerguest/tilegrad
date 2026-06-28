from tinygrad import Tensor

from tilegrad.ir import Arg, Kernel, Load, Range, Store
from tilegrad.lowerer import lower_kernel


def copy_kernel(out, inp):
  ir = Kernel("tilegrad_ir_copy", (Arg("out"), Arg("inp")), (Range("i", "out.numel", (Store("out", "i", Load("inp", "i")),)),))
  return lower_kernel(ir, out, inp)


if __name__ == "__main__":
  inp = Tensor([1.0, 2.0, 3.0, 4.0])
  out = Tensor.empty(4)
  out = out.custom_kernel(inp, fxn=copy_kernel)[0].realize()
  print(out.tolist())
