from tinygrad import Tensor

from tilegrad.ir import Alloc, Arg, Barrier, Kernel, Load, Range, Store
from tilegrad.lowerer import lower_kernel


def shared_copy_kernel(out, inp):
  ir = Kernel(
    "tilegrad_ir_shared_copy",
    (Arg("out"), Arg("inp")),
    (
      Alloc("smem", "out.numel", "float32", "shared"),
      Range("i", "out.numel", (Store("smem", "i", Load("inp", "i")),)),
      Barrier(),
      Range("j", "out.numel", (Store("out", "j", Load("smem", "j")),)),
    ),
  )
  return lower_kernel(ir, out, inp)


if __name__ == "__main__":
  inp = Tensor([1.0, 2.0, 3.0, 4.0])
  out = Tensor.empty(4)
  out = out.custom_kernel(inp, fxn=shared_copy_kernel)[0].realize()
  print(out.tolist())
