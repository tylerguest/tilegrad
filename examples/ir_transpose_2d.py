from tinygrad import Tensor
from tilegrad.ir import Add, Arg, FloorDiv, Kernel, Load, Mod, Mul, Range, Store
from tilegrad.lowerer import lower_kernel

def transpose_2d_kernel(out, inp):
  row = FloorDiv("i", 3)
  col = Mod("i", 3)
  out_idx = Add(Mul(col, 2), row)
  ir = Kernel(
    "tilegrad_ir_transpose_2d",
    (Arg("out"), Arg("inp")),
    (Range("i", 6, (Store("out", out_idx, Load("inp", "i")),)),),
  )
  return lower_kernel(ir, out, inp)

if __name__ == "__main__":
  inp = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  out = Tensor.empty(6)
  out = out.custom_kernel(inp, fxn=transpose_2d_kernel)[0].realize()
  print(out.tolist())