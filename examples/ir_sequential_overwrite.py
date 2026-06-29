from tinygrad import Tensor
from tilegrad.ir import Add, Arg, Kernel, Load, Range, Store
from tilegrad.lowerer import lower_kernel

def sequential_overwrite_kernel(out, inp):
  ir = Kernel(
    "tilegrad_ir_sequential_overwrite",
    (Arg("out"), Arg("inp")),
    (
      Range("i", 4, (Store("out", "i", 0),)),
      Range("j", 3, (Store("out", Add("j", 1), Load("inp", "j")),)),
    ),
  )
  return lower_kernel(ir, out, inp)

if __name__ == "__main__":
  inp = Tensor([1.0, 2.0, 3.0])
  out = Tensor.empty(4)
  out = out.custom_kernel(inp, fxn=sequential_overwrite_kernel)[0].realize()
  print(out.tolist())