from tinygrad import Tensor
from tilegrad.ir import Add, Alloc, Arg, Barrier, Kernel, Load, Mul, Range, Store
from tilegrad.lowerer import lower_kernel

def two_shared_allocs_kernel(out, inp):
  ir = Kernel(
    "tilegrad_ir_two_shared_allocs",
    (Arg("out"), Arg("inp")),
    (
      Alloc("a", 2, "float32", "shared"),
      Alloc("b", 2, "float32", "shared"),
      Range("i", 2, (
        Store("a", "i", Load("inp", "i")),
        Store("b", "i", Add(Load("inp", "i"), 10)),
      )),
      Barrier(),
      Range("j", 2, (
        Store("out", Mul("j", 2), Load("a", "j")),
        Store("out", Add(Mul("j", 2), 1), Load("b", "j"))
      )),
    ),
  )
  return lower_kernel(ir, out, inp)

if __name__ == "__main__":
  inp = Tensor([1.0, 2.0])
  out = Tensor.empty(4)
  out = out.custom_kernel(inp, fxn=two_shared_allocs_kernel)[0].realize()
  print(out.tolist())