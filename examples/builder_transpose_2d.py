from tinygrad import Tensor
from tilegrad.builder import KernelBuilder
from tilegrad.ir import Add, FloorDiv, Mod, Mul
from tilegrad.lowerer import lower_kernel

def builder_transpose_2d_kernel(out, inp):
  k = KernelBuilder("tilegrad_builder_transpose_2d", ("out", "inp"))
  row = FloorDiv("i", 3)
  col = Mod("i", 3)
  out_idx = Add(Mul(col, 2), row)
  with k.range("i", 6): k.store("out", out_idx, k.load("inp", "i"))
  return lower_kernel(k.build(), out, inp)

if __name__ == "__main__":
  inp = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  out = Tensor.empty(6)
  out = out.custom_kernel(inp, fxn=builder_transpose_2d_kernel)[0].realize()
  print(out.tolist())