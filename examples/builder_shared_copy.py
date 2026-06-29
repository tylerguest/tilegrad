from tinygrad import Tensor
from tilegrad.builder import KernelBuilder
from tilegrad.lowerer import lower_kernel

def builder_shared_copy_kernel(out, inp):
  k = KernelBuilder("tilegrad_builder_shared_copy", ("out", "inp"))
  k.alloc("smem", "out.numel", "float32")
  with k.range("i", "out.numel"): k.store("smem", "i", k.load("inp", "i"))
  k.barrier()
  with k.range("j", "out.numel"): k.store("out", "j", k.load("smem", "j"))
  return lower_kernel(k.build(), out, inp)

if __name__ == "__main__":
  inp = Tensor([1.0, 2.0, 3.0, 4.0])
  out = Tensor.empty(4)
  out = out.custom_kernel(inp, fxn=builder_shared_copy_kernel)[0].realize()
  print(out.tolist())