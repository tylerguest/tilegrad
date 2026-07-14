from tinygrad import Tensor
from tilegrad.builder import KernelBuilder
from tilegrad.lowerer import lower_kernel

def builder_shared_copy_kernel(out_uop, inp_uop):
  k = KernelBuilder("shared_copy", ("out", "inp"))
  out = k.buffer("out", shape=(4,), dtype="float32")
  inp = k.buffer("inp", shape=(4,), dtype="float32")
  smem = k.shared("smem", shape=(4,), dtype="float32")

  k.copy(inp.tile(), smem.tile())
  k.barrier()
  k.copy(smem.tile(), out.tile())

  return lower_kernel(k.build(), out_uop, inp_uop)

if __name__ == "__main__":
  inp = Tensor([1.0, 2.0, 3.0, 4.0])
  out = Tensor.empty(4)
  out = out.custom_kernel(inp, fxn=builder_shared_copy_kernel)[0].realize()
  print(out.tolist())