from tinygrad import Tensor
from tilegrad import KernelBuilder, run

k = KernelBuilder("tilegrad_builder_pipelined_copy", ("out", "inp"))
out = k.buffer("out")
inp = k.buffer("inp")

with k.pipelined("i", 4, stages=2) as i:
  out[i] = inp[i]

if __name__ == "__main__":
  inp_t = Tensor([1.0, 2.0, 3.0, 4.0])
  out_t = Tensor.empty(4)
  print(run(k, out_t, inp_t).tolist())
