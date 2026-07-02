from tinygrad import Tensor
from tilegrad import KernelBuilder, run

k = KernelBuilder("tilegrad_builder_grid_threads_copy", ("out", "inp"))
out = k.buffer("out")
inp = k.buffer("inp")

with k.grid(2) as block:
  with k.threads(4) as tid:
    i = block * 4 + tid
    out[i] = inp[i]

if __name__ == "__main__":
  inp_t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
  out_t = Tensor.empty(8)
  print(run(k, out_t, inp_t).tolist())