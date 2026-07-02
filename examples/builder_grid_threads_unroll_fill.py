from tinygrad import Tensor
from tilegrad import KernelBuilder, run

k = KernelBuilder("tilegrad_grid_threads_unroll_fill", ("out", "inp"))
out = k.buffer("out")
inp = k.buffer("inp")

with k.grid(2) as block:
  with k.threads(2) as tid:
    base = block * 4 + tid * 2
    with k.range("u", 2, axis="unroll") as u:
      i = base + u 
      out[i] = inp[i]

if __name__ == "__main__":
  inp_t = Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
  out_t = Tensor.empty(8)
  print(run(k, out_t, inp_t).tolist())