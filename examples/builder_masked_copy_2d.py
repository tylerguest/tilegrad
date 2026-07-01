from tinygrad import Tensor
from tilegrad import KernelBuilder, run

k = KernelBuilder("tilegrad_builder_masked_copy_2d", ("out", "inp"))
out = k.buffer("out", shape=(4, 4))
inp = k.buffer("inp", shape=(3, 2))

with k.parallel(4, 4) as (i, j):
  out[i, j] = k.load_if((i < 3) & (j < 2), inp, (i, j))

if __name__ == "__main__":
  inp_t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  out_t = Tensor.empty(16)
  print(run(k, out_t, inp_t).tolist())
