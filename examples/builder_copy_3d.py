from tinygrad import Tensor
from tilegrad import KernelBuilder, run

k = KernelBuilder("tilegrad_builder_copy_3d", ("out", "inp"))
out = k.buffer("out", shape=(2, 2, 3))
inp = k.buffer("inp", shape=(2, 2, 3))

with k.range("b", 2) as b:
  with k.range("i", 2) as i:
    with k.range("j", 3) as j:
      out[b, i, j] = inp[b, i, j]

if __name__ == "__main__":
  inp_t = Tensor([
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    10.0, 11.0, 12.0,
  ])
  out_t = Tensor.empty(12)
  print(run(k, out_t, inp_t).tolist())
