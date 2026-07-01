from tinygrad import Tensor
from tilegrad import KernelBuilder, run
from tilegrad.ir import Add

k = KernelBuilder("tilegrad_builder_row_sum", ("out", "inp"))
k.alloc("acc", 1, "float32", "register")
inp = k.buffer("inp", shape=(2, 3))
out = k.buffer("out")
acc = k.buffer("acc")
with k.range("i", 2) as i:
  acc[0] = 0
  with k.range("j", 3, axis="reduce") as j:
    acc[0] = Add(acc[0], inp[i, j])
  out[i] = acc[0]

if __name__ == "__main__":
  inp_t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  out_t = Tensor.empty(2)
  print(run(k, out_t, inp_t).tolist())