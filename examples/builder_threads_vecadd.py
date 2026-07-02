from tinygrad import Tensor
from tilegrad import KernelBuilder, run

k = KernelBuilder("tilegrad_builder_threads_vecadd", ("out", "a", "b"))
out = k.buffer("out")
a = k.buffer("a")
b = k.buffer("b")

with k.threads(4) as tid: out[tid] = a[tid] + b[tid]

if __name__ == "__main__":
  a_t = Tensor([1.0, 2.0, 3.0, 4.0])
  b_t = Tensor([10.0, 20.0, 30.0, 40.0])
  out_t = Tensor.empty(4)
  print(run(k, out_t, a_t, b_t).tolist())