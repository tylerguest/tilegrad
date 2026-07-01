from tinygrad import Tensor
from tilegrad import KernelBuilder, lower_kernel

def builder_parallel_vecadd_2d_kernel(out, a, b):
  k = KernelBuilder("tilegrad_builder_parallel_vecadd_2d", ("out", "a", "b"))
  out_ref = k.buffer("out", shape=(2, 2))
  a_ref = k.buffer("a", shape=(2, 2))
  b_ref = k.buffer("b", shape=(2, 2))
  with k.parallel(2, 2) as (i, j): out_ref[i, j] = a_ref[i, j] + b_ref[i, j]
  return lower_kernel(k.build(), out, a, b)

if __name__ == "__main__":
  a = Tensor([1.0, 2.0, 3.0, 4.0])
  b = Tensor([10.0, 20.0, 30.0, 40.0])
  out = Tensor.empty(4)
  out = out.custom_kernel(a, b, fxn=builder_parallel_vecadd_2d_kernel)[0].realize()
  print(out.tolist())