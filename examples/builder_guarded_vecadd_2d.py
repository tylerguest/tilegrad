from tinygrad import Tensor
from tilegrad import KernelBuilder, lower_kernel

def builder_guarded_vecadd_2d_kernel(out, a, b):
  k = KernelBuilder("tilegrad_builder_guarded_vecadd_2d", ("out", "a", "b"))
  out_ref = k.buffer("out", shape=(3, 2))
  a_ref = k.buffer("a", shape=(3, 2))
  b_ref = k.buffer("b", shape=(3, 2))
  with k.parallel(4, 4) as (i, j): k.store_if((i < 3) & (j < 2), out_ref, (i, j), a_ref[i, j] + b_ref[i, j])
  return lower_kernel(k.build(), out, a, b)

if __name__ == "__main__":
  a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  b = Tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
  out = Tensor.empty(6)
  out = out.custom_kernel(a, b, fxn=builder_guarded_vecadd_2d_kernel)[0].realize()
  print(out.tolist())