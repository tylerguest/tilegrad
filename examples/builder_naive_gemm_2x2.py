from tinygrad import Tensor

from tilegrad.builder import KernelBuilder
from tilegrad.ir import Add, Index2D, Mul
from tilegrad.lowerer import lower_kernel


def builder_naive_gemm_2x2_kernel(out, a, b):
  k = KernelBuilder("tilegrad_builder_naive_gemm_2x2", ("out", "a", "b"))
  with k.range("i", 2):
    with k.range("j", 2):
      k.set("out", Index2D("i", "j", 2), 0)
      with k.range("k", 2, axis="reduce"):
        k.set(
          "out",
          Index2D("i", "j", 2),
          Add(
            k.load("out", Index2D("i", "j", 2)),
            Mul(
              k.load("a", Index2D("i", "k", 2)),
              k.load("b", Index2D("k", "j", 2)),
            ),
          ),
        )
  return lower_kernel(k.build(), out, a, b)


if __name__ == "__main__":
  a = Tensor([1.0, 2.0, 3.0, 4.0])
  b = Tensor([5.0, 6.0, 7.0, 8.0])
  out = Tensor.empty(4)
  out = out.custom_kernel(a, b, fxn=builder_naive_gemm_2x2_kernel)[0].realize()
  print(out.tolist())
