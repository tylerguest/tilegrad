from tinygrad import Tensor

from tilegrad.ir import Add, Arg, Index2D, Kernel, Load, Mul, Range, Set
from tilegrad.lowerer import lower_kernel


def naive_gemm_2x2_kernel(out, a, b):
  ir = Kernel(
    "tilegrad_ir_naive_gemm_2x2",
    (Arg("out"), Arg("a"), Arg("b")),
    (
      Range("i", 2, (
        Range("j", 2, (
          Set("out", Index2D("i", "j", 2), 0),
          Range("k", 2, (
            Set(
              "out",
              Index2D("i", "j", 2),
              Add(
                Load("out", Index2D("i", "j", 2)),
                Mul(
                  Load("a", Index2D("i", "k", 2)),
                  Load("b", Index2D("k", "j", 2)),
                ),
              ),
            ),
          ), axis="reduce"),
        )),
      )),
    ),
  )
  return lower_kernel(ir, out, a, b)


if __name__ == "__main__":
  a = Tensor([1.0, 2.0, 3.0, 4.0])
  b = Tensor([5.0, 6.0, 7.0, 8.0])
  out = Tensor.empty(4)
  out = out.custom_kernel(a, b, fxn=naive_gemm_2x2_kernel)[0].realize()
  print(out.tolist())
