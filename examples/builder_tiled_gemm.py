from tinygrad import Tensor
from tilegrad.builder import KernelBuilder
from tilegrad.ir import Add, Index2D, Mul
from tilegrad.lowerer import lower_kernel

def gemm_kernel(out, a, b):
  k = KernelBuilder("builder_tiled_gemm", ("out", "a", "b"))
  k.alloc("as", 3, "float32")
  k.alloc("bs", 3, "float32")
  k.alloc("acc", 1, "float32", "register")

  with k.range("i", 2):
    with k.range("j", 2):
      k.set("acc", 0, 0)
      with k.range("ko", 2):
        k.copy("a", "as", shape=(1, 3), stride=6, src_row_off="i", src_col_off=Mul("ko", 3))
        k.copy("b", "bs", shape=(3,), stride=2, src_row_off=Mul("ko", 3), src_col_off="j")
        k.barrier()
        with k.range("kk", 3, axis="reduce"):
          k.set(
            "acc",
            0,
            Add(
              k.load("acc", 0),
              Mul(k.load("as", "kk"), k.load("bs", "kk")),
            ),
          )
      k.set("out", Index2D("i", "j", 2), k.load("acc", 0))
  return lower_kernel(k.build(), out, a, b)

if __name__ == "__main__":
  a = Tensor([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
  ])
  b = Tensor([
    13.0, 14.0,
    15.0, 16.0,
    17.0, 18.0,
    19.0, 20.0,
    21.0, 22.0,
    23.0, 24.0,
  ])
  out = Tensor.empty(4)
  out = out.custom_kernel(a, b, fxn=gemm_kernel)[0].realize()
  print(out.tolist())