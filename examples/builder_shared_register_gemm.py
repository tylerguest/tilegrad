from tinygrad import Tensor
from tilegrad.builder import KernelBuilder
from tilegrad.lowerer import lower_kernel
from tilegrad.ir import Add, Index2D, Mul

def shared_register_gemm_kernel(out, a, b):
  k = KernelBuilder("tilegrad_shared_register_gemm", ("out", "a", "b"))
  k.alloc("as", 4, "float32", space="shared")
  k.alloc("bs", 4, "float32", space="shared")
  k.alloc("acc", 1, "float32", space="register")

  with k.range("i", 2):
    with k.range("j", 2):
      k.store("as", Index2D("i", "j", 2), k.load("a", Index2D("i", "j", 2)))
      k.store("bs", Index2D("i", "j", 2), k.load("b", Index2D("i", "j", 2)))
  
  k.barrier()

  with k.range("i", 2):
    with k.range("j", 2):
      k.set("acc", 0, 0)
      with k.range("k", 2, axis="reduce"):
        k.set(
          "acc",
          0,
          Add(
            k.load("acc", 0),
            Mul(
              k.load("as", Index2D("i", "k", 2)),
              k.load("bs", Index2D("k", "j", 2)),
            ),
          ),
        )
      k.set("out", Index2D("i", "j", 2), k.load("acc", 0))
  
  return lower_kernel(k.build(), out, a, b)

if __name__ == "__main__":
  a = Tensor([1.0, 2.0, 3.0, 4.0])
  b = Tensor([5.0, 6.0, 7.0, 8.0])
  out = Tensor.empty(4)
  out = out.custom_kernel(a, b, fxn=shared_register_gemm_kernel)[0].realize()
  print(out.tolist())