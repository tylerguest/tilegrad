import unittest
from tinygrad import Tensor
from tilegrad.builder import KernelBuilder
from tilegrad.ir import Add, Alloc, Arg, Barrier, Index2D, Kernel, Load, Mul, Range, Set, Store
from tilegrad.lowerer import lower_kernel

class TestBuilder(unittest.TestCase):
  def test_builder_copy_ir(self):
    k = KernelBuilder("copy", ("out", "inp"))
    with k.range("i", "out.numel"): k.store("out", "i", k.load("inp", "i"))
    expected = Kernel(
      "copy", 
      (Arg("out"), Arg("inp")),
      (Range("i", "out.numel", (Store("out", "i", Load("inp", "i")),)),),
    )
    self.assertEqual(k.build(), expected)
  
  def test_builder_shared_copy_ir(self):
    k = KernelBuilder("shared_copy", ("out", "inp"))
    k.alloc("smem", "out.numel", "float32")
    with k.range("i", "out.numel"): k.store("smem", "i", k.load("inp", "i"))
    k.barrier()
    with k.range("j", "out.numel"): k.store("out", "j", k.load("smem", "j"))
    expected = Kernel(
      "shared_copy",
      (Arg("out"), Arg("inp")),
      (
        Alloc("smem", "out.numel", "float32", "shared"),
        Range("i", "out.numel", (Store("smem", "i", Load("inp", "i")),)),
        Barrier(),
        Range("j", "out.numel", (Store("out", "j", Load("smem", "j")),)),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_copy_kernel_runs(self):
    def copy_kernel(out, inp):
      k = KernelBuilder("builder_copy", ("out", "inp"))
      with k.range("i", "out.numel"): k.store("out", "i", k.load("inp", "i"))
      return lower_kernel(k.build(), out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(inp, fxn=copy_kernel)[0].realize()
    self.assertEqual(out.tolist(), [1.0, 2.0, 3.0, 4.0])
  
  def test_store_outside_range_fails(self):
    k = KernelBuilder("bad", ("out",))
    with self.assertRaisesRegex(ValueError, "store requires an active range"): k.store("out", "i", 0)
  
  def test_alloc_inside_range_fails(self):
    k = KernelBuilder("bad", ("out",))
    with k.range("i", 2):
      with self.assertRaisesRegex(ValueError, "alloc must be top-level"): k.alloc("smem", 2, "float32")
  
  def test_barrier_inside_range_fails(self):
    k = KernelBuilder("bad", ("out",))
    with k.range("i", 2): 
      with self.assertRaisesRegex(ValueError, "barrier must be top-level"): k.barrier()

  def test_builder_sum_reduce_ir(self):
    k = KernelBuilder("sum", ("out", "inp"))
    k.set("out", 0, 0)
    with k.range("i", "inp.numel", axis="reduce"):
      k.set("out", 0, Add(k.load("out", 0), k.load("inp", "i")))
    expected = Kernel(
      "sum",
      (Arg("out"), Arg("inp")),
      (
        Set("out", 0, 0),
        Range("i", "inp.numel", (Set("out", 0, Add(Load("out", 0), Load("inp", "i"))),), "reduce"),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_naive_gemm_2x2_ir(self):
    k = KernelBuilder("gemm", ("out", "a", "b"))
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
    expected = Kernel(
      "gemm",
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
            ), "reduce"),
          )),
        )),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_naive_gemm_2x2_kernel_runs(self):
    def gemm_kernel(out, a, b):
      k = KernelBuilder("builder_naive_gemm_2x2", ("out", "a", "b"))
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

    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([5.0, 6.0, 7.0, 8.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(a, b, fxn=gemm_kernel)[0].realize()
    self.assertEqual(out.tolist(), [19.0, 22.0, 43.0, 50.0])

if __name__ == "__main__":
  unittest.main()
