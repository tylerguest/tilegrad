import unittest
from tinygrad import Tensor
from tilegrad.builder import KernelBuilder
from tilegrad.ir import Alloc, Arg, Barrier, Kernel, Load, Range, Store
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

if __name__ == "__main__":
  unittest.main()