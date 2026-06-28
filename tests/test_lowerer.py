import unittest
from tinygrad import Tensor
from tilegrad.ir import Alloc, Arg, Barrier, Kernel, Load, Range, Store
from tilegrad.lowerer import lower_kernel


class TestLowerer(unittest.TestCase):
  def test_ir_zero_kernel(self):
    def zero_kernel(out):
      ir = Kernel("test_ir_zero", (Arg("out"),), (Range("i", "out.numel", (Store("out", "i", 0),)),))
      return lower_kernel(ir, out)
    out = Tensor.empty(4)
    out = out.custom_kernel(fxn=zero_kernel)[0].realize()
    self.assertEqual(out.tolist(), [0.0, 0.0, 0.0, 0.0])
  
  def test_ir_copy_kernel(self):
    def copy_kernel(out, inp):
      ir = Kernel("test_ir_copy", (Arg("out"), Arg("inp")), (Range("i", "out.numel", (Store("out", "i", Load("inp", "i")),)),))
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(inp, fxn=copy_kernel)[0].realize()
    self.assertEqual(out.tolist(), [1.0, 2.0, 3.0, 4.0])
  
  def test_ir_shared_copy_kernel(self):
    def shared_copy_kernel(out, inp):
      ir = Kernel(
        "test_ir_shared_copy",
        (Arg("out"), Arg("inp")),
        (
          Alloc("smem", "out.numel", "float32", "shared"),
          Range("i", "out.numel", (Store("smem", "i", Load("inp", "i")),)),
          Barrier(),
          Range("j", "out.numel", (Store("out", "j", Load("smem", "j")),)),
        ),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(inp, fxn=shared_copy_kernel)[0].realize()
    self.assertEqual(out.tolist(), [1.0, 2.0, 3.0, 4.0])

if __name__ == "__main__":
  unittest.main()
