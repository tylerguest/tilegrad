import unittest
from tilegrad.ir import Add, Arg, Barrier, Const, Kernel, Load, Range, Store
from tilegrad.validate import validate_kernel

class TestValidate(unittest.TestCase):
  def test_valid_copy_kernel(self):
    kernel = Kernel(
      "copy",
      (Arg("out"), Arg("inp")),
      (Range("i", "out.numel", (Store("out", "i", Load("inp", "i")),)),),
    )
    validate_kernel(kernel)
  
  def test_duplicate_arg_name_fails(self):
    kernel = Kernel(
      "bad",
      (Arg("out"), Arg("out")),
      (Range("i", "out.numel", (Store("out", "i", 0),)),),
    )
    with self.assertRaisesRegex(ValueError, "duplicate arg name: out"): validate_kernel(kernel)
  
  def test_store_unknown_buffer_fails(self):
    kernel = Kernel(
      "bad",
      (Arg("out"),),
      (Range("i", "out.numel", (Store("missing", "i", 0),)),),
    )
    with self.assertRaisesRegex(ValueError, "unknown buffer: missing"): validate_kernel(kernel)
  
  def test_load_unknown_buffer_fails(self):
    kernel = Kernel(
      "bad",
      (Arg("out"),),
      (Range("i", "out.numel", (Store("out", "i", Load("missing", "i")),)),),
    )
    with self.assertRaisesRegex(ValueError, "unknown buffer: missing"): validate_kernel(kernel)
  
  def test_unknown_index_variable_fails(self):
    kernel = Kernel(
      "bad",
      (Arg("out"),),
      (Range("i", "out.numel", (Store("out", "j", 0),)),),
    )
    with self.assertRaisesRegex(ValueError, "unknown index variable: j"): validate_kernel(kernel)
  
  def test_range_variable_shadowing_fails(self):
    kernel = Kernel(
      "bad",
      (Arg("out"),),
      (Range("i", 4, (Range("i", 4, (Store("out", "i", 0),)),)),),
    )
    with self.assertRaisesRegex(ValueError, "duplicate range variable: i"): validate_kernel(kernel)
  
  def test_leading_barrier_fails(self):
    kernel = Kernel(
      "bad",
      (Arg("out"),),
      (Barrier(), Range("i", 2, (Store("out", "i", 0),)),),
    )
    with self.assertRaisesRegex(ValueError, "barrier requires a previous effect"): validate_kernel(kernel)
  
  def test_empty_kernel_fails(self):
    kernel = Kernel("bad", (Arg("out"),), ())
    with self.assertRaisesRegex(ValueError, "kernel must produce at least one effect"): validate_kernel(kernel)
  
  def test_const_string_fails(self):
    kernel = Kernel(
      "bad",
      (Arg("out"),),
      (Range("i", 2, (Store("out", "i", Const("i")),)),),
    )
    with self.assertRaisesRegex(TypeError, "const value must be int or float, got str"): validate_kernel(kernel)
  
  def test_zero_range_extent_fails(self):
    kernel = Kernel(
      "bad",
      (Arg("out"),),
      (Range("i", 0, (Store("out", "i", 0),)),),
    )
    with self.assertRaisesRegex(ValueError, "shape must be positive: 0"): validate_kernel(kernel)
  
  def test_unknown_shape_buffer_fails(self):
    kernel = Kernel(
      "bad",
      (Arg("out"),),
      (Range("i", "missing.numel", (Store("out", "i", 0),)),),
    )
    with self.assertRaisesRegex(ValueError, "unknown buffer: missing"): validate_kernel(kernel)

if __name__ == "__main__":
  unittest.main()