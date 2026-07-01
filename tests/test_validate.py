import unittest
from tilegrad.ir import Add, Arg, Barrier, Const, Kernel, Load, LoadIf, Range, Set, SetIf, Store, Var, lt, And, Lt, StoreIf
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
  
  def test_top_level_set_valid(self):
    kernel = Kernel(
      "set",
      (Arg("out"),),
      (Set("out", 0, 1),),
    )
    validate_kernel(kernel)
  
  def test_unknown_range_axis_fails(self):
    kernel = Kernel(
      "bad",
      (Arg("out"),),
      (Range("i", 2, (Store("out", "i", 0),), axis="bad"),),
    )
    with self.assertRaisesRegex(ValueError, "unknown range axis: bad"): validate_kernel(kernel)
  
  def test_validate_accepts_var_indices(self):
    k = Kernel(
      "copy",
      (Arg("out"), Arg("inp")),
      (Range("i", 4, (Set("out", Var("i"), Load("inp", Var("i"))),)),),
    )
    validate_kernel(k)

  def test_validate_accepts_predicate_expr(self):
    kernel = Kernel(
      "pred",
      (Arg("out"),),
      (Range("i", 4, (Set("out", 0, lt(Var("i"), 3)),)),),
    )
    validate_kernel(kernel)

  def test_validate_accepts_store_if(self):
    kernel = Kernel(
      "guarded",
      (Arg("out"), Arg("inp")),
      (Range("i", 4, (StoreIf(lt(Var("i"), 3), "out", Var("i"), Load("inp", Var("i"))),)),),
    )
    validate_kernel(kernel)

  def test_validate_accepts_set_if(self):
    kernel = Kernel(
      "guarded_set",
      (Arg("out"), Arg("inp")),
      (Range("i", 4, (SetIf(lt(Var("i"), 3), "out", Var("i"), Load("inp", Var("i"))),)),),
    )
    validate_kernel(kernel)

  def test_validate_rejects_set_if_unknown_predicate_var(self):
    kernel = Kernel(
      "bad_guarded_set",
      (Arg("out"), Arg("inp")),
      (Range("i", 4, (SetIf(lt(Var("j"), 3), "out", Var("i"), Load("inp", Var("i"))),)),),
    )
    with self.assertRaisesRegex(ValueError, "unknown index variable: j"): validate_kernel(kernel)

  def test_validate_rejects_set_if_unknown_buffer(self):
    kernel = Kernel(
      "bad_guarded_set_buffer",
      (Arg("out"), Arg("inp")),
      (Range("i", 4, (SetIf(lt(Var("i"), 3), "missing", Var("i"), Load("inp", Var("i"))),)),),
    )
    with self.assertRaisesRegex(ValueError, "unknown buffer: missing"): validate_kernel(kernel)

  def test_validate_accepts_load_if(self):
    kernel = Kernel(
      "masked_load",
      (Arg("out"), Arg("inp")),
      (Range("i", 5, (Set("out", Var("i"), LoadIf(lt(Var("i"), 4), "inp", Var("i"))),)),),
    )
    validate_kernel(kernel)

  def test_validate_rejects_load_if_unknown_predicate_var(self):
    kernel = Kernel(
      "bad_masked_load",
      (Arg("out"), Arg("inp")),
      (Range("i", 5, (Set("out", Var("i"), LoadIf(lt(Var("j"), 4), "inp", Var("i"))),)),),
    )
    with self.assertRaisesRegex(ValueError, "unknown index variable: j"): validate_kernel(kernel)

  def test_validate_rejects_load_if_unknown_buffer(self):
    kernel = Kernel(
      "bad_masked_load_buffer",
      (Arg("out"), Arg("inp")),
      (Range("i", 5, (Set("out", Var("i"), LoadIf(lt(Var("i"), 4), "missing", Var("i"))),)),),
    )
    with self.assertRaisesRegex(ValueError, "unknown buffer: missing"): validate_kernel(kernel)

if __name__ == "__main__":
  unittest.main()
