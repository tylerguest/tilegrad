import unittest
from tilegrad.ir import *
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

  def test_register_indexed_by_range_var_fails(self):
    kernel = Kernel(
      "bad_reg_index",
      (Arg("out"), Arg("a"), Arg("b")),
      (
        Alloc("acc", 4, "float32", "register"),
        Alloc("as", 6, "float32", "shared"),
        Alloc("bs", 6, "float32", "shared"),
        Range("bi", 2, (
          Range("bj", 2, (
            Range("ii", 2, (
              Range("jj", 2, (
                Set("acc", Index2D(Var("ii"), Var("jj"), 2), 0),
              )),
            )),
          )),
        )),
      ),
    )
    with self.assertRaisesRegex(ValueError, "register buffer 'acc' cannot be indexed by a range variable"): validate_kernel(kernel)

  def test_register_constant_index_ok(self):
    kernel = Kernel(
      "reg_const_index",
      (Arg("out"), Arg("a"), Arg("b")),
      (
        Alloc("acc", 4, "float32", "register"),
        Alloc("as", 6, "float32", "shared"),
        Alloc("bs", 6, "float32", "shared"),
        Range("bi", 2, (
          Range("bj", 2, (
            Range("ii", 2, (
              Range("jj", 2, (
                Set("acc", 0, 0),
              )),
            )),
          )),
        )),
      ),
    )
    validate_kernel(kernel)

  def test_register_1d_const_in_loop_ok(self):
    kernel = Kernel(
      "reg_const_1d",
      (Arg("out"), Arg("inp")),
      (
        Alloc("acc", 1, "float32", "register"),
        Range("i", 4, (
          Set("acc", 0, Load("inp", Var("i"))),
          Set("out", Var("i"), Load("acc", 0)),
        )),
      ),
    )
    validate_kernel(kernel)

  def test_register_load_indexed_by_range_var_fails(self):
    kernel = Kernel(
      "bad_reg_load_index",
      (Arg("out"),),
      (
        Alloc("acc", 4, "float32", "register"),
        Range("ii", 2, (
          Range("jj", 2, (
            Set("out", 0, Load("acc", Index2D(Var("ii"), Var("jj"), 2))),
          )),
        )),
      ),
    )
    with self.assertRaisesRegex(ValueError, "register buffer 'acc' cannot be indexed by a range variable"): validate_kernel(kernel)

  def test_validate_accepts_fragment_alloc_clear_gemm_store(self):
    kernel = Kernel(
      "fragment_ok",
      (Arg("out"),),
      (
        Alloc("as", 6, "float32", "shared"),
        Alloc("bs", 6, "float32", "shared"),
        FragmentAlloc("acc", (2, 2), "float32"),
        Range("i", 1, (
          FragmentClear("acc"),
          FragmentGemm("as", "bs", "acc", (2, 3), (3, 2), (2, 2)),
          FragmentStore("acc", "out", 0, 0, 3, Lt(Var("i"), 1)),
        )),
      ),
    )
    validate_kernel(kernel)

  def test_fragment_duplicate_name_fails(self):
    kernel = Kernel("bad_fragment_dup", (Arg("out"),), (FragmentAlloc("out", (2, 2), "float32"),))
    with self.assertRaisesRegex(ValueError, "duplicate buffer name: out"): validate_kernel(kernel)

  def test_fragment_invalid_shape_fails(self):
    kernel = Kernel("bad_fragment_shape", (Arg("out"),), (FragmentAlloc("acc", (2, 0), "float32"),))
    with self.assertRaisesRegex(ValueError, "fragment shape must contain positive integers"): validate_kernel(kernel)

  def test_fragment_clear_unknown_fails(self):
    kernel = Kernel("bad_fragment_clear", (Arg("out"),), (FragmentClear("missing"),))
    with self.assertRaisesRegex(ValueError, "unknown buffer: missing"): validate_kernel(kernel)

  def test_fragment_clear_non_fragment_fails(self):
    kernel = Kernel("bad_fragment_clear", (Arg("out"),), (Alloc("acc", 4, "float32", "register"), FragmentClear("acc")))
    with self.assertRaisesRegex(ValueError, "fragment clear buffer must be a fragment: acc"): validate_kernel(kernel)

  def test_fragment_gemm_unknown_input_fails(self):
    kernel = Kernel(
      "bad_fragment_gemm_input",
      (Arg("out"),),
      (Alloc("bs", 6, "float32", "shared"), FragmentAlloc("acc", (2, 2), "float32"), FragmentGemm("missing", "bs", "acc", (2, 3), (3, 2), (2, 2))),
    )
    with self.assertRaisesRegex(ValueError, "unknown buffer: missing"): validate_kernel(kernel)

  def test_fragment_gemm_c_must_be_fragment_fails(self):
    kernel = Kernel(
      "bad_fragment_gemm_c",
      (Arg("out"),),
      (
        Alloc("as", 6, "float32", "shared"),
        Alloc("bs", 6, "float32", "shared"),
        Alloc("acc", 4, "float32", "register"),
        FragmentGemm("as", "bs", "acc", (2, 3), (3, 2), (2, 2)),
      ),
    )
    with self.assertRaisesRegex(ValueError, "fragment gemm C must be a fragment: acc"): validate_kernel(kernel)

  def test_fragment_gemm_shape_mismatch_fails(self):
    kernel = Kernel(
      "bad_fragment_gemm_shape",
      (Arg("out"),),
      (
        Alloc("as", 6, "float32", "shared"),
        Alloc("bs", 6, "float32", "shared"),
        FragmentAlloc("acc", (2, 2), "float32"),
        FragmentGemm("as", "bs", "acc", (2, 4), (3, 2), (2, 2)),
      ),
    )
    with self.assertRaisesRegex(ValueError, "fragment gemm shape mismatch"): validate_kernel(kernel)

  def test_scalar_load_from_fragment_fails(self):
    kernel = Kernel(
      "bad_fragment_load",
      (Arg("out"),),
      (FragmentAlloc("acc", (2, 2), "float32"), Set("out", 0, Load("acc", 0))),
    )
    with self.assertRaisesRegex(ValueError, "fragment buffer 'acc' cannot be used with scalar Load/Set/Store"): validate_kernel(kernel)

  def test_scalar_set_to_fragment_fails(self):
    kernel = Kernel("bad_fragment_set", (Arg("out"),), (FragmentAlloc("acc", (2, 2), "float32"), Set("acc", 0, 0)))
    with self.assertRaisesRegex(ValueError, "fragment buffer 'acc' cannot be used with scalar Load/Set/Store"): validate_kernel(kernel)

  def test_fragment_store_unknown_src_fails(self):
    kernel = Kernel("bad_fragment_store_src", (Arg("out"),), (FragmentStore("missing", "out", 0, 0, 3),))
    with self.assertRaisesRegex(ValueError, "unknown buffer: missing"): validate_kernel(kernel)

  def test_fragment_store_unknown_dst_fails(self):
    kernel = Kernel("bad_fragment_store_dst", (Arg("out"),), (FragmentAlloc("acc", (2, 2), "float32"), FragmentStore("acc", "missing", 0, 0, 3)))
    with self.assertRaisesRegex(ValueError, "unknown buffer: missing"): validate_kernel(kernel)

  def test_fragment_store_predicate_unknown_var_fails(self):
    kernel = Kernel(
      "bad_fragment_store_predicate",
      (Arg("out"),),
      (FragmentAlloc("acc", (2, 2), "float32"), FragmentStore("acc", "out", 0, 0, 3, Lt(Var("missing"), 1))),
    )
    with self.assertRaisesRegex(ValueError, "unknown index variable: missing"): validate_kernel(kernel)

  def test_validate_accepts_fragment_store_bounds(self):
    kernel = Kernel(
      "fragment_store_bounds",
      (Arg("out"),),
      (
        FragmentAlloc("acc", (2, 2), "float32"),
        Range("i", 1, (FragmentStore("acc", "out", 0, 0, 3, bounds=(3, 3)),)),
      ),
    )
    validate_kernel(kernel)

  def test_fragment_store_bounds_unknown_var_fails(self):
    kernel = Kernel(
      "bad_fragment_store_bounds",
      (Arg("out"),),
      (FragmentAlloc("acc", (2, 2), "float32"), FragmentStore("acc", "out", 0, 0, 3, bounds=(Var("missing"), 3))),
    )
    with self.assertRaisesRegex(ValueError, "unknown index variable: missing"): validate_kernel(kernel)
  
  def test_validate_accepts_execution_axes(self):
    for axis in ("loop", "reduce", "global", "local", "unroll"):
      kernel = Kernel(
        f"axis_{axis}",
        (Arg("out"),),
        (Range("i", 4, (Store("out", "i", 0),), axis=axis),),
      )
      validate_kernel(kernel)

  def test_validate_accepts_shape_dim_range(self):
    kernel = Kernel(
      "shape_dim_range",
      (Arg("out"), Arg("inp")),
      (Range("i", "inp.shape.0", (Store("out", "i", Load("inp", "i")),)),),
    )
    validate_kernel(kernel)

  def test_validate_accepts_shape_dim_alloc(self):
    kernel = Kernel(
      "shape_dim_alloc",
      (Arg("out"), Arg("inp")),
      (
        Alloc("smem", "inp.shape.0", "float32", "shared"),
        Range("i", "inp.shape.0", (Store("smem", "i", Load("inp", "i")),)),
        Barrier(),
        Range("j", "inp.shape.0", (Store("out", "j", Load("smem", "j")),)),
      ),
    )
    validate_kernel(kernel)

  def test_shape_dim_unknown_buffer_fails(self):
    kernel = Kernel(
      "bad_shape_dim",
      (Arg("out"),),
      (Range("i", "missing.shape.0", (Store("out", "i", 0),)),),
    )
    with self.assertRaisesRegex(ValueError, "unknown buffer: missing"):
      validate_kernel(kernel)

  def test_shape_dim_invalid_dim_fails(self):
    kernel = Kernel(
      "bad_shape_dim",
      (Arg("out"), Arg("inp")),
      (Range("i", "inp.shape.x", (Store("out", "i", 0),)),),
    )
    with self.assertRaisesRegex(ValueError, "invalid shape dimension"):
      validate_kernel(kernel)
  
  def test_validate_accepts_tile_copy(self):
    kernel = Kernel(
      "tile_copy_ok",
      (Arg("out"), Arg("inp")),
      (TileCopy("inp", "out", (2,3), (0,0), (0,0), src_stride=3, dst_stride=3, index_names=("i", "j")),),
    )
    validate_kernel(kernel)
  
  def test_tile_copy_unknown_src_fails(self):
    kernel = Kernel(
      "tile_copy_bad_src",
      (Arg("out"),),
      (TileCopy("missing", "out", (2,3), (0,0), (0,0), src_stride=3, dst_stride=3),),
    )
    with self.assertRaisesRegex(ValueError, "unknown buffer: missing"): validate_kernel(kernel)
  
  def test_tile_copy_unknown_dst_fails(self):
    kernel = Kernel(
      "tile_copy_bad_dst",
      (Arg("inp"),),
      (TileCopy("inp", "missing", (2,3), (0,0), (0,0), src_stride=3, dst_stride=3),),
    )
    with self.assertRaisesRegex(ValueError, "unknown buffer: missing"): validate_kernel(kernel)
  
  def test_tile_copy_empty_shape_fails(self):
    kernel = Kernel(
      "tile_copy_empty",
      (Arg("out"), Arg("inp")), 
      (TileCopy("inp", "out", (), (), ()),)
    )
    with self.assertRaisesRegex(ValueError, "tile copy shape must not be empty"): validate_kernel(kernel)

  def test_tile_copy_bad_origin_rank_fails(self):
    kernel = Kernel(
      "tile_copy_bad_origin",
      (Arg("out"), Arg("inp")),
      (TileCopy("inp", "out", (2,3), (0,), (0,0), src_stride=3, dst_stride=3),),
    )
    with self.assertRaisesRegex(ValueError, "src_origin rank"): validate_kernel(kernel)
  
  def test_tile_copy_bad_bounds_rank_fails(self):
    kernel = Kernel(
      "tile_copy_bad_names",
      (Arg("out"), Arg("inp")),
      (TileCopy("inp", "out", (2,3), (0,0), (0,0), src_stride=3, dst_stride=3, src_bounds=(2,)),),
    )
    with self.assertRaisesRegex(ValueError, "src_bounds rank"): validate_kernel(kernel)
  
  def test_tile_copy_bad_index_name_count_fails(self):
    kernel = Kernel(
      "tile_copy_bad_names",
      (Arg("out"), Arg("inp")),
      (TileCopy("inp", "out", (2,3), (0,0), (0,0), src_stride=3, dst_stride=3, index_names=("i",)),),
    )
    with self.assertRaisesRegex(ValueError, "index name count"): validate_kernel(kernel)
  
  def test_tile_copy_unknown_mask_var_fails(self):
    kernel = Kernel(
      "tile_copy_bad_mask",
      (Arg("out"), Arg("inp")),
      (TileCopy("inp", "out", (4,), (0,), (0,), src_mask=Lt(Var("missing"), 3), index_names=("i",)),),
    )
    with self.assertRaisesRegex(ValueError, "unknown index variable: missing"): validate_kernel(kernel)
  
  def test_tile_copy_mask_can_reference_copy_index(self):
    kernel = Kernel(
      "tile_copy_mask_index",
      (Arg("out"), Arg("inp")),
      (TileCopy("inp", "out", (4,), (0,), (0,), src_mask=Lt(Var("i"), 3), index_names=("i",)),),
    )
    validate_kernel(kernel)
  
  def test_tile_copy_unknown_guard_var_fails(self):
    kernel = Kernel(
      "tile_copy_bad_guard",
      (Arg("out"), Arg("inp")),
      (TileCopy("inp", "out", (4,), (0,), (0,), guard=Lt(Var("missing"), 3), index_names=("i",)),),
    )
    with self.assertRaisesRegex(ValueError, "unknown index variable: missing"): validate_kernel(kernel)
  
  def test_tile_copy_nonzero_fill_fails(self):
    kernel = Kernel(
      "tile_copy_bad_fill",
      (Arg("out"), Arg("inp")),
      (TileCopy("inp", "out", (4,), (0,), (0,), fill=1),)
    )
    with self.assertRaisesRegex(NotImplementedError, "fill=0"): validate_kernel(kernel)

  def test_tile_copy_accepts_coalesced_width(self):
    kernel = Kernel(
      "tile_copy_coalesced_width",
      (Arg("out"), Arg("inp")),
      (TileCopy("inp", "out", (4,), (0,), (0,), coalesced_width=4),)
    )
    validate_kernel(kernel)

  def test_tile_copy_bad_coalesced_width_fails(self):
    kernel = Kernel(
      "tile_copy_bad_coalesced_width",
      (Arg("out"), Arg("inp")),
      (TileCopy("inp", "out", (4,), (0,), (0,), coalesced_width=0),)
    )
    with self.assertRaisesRegex(ValueError, "coalesced_width"):
      validate_kernel(kernel)

  def test_tile_copy_layout_fails(self):
    kernel = Kernel(
      "tile_copy_bad_layout",
      (Arg("out"), Arg("inp")),
      (TileCopy("inp", "out", (4,), (0,), (0,), src_layout="coalesced"),)
    )
    with self.assertRaisesRegex(NotImplementedError, "layouts are not supported"): validate_kernel(kernel)

if __name__ == "__main__":
  unittest.main()
