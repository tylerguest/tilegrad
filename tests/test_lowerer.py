import unittest
from tinygrad import Tensor
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, Ops, UOp
from tilegrad.ir import Add, Alloc, Arg, Barrier, FragmentAlloc, FragmentClear, FragmentGemm, FragmentStore, FloorDiv, Kernel, Load, Lt, Mod, Mul, Range, Set, SetIf, Store, Index2D, Sub, Var
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

  def test_lower_shape_dim_range_1d(self):
    def copy_kernel(out, inp):
      ir = Kernel(
        "test_shape_dim_range_1d",
        (Arg("out"), Arg("inp")),
        (Range("i", "inp.shape.0", (Store("out", "i", Load("inp", "i")),)),),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(inp, fxn=copy_kernel)[0].realize()
    self.assertEqual(out.tolist(), [1.0, 2.0, 3.0, 4.0])

  def test_lower_shape_dim_range_2d(self):
    def copy_kernel(out, inp):
      ir = Kernel(
        "test_shape_dim_range_2d",
        (Arg("out"), Arg("inp")),
        (
          Range("i", "inp.shape.0", (
            Range("j", "inp.shape.1", (
              Store("out", Index2D("i", "j", 3), Load("inp", Index2D("i", "j", 3))),
            )),
          )),
        ),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(2, 3).realize()
    out = Tensor.empty(6)
    out = out.custom_kernel(inp, fxn=copy_kernel)[0].realize()
    self.assertEqual(out.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  
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

  def test_ir_offset_load_kernel(self):
    def offset_load_kernel(out, inp):
      ir = Kernel(
        "test_ir_offset_load",
        (Arg("out"), Arg("inp")),
        (Range("i", "out.numel", (Store("out", "i", Load("inp", Add("i", 1))),)),),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([10.0, 20.0, 30.0, 40.0])
    out = Tensor.empty(3)
    out = out.custom_kernel(inp, fxn=offset_load_kernel)[0].realize()
    self.assertEqual(out.tolist(), [20.0, 30.0, 40.0])
  
  def test_ir_offset_store_kernel(self):
    def offset_store_kernel(out, inp):
      ir = Kernel(
        "test_ir_offset_store",
        (Arg("out"), Arg("inp")),
        (
          Range("i", "out.numel", (Store("out", "i", 0),)),
          Range("j", 3, (Store("out", Add("j", 1), Load("inp", "j")),)),
        ),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(inp, fxn=offset_store_kernel)[0].realize()
    self.assertEqual(out.tolist(), [0.0, 1.0, 2.0, 3.0])

  def test_ir_mul_index_kernel(self):
    def mul_index_kernel(out, inp):
      ir = Kernel(
        "test_ir_mul_index",
        (Arg("out"), Arg("inp")),
        (Range("j", 2, (
          Store("out", Mul("j", 2), Load("inp", "j")),
          Store("out", Add(Mul("j", 2), 1), 0),
        )),),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([5.0, 7.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(inp, fxn=mul_index_kernel)[0].realize()
    self.assertEqual(out.tolist(), [5.0, 0.0, 7.0, 0.0])

  def test_ir_two_shared_allocs_kernel(self):
    def two_shared_allocs_kernel(out, inp):
      ir = Kernel(
        "test_ir_two_shared_allocs",
        (Arg("out"), Arg("inp")),
        (
          Alloc("a", 2, "float32", "shared"),
          Alloc("b", 2, "float32", "shared"),
          Range("i", 2, (
            Store("a", "i", Load("inp", "i")),
            Store("b", "i", Add(Load("inp", "i"), 10)),
          )),
          Barrier(),
          Range("j", 2, (
            Store("out", Mul("j", 2), Load("a", "j")),
            Store("out", Add(Mul("j", 2), 1), Load("b", "j")),
          )),
        ),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(inp, fxn=two_shared_allocs_kernel)[0].realize()
    self.assertEqual(out.tolist(), [1.0, 11.0, 2.0, 12.0])

  def test_duplicate_alloc_name_fails(self):
    def duplicate_alloc_kernel(out):
      ir = Kernel(
        "test_duplicate_alloc_name",
        (Arg("out"),),
        (
          Alloc("smem", 2, "float32", "shared"),
          Alloc("smem", 2, "float32", "shared"),
          Range("i", 2, (Store("out", "i", 0),)),
        ),
      )
      return lower_kernel(ir, out)
    out = Tensor.empty(2)
    with self.assertRaises(ValueError): out.custom_kernel(fxn=duplicate_alloc_kernel)[0].realize()

  def test_leading_barrier_fails(self):
    def leading_barrier_kernel(out):
      ir = Kernel(
        "test_leading_barrier",
        (Arg("out"),),
        (
          Barrier(),
          Range("i", 2, (Store("out", "i", 0),)),
        ),
      )
      return lower_kernel(ir, out)
    out = Tensor.empty(2)
    with self.assertRaisesRegex(ValueError, "barrier requires a previous effect"): out.custom_kernel(fxn=leading_barrier_kernel)[0].realize()

  def test_empty_kernel_fails(self):
    ir = Kernel("test_empty_kernel", (Arg("out"),), ())
    out = Tensor.empty(2)
    with self.assertRaisesRegex(ValueError, "kernel must produce at least one effect"): lower_kernel(ir, out.uop)

  def test_ir_transpose_2d_index_kernel(self):
    def transpose_2d_index_kernel(out, inp):
      row = FloorDiv("i", 3)
      col = Mod("i", 3)
      out_idx = Add(Mul(col, 2), row)
      ir = Kernel(
        "test_ir_transpose_2d_index",
        (Arg("out"), Arg("inp")),
        (Range("i", 6, (Store("out", out_idx, Load("inp", "i")),)),),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = Tensor.empty(6)
    out = out.custom_kernel(inp, fxn=transpose_2d_index_kernel)[0].realize()
    self.assertEqual(out.tolist(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
  
  def test_sequential_ranges_overwrite_in_order(self):
    def sequential_ranges_kernel(out, inp):
      ir = Kernel(
        "test_sequential_ranges_overwrite",
        (Arg("out"), Arg("inp")),
        (
          Range("i", 4, (Store("out", "i", 0),)),
          Range("j", 3, (Store("out", Add("j", 1), Load("inp", "j")),)),
        ),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(inp, fxn=sequential_ranges_kernel)[0].realize()
    self.assertEqual(out.tolist(), [0.0, 1.0, 2.0, 3.0])
  
  def test_range_body_overwrites_in_order(self):
    def range_body_order_kernel(out):
      ir = Kernel(
        "test_range_body_overwrites",
        (Arg("out"),),
        (Range("i", 4, (
          Store("out", "i", 1),
          Store("out", "i", 2),
        )),),
      )
      return lower_kernel(ir, out)
    out = Tensor.empty(4)
    out = out.custom_kernel(fxn=range_body_order_kernel)[0].realize()
    self.assertEqual(out.tolist(), [2.0, 2.0, 2.0, 2.0])
  
  def test_lower_kernel_too_few_args_fails(self):
    ir = Kernel(
      "test_too_few_args",
      (Arg("out"), Arg("inp")),
      (Range("i", 2, (Store("out", "i", Load("inp", "i")),)),),
    )
    out = Tensor.empty(2)
    with self.assertRaisesRegex(ValueError, "expected 2 args, got 1"): lower_kernel(ir, out.uop)

  def test_lower_kernel_too_many_args_fails(self):
    ir = Kernel(
      "test_too_many_args",
      (Arg("out"),),
      (Range("i", 2, (Store("out", "i", 0),)),),
    )
    out = Tensor.empty(2)
    extra = Tensor.empty(2)
    with self.assertRaisesRegex(ValueError, "expected 1 args, got 2"): lower_kernel(ir, out.uop, extra.uop)
  
  def test_ir_sum_reduce_kernel(self):
    def sum_kernel(out, inp):
      ir = Kernel(
        "test_ir_sum_reduce",
        (Arg("out"), Arg("inp")),
        (
          Set("out", 0, 0),
          Range("i", "inp.numel", (Set("out", 0, Add(Load("out", 0), Load("inp", "i"))),), axis="reduce"),
        ),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(1)
    out = out.custom_kernel(inp, fxn=sum_kernel)[0].realize()
    self.assertEqual(out.tolist(), [10.0])

  def test_ir_index2d_transpose_kernel(self):
    def transpose_kernel(out, inp):
      row = FloorDiv("i", 3)
      col = Mod("i", 3)
      ir = Kernel(
        "test_ir_index2d_transpose",
        (Arg("out"), Arg("inp")),
        (
          Range("i", 6, (Store("out", Index2D(col, row, 2), Load("inp", "i")),)),
        ),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = Tensor.empty(6)
    out = out.custom_kernel(inp, fxn=transpose_kernel)[0].realize()
    self.assertEqual(out.tolist(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
  
  def test_ir_sub_reverse_kernel(self):
    def reverse_kernel(out, inp):
      ir = Kernel(
        "test_ir_sub_reverse",
        (Arg("out"), Arg("inp")),
        (
          Range("i", 4, (Store("out", "i", Load("inp", Sub(3, "i"))),)),
        ),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(inp, fxn=reverse_kernel)[0].realize()
    self.assertEqual(out.tolist(), [4.0, 3.0, 2.0, 1.0])

  def test_ir_nested_range_2d_fill_kernel(self):
    def fill_kernel(out):
      ir = Kernel(
        "test_ir_nested_range_2d_fill",
        (Arg("out"),),
        (
          Range("i", 2, (Range("j", 3, (Store("out", Index2D("i", "j", 3), Add(Mul("i", 10), "j")),)),)),
        ),
      )
      return lower_kernel(ir, out)
    out = Tensor.empty(6)
    out = out.custom_kernel(fxn=fill_kernel)[0].realize()
    self.assertEqual(out.tolist(), [0.0, 1.0, 2.0, 10.0, 11.0, 12.0])

  def test_ir_naive_gemm_2x2_kernel(self):
    def gemm_kernel(out, a, b):
      ir = Kernel(
        "test_ir_naive_gemm_2x2",
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
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([5.0, 6.0, 7.0, 8.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(a, b, fxn=gemm_kernel)[0].realize()
    self.assertEqual(out.tolist(), [19.0, 22.0, 43.0, 50.0])

  def test_ir_naive_gemm_2x3x2_kernel(self):
    def gemm_kernel(out, a, b):
      ir = Kernel(
        "test_ir_naive_gemm_2x3x2",
        (Arg("out"), Arg("a"), Arg("b")),
        (
          Range("i", 2, (
            Range("j", 2, (
              Set("out", Index2D("i", "j", 2), 0),
              Range("k", 3, (
                Set(
                  "out",
                  Index2D("i", "j", 2),
                  Add(
                    Load("out", Index2D("i", "j", 2)),
                    Mul(
                      Load("a", Index2D("i", "k", 3)),
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

    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(a, b, fxn=gemm_kernel)[0].realize()
    self.assertEqual(out.tolist(), [58.0, 64.0, 139.0, 154.0])

  def test_ir_register_scalar_set_kernel(self):
    def register_scalar_kernel(out):
      ir = Kernel(
        "test_ir_register_scalar_set",
        (Arg("out"),),
        (
          Alloc("acc", 1, "float32", "register"),
          Set("acc", 0, 7),
          Set("out", 0, Load("acc", 0)),
        ),
      )
      return lower_kernel(ir, out)
    out = Tensor.empty(1)
    out = out.custom_kernel(fxn=register_scalar_kernel)[0].realize()
    self.assertEqual(out.tolist(), [7.0])
  
  def test_ir_register_sum_reduce_kernel(self):
    def register_sum_kernel(out, inp):
      ir = Kernel(
        "test_ir_register_sum_reduce",
        (Arg("out"), Arg("inp")),
        (
          Alloc("acc", 1, "float32", "register"),
          Set("acc", 0, 0),
          Range("i", "inp.numel", (Set("acc", 0, Add(Load("acc", 0), Load("inp", "i"))),), axis="reduce"),
          Set("out", 0, Load("acc", 0)),
        ),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(1)
    out = out.custom_kernel(inp, fxn=register_sum_kernel)[0].realize()
    self.assertEqual(out.tolist(), [10.0])

  def test_ir_register_row_sum_kernel(self):
    def row_sum_kernel(out, inp):
      ir = Kernel(
        "test_ir_register_row_sum",
        (Arg("out"), Arg("inp")),
        (
          Alloc("acc", 1, "float32", "register"),
          Range("i", 2, (
            Set("acc", 0, 0),
            Range("j", 3, (
              Set("acc", 0, Add(Load("acc", 0), Load("inp", Index2D("i", "j", 3)))),
            ), axis="reduce"),
            Set("out", "i", Load("acc", 0)),
          )),
        ),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = Tensor.empty(2)
    out = out.custom_kernel(inp, fxn=row_sum_kernel)[0].realize()
    self.assertEqual(out.tolist(), [6.0, 15.0])
  
  def test_ir_register_dot_product_kernel(self):
    def dot_kernel(out, a, b):
      ir = Kernel(
        "test_ir_register_dot_product",
        (Arg("out"), Arg("a"), Arg("b")),
        (
          Alloc("acc", 1, "float32", "register"),
          Set("acc", 0, 0),
          Range("k", 3, (
            Set(
              "acc",
              0,
              Add(
                Load("acc", 0),
                Mul(Load("a", "k"), Load("b", "k")),
              ),
            ),
          ), axis="reduce"),
          Set("out", 0, Load("acc", 0)),
        ),
      )
      return lower_kernel(ir, out, a, b)
    
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    out = Tensor.empty(1)
    out = out.custom_kernel(a, b, fxn=dot_kernel)[0].realize()
    self.assertEqual(out.tolist(), [32.0])
  
  def test_ir_register_naive_gemm_2x2_kernel(self):
    def gemm_kernel(out, a, b):
      ir = Kernel(
        "test_ir_register_naive_gemm_2x2",
        (Arg("out"), Arg("a"), Arg("b")),
        (
          Alloc("acc", 1, "float32", "register"),
          Range("i", 2, (
            Range("j", 2, (
              Set("acc", 0, 0),
              Range("k", 2, (
                Set(
                  "acc",
                  0,
                  Add(
                    Load("acc", 0),
                    Mul(
                      Load("a", Index2D("i", "k", 2)),
                      Load("b", Index2D("k", "j", 2)),
                    ),
                  ),
                ),
              ), axis="reduce"),
              Set("out", Index2D("i", "j", 2), Load("acc", 0)),
            )),
          )),
        ),
      )
      return lower_kernel(ir, out, a, b)
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([5.0, 6.0, 7.0, 8.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(a, b, fxn=gemm_kernel)[0].realize()
    self.assertEqual(out.tolist(), [19.0, 22.0, 43.0, 50.0])

  def test_ir_shared_tile_2x2_copy_kernel(self):
    def shared_tile_kernel(out, inp):
      ir = Kernel(
        "test_ir_shared_tile_2x2_copy",
        (Arg("out"), Arg("inp")),
        (
          Alloc("tile", 4, "float32", "shared"),
          Range("i", 2, (
            Range("j", 2, (
              Store("tile", Index2D("i", "j", 2), Load("inp", Index2D("i", "j", 2))),
            )),
          )),
          Barrier(),
          Range("i", 2, (
            Range("j", 2, (
              Store("out", Index2D("i", "j", 2), Load("tile", Index2D("i", "j", 2))),
            )),
          )),
        ),
      )
      return lower_kernel(ir, out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(inp, fxn=shared_tile_kernel)[0].realize()
    self.assertEqual(out.tolist(), [1.0, 2.0, 3.0, 4.0])

  def test_ir_shared_register_gemm_2x2_kernel(self):
    def gemm_kernel(out, a, b):
      ir = Kernel(
        "test_ir_shared_register_gemm_2x2",
        (Arg("out"), Arg("a"), Arg("b")),
        (
          Alloc("as", 4, "float32", "shared"),
          Alloc("bs", 4, "float32", "shared"),
          Alloc("acc", 1, "float32", "register"),
          Range("i", 2, (
            Range("j", 2, (
              Store("as", Index2D("i", "j", 2), Load("a", Index2D("i", "j", 2))),
              Store("bs", Index2D("i", "j", 2), Load("b", Index2D("i", "j", 2))),
            )),
          )),
          Barrier(),
          Range("i", 2, (
            Range("j", 2, (
              Set("acc", 0, 0),
              Range("k", 2, (
                Set(
                  "acc",
                  0,
                  Add(
                    Load("acc", 0),
                    Mul(
                      Load("as", Index2D("i", "k", 2)),
                      Load("bs", Index2D("k", "j", 2)),
                    ),
                  ),
                ),
              ), axis="reduce"),
              Set("out", Index2D("i", "j", 2), Load("acc", 0)),
            )),
          )),
        ),
      )
      return lower_kernel(ir, out, a, b)
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([5.0, 6.0, 7.0, 8.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(a, b, fxn=gemm_kernel)[0].realize()
    self.assertEqual(out.tolist(), [19.0, 22.0, 43.0, 50.0])

  def test_ir_shared_register_gemm_2x3x2_kernel(self):
    def gemm_kernel(out, a, b):
      ir = Kernel(
        "test_ir_shared_register_gemm_2x3x2",
        (Arg("out"), Arg("a"), Arg("b")),
        (
          Alloc("as", 6, "float32", "shared"),
          Alloc("bs", 6, "float32", "shared"),
          Alloc("acc", 1, "float32", "register"),
          Range("i", 2, (
            Range("k", 3, (
              Store("as", Index2D("i", "k", 3), Load("a", Index2D("i", "k", 3))),
            )),
          )),
          Range("k", 3, (
            Range("j", 2, (
              Store("bs", Index2D("k", "j", 2), Load("b", Index2D("k", "j", 2))),
            )),
          )),
          Barrier(),
          Range("i", 2, (
            Range("j", 2, (
              Set("acc", 0, 0),
              Range("k", 3, (
                Set(
                  "acc",
                  0,
                  Add(
                    Load("acc", 0),
                    Mul(
                      Load("as", Index2D("i", "k", 3)),
                      Load("bs", Index2D("k", "j", 2)),
                    ),
                  ),
                ),
              ), axis="reduce"),
              Set("out", Index2D("i", "j", 2), Load("acc", 0)),
            )),
          )),
        ),
      )
      return lower_kernel(ir, out, a, b)
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(a, b, fxn=gemm_kernel)[0].realize()
    self.assertEqual(out.tolist(), [58.0, 64.0, 139.0, 154.0])

  def test_lower_set_if_with_nested_ranges_register(self):
    ir = Kernel(
      "test_lower_set_if_nested_ranges_register",
      (Arg("out"),),
      (
        Alloc("acc", 1, "float32", "register"),
        Range("i", 2, (
          SetIf(Lt(Var("i"), 1), "acc", 0, 7),
          Store("out", "i", Load("acc", 0)),
        )),
      ),
    )
    out = UOp.placeholder((2,), dtypes.float, slot=-1)
    sink = lower_kernel(ir, out)

    ranges = [u for u in sink.toposort() if u.op is Ops.RANGE]
    self.assertEqual(len(ranges), 1)
    i = ranges[0]

    reg_store_ends = [
      u for u in sink.toposort()
      if u.op is Ops.END and u.src and u.src[0].op is Ops.STORE and u.src[0].src[0].addrspace is AddrSpace.REG
    ]

    # The loop-axis SetIf on the register buffer must end the conditional
    # register store over the newly opened register scope. Without lower_set
    # passing register_end_ranges into set(end=...) for conditional loop-axis
    # register sets, the register STORE containing the WHERE is not wrapped in
    # END(i), and later control-flow rewrites can see an inconsistent register
    # AFTER chain.
    self.assertTrue(any(
      end.src[0].src[1].op is Ops.WHERE and i in end.src[1:]
      for end in reg_store_ends
    ))

  def test_lower_unrolls_register_tile_indices(self):
    ir = Kernel(
      "test_lower_unrolls_register_tile_indices",
      (Arg("out"),),
      (
        Alloc("acc", 4, "float32", "register"),
        Range("ii", 2, (
          Range("jj", 2, (
            Set("acc", Index2D(Var("ii"), Var("jj"), 2), Add(Mul(Var("ii"), 10), Var("jj"))),
          )),
        )),
        Range("ii", 2, (
          Range("jj", 2, (
            Store("out", Index2D(Var("ii"), Var("jj"), 2), Load("acc", Index2D(Var("ii"), Var("jj"), 2))),
          )),
        )),
      ),
    )
    out = UOp.placeholder((4,), dtypes.float, slot=-1)
    sink = lower_kernel(ir, out)

    reg_indexes = [u for u in sink.toposort() if u.op is Ops.INDEX and u.src[0].addrspace is AddrSpace.REG]
    self.assertTrue(reg_indexes)
    self.assertTrue(all(u.src[1].op is Ops.CONST for u in reg_indexes))
    self.assertEqual(sorted({u.src[1].arg for u in reg_indexes}), [0, 1, 2, 3])

  def test_lower_fragment_alloc_clear_indices_constant(self):
    ir = Kernel(
      "test_lower_fragment_alloc_clear_indices_constant",
      (Arg("out"),),
      (
        FragmentAlloc("acc", (2, 2), "float32"),
        FragmentClear("acc"),
        Store("out", 0, Load("acc", 0)),
      ),
    )
    out = UOp.placeholder((1,), dtypes.float, slot=-1)
    sink = lower_kernel(ir, out)

    reg_indexes = [u for u in sink.toposort() if u.op is Ops.INDEX and u.src[0].addrspace is AddrSpace.REG]
    self.assertTrue(reg_indexes)
    self.assertTrue(all(u.src[1].op is Ops.CONST for u in reg_indexes))

  def test_lower_grid_threads_axis_types(self):
    ir = Kernel(
      "test_lower_grid_threads_axis_types",
      (Arg("out"),),
      (
        Range("b", 2, (
          Range("t", 4, (
            Store("out", Add(Mul(Var("b"), 4), Var("t")), 1),
          ), axis="local"),
        ), axis="global"),
      ),
    )
    out = UOp.placeholder((8,), dtypes.float, slot=-1)
    sink = lower_kernel(ir, out)
    ranges = [u for u in sink.toposort() if u.op is Ops.RANGE]
    axis_types = [r.arg[1] for r in ranges]
    self.assertIn(AxisType.GLOBAL, axis_types)
    self.assertIn(AxisType.LOCAL, axis_types)
  
  def test_lower_unroll_axis_type(self):
    ir = Kernel(
      "test_lower_unroll_axis_type",
      (Arg("out"),),
      (
        Range("u", 4, (
          Store("out", Var("u"), Var("u")),
        ), axis="unroll"),
      ),
    )

    out = UOp.placeholder((4,), dtypes.float, slot=-1)
    sink = lower_kernel(ir, out)
    ranges = [u for u in sink.toposort() if u.op is Ops.RANGE]
    axis_types = [r.arg[1] for r in ranges]
    self.assertIn(AxisType.UNROLL, axis_types)

if __name__ == "__main__":
  unittest.main()
