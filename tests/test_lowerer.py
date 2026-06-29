import unittest
from tinygrad import Tensor
from tilegrad.ir import Add, Alloc, Arg, Barrier, FloorDiv, Kernel, Load, Mod, Mul, Range, Store
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

if __name__ == "__main__":
  unittest.main()
