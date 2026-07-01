import unittest
from tinygrad import Tensor
from tilegrad.builder import KernelBuilder
from tilegrad.ir import Add, Alloc, Arg, Barrier, Index2D, Kernel, Load, Mul, Range, Set, Store, Var
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
  
  def test_barrier_inside_range_ir(self):
    k = KernelBuilder("nested_barrier", ("out", "inp"))
    k.alloc("smem", 2, "float32")
    with k.range("i", 2):
      k.store("smem", "i", k.load("inp", "i"))
      k.barrier()
      k.store("out", "i", k.load("smem", "i"))
    expected = Kernel(
      "nested_barrier",
      (Arg("out"), Arg("inp")),
      (
        Alloc("smem", 2, "float32", "shared"),
        Range("i", 2, (Store("smem", "i", Load("inp", "i")), Barrier(), Store("out", "i", Load("smem", "i")))),
      ),
    )
    self.assertEqual(k.build(), expected)

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

  def test_builder_copy_1d_ir(self):
    k = KernelBuilder("copy_1d", ("out", "inp"))
    k.alloc("smem", 4, "float32")
    k.copy("inp", "smem", shape=(4,))
    k.barrier()
    with k.range("j", 4): k.store("out", "j", k.load("smem", "j"))
    expected = Kernel(
      "copy_1d",
      (Arg("out"), Arg("inp")),
      (
        Alloc("smem", 4, "float32", "shared"),
        Range("_c0_i0", 4, (
          Store("smem", "_c0_i0", Load("inp", "_c0_i0")),
        )),
        Barrier(),
        Range("j", 4, (Store("out", "j", Load("smem", "j")),)),
      ),
    )
    self.assertEqual(k.build(), expected)
  
  def test_builder_copy_2d_ir(self):
    k = KernelBuilder("copy_2d", ("out", "inp"))
    k.alloc("smem", 6, "float32")
    k.copy("inp", "smem", shape=(2, 3), stride=3)
    k.barrier()
    with k.range("j", 6): k.store("out", "j", k.load("smem", "j"))
    expected = Kernel(
      "copy_2d",
      (Arg("out"), Arg("inp")),
      (
        Alloc("smem", 6, "float32", "shared"),
        Range("_c0_i0", 2, (
          Range("_c0_i1", 3, (
            Store("smem", Index2D("_c0_i0", "_c0_i1", 3), Load("inp", Index2D("_c0_i0", "_c0_i1", 3))),
          )),
        )),
        Barrier(),
        Range("j", 6, (Store("out", "j", Load("smem", "j")),)),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_copy_2d_offset_ir(self):
    k = KernelBuilder("copy_2d_offset", ("out", "inp"))
    k.alloc("smem", 6, "float32")
    with k.range("ko", 2):
      k.copy("inp", "smem", shape=(2, 3), stride=6, src_row_off=Mul("ko", 3))
    k.barrier()
    with k.range("j", 6): k.store("out", "j", k.load("smem", "j"))
    expected = Kernel(
      "copy_2d_offset",
      (Arg("out"), Arg("inp")),
      (
        Alloc("smem", 6, "float32", "shared"),
        Range("ko", 2, (
          Range("_c0_i0", 2, (
            Range("_c0_i1", 3, (
              Store("smem", Index2D("_c0_i0", "_c0_i1", 3), 
              Load("inp", Index2D(Add(Mul("ko", 3), "_c0_i0"), "_c0_i1", 6))),
            )),
          )),
        )),
        Barrier(),
        Range("j", 6, (Store("out", "j", Load("smem", "j")),)),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_copy_1d_col_offset_ir(self):
    k = KernelBuilder("copy_1d_col_offset", ("out", "inp"))
    k.alloc("smem", 3, "float32")
    with k.range("ko", 2):
      k.copy("inp", "smem", shape=(3,), src_col_off=Mul("ko", 3))
    expected = Kernel(
      "copy_1d_col_offset",
      (Arg("out"), Arg("inp")),
      (
        Alloc("smem", 3, "float32", "shared"),
        Range("ko", 2, (
          Range("_c0_i0", 3, (
            Store("smem", "_c0_i0", Load("inp", Add(Mul("ko", 3), "_c0_i0"))),
          )),
        )),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_copy_1d_strided_ir(self):
    k = KernelBuilder("copy_1d_strided", ("out", "inp"))
    k.alloc("smem", 3, "float32")
    with k.range("ko", 2):
      with k.range("j", 2):
        k.copy("inp", "smem", shape=(3,), stride=2, src_row_off=Mul("ko", 3), src_col_off="j")
    expected = Kernel(
      "copy_1d_strided",
      (Arg("out"), Arg("inp")),
      (
        Alloc("smem", 3, "float32", "shared"),
        Range("ko", 2, (
          Range("j", 2, (
            Range("_c0_i0", 3, (
              Store("smem", "_c0_i0", Load("inp", Index2D(Add(Mul("ko", 3), "_c0_i0"), "j", 2))),
            )),
          )),
        )),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_copy_2d_col_offset_ir(self):
    k = KernelBuilder("copy_2d_col_offset", ("out", "inp"))
    k.alloc("smem", 6, "float32")
    with k.range("i", 2):
      with k.range("ko", 2):
        k.copy("inp", "smem", shape=(2, 3), stride=6, src_row_off="i", src_col_off=Mul("ko", 3))
    expected = Kernel(
      "copy_2d_col_offset",
      (Arg("out"), Arg("inp")),
      (
        Alloc("smem", 6, "float32", "shared"),
        Range("i", 2, (
          Range("ko", 2, (
            Range("_c0_i0", 2, (
              Range("_c0_i1", 3, (
                Store(
                  "smem",
                  Index2D("_c0_i0", "_c0_i1", 3),
                  Load("inp", Index2D(Add("i", "_c0_i0"), Add(Mul("ko", 3), "_c0_i1"), 6)),
                ),
              )),
            )),
          )),
        )),
      ),
    )
    self.assertEqual(k.build(), expected)
  
  def test_builder_copy_2d_kernel_runs(self):
    def copy_kernel(out, inp):
      k = KernelBuilder("builder_copy_2d", ("out", "inp"))
      k.alloc("smem", 4, "float32")
      k.copy("inp", "smem", shape=(2, 2), stride=2)
      k.barrier()
      with k.range("j", 4): k.store("out", "j", k.load("smem", "j"))
      return lower_kernel(k.build(), out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(inp, fxn=copy_kernel)[0].realize()
    self.assertEqual(out.tolist(), [1.0, 2.0, 3.0, 4.0])

  def test_ir_tiled_gemm_2x3x2_ko1(self):
    def tiled_gemm_kernel(out, a, b):
      ir = Kernel(
        "test_ir_tiled_gemm_2x3x2_ko1",
        (Arg("out"), Arg("a"), Arg("b")),
        (
          Alloc("as", 3, "float32", "shared"),
          Alloc("bs", 3, "float32", "shared"),
          Alloc("acc", 1, "float32", "register"),
          Range("i", 2, (
            Range("j", 2, (
              Set("acc", 0, 0),
              Range("ko", 1, (
                Range("ki", 3, (
                  Store("as", "ki", Load("a", Index2D("i", Add(Mul("ko", 3), "ki"), 3))),
                )),
                Range("ki", 3, (
                  Store("bs", "ki", Load("b", Index2D(Add(Mul("ko", 3), "ki"), "j", 2))),
                )),
                Barrier(),
                Range("k", 3, (
                  Set("acc", 0, Add(Load("acc", 0), Mul(Load("as", "k"), Load("bs", "k")))),
                ), axis="reduce"),
              )),
              Set("out", Index2D("i", "j", 2), Load("acc", 0)),
            )),
          )),
        ),
      )
      return lower_kernel(ir, out, a, b)
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(a, b, fxn=tiled_gemm_kernel)[0].realize()
    self.assertEqual(out.tolist(), [58.0, 64.0, 139.0, 154.0])

  def test_ir_tiled_gemm_2x6x2_ko2(self):
    def tiled_gemm_kernel(out, a, b):
      ir = Kernel(
        "test_ir_tiled_gemm_2x6x2_ko2",
        (Arg("out"), Arg("a"), Arg("b")),
        (
          Alloc("as", 3, "float32", "shared"),
          Alloc("bs", 3, "float32", "shared"),
          Alloc("acc", 1, "float32", "register"),
          Range("i", 2, (
            Range("j", 2, (
              Set("acc", 0, 0),
              Range("ko", 2, (
                Range("ki", 3, (
                  Store("as", "ki", Load("a", Index2D("i", Add(Mul("ko", 3), "ki"), 6))),
                )),
                Range("ki", 3, (
                  Store("bs", "ki", Load("b", Index2D(Add(Mul("ko", 3), "ki"), "j", 2))),
                )),
                Barrier(),
                Range("k", 3, (
                  Set("acc", 0, Add(Load("acc", 0), Mul(Load("as", "k"), Load("bs", "k")))),
                ), axis="reduce"),
              )),
              Set("out", Index2D("i", "j", 2), Load("acc", 0)),
            )),
          )),
        ),
      )
      return lower_kernel(ir, out, a, b)
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
    out = out.custom_kernel(a, b, fxn=tiled_gemm_kernel)[0].realize()
    self.assertEqual(out.tolist(), [413.0, 434.0, 1061.0, 1118.0])

  def test_builder_tiled_gemm_2x6x2_ko2_kernel_runs(self):
    def gemm_kernel(out, a, b):
      k = KernelBuilder("builder_tiled_gemm_2x6x2_ko2", ("out", "a", "b"))
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
              k.set("acc", 0, Add(k.load("acc", 0), Mul(k.load("as", "kk"), k.load("bs", "kk"))))
          k.set("out", Index2D("i", "j", 2), k.load("acc", 0))

      return lower_kernel(k.build(), out, a, b)
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
    self.assertEqual(out.tolist(), [413.0, 434.0, 1061.0, 1118.0])
  
  def test_ir_register_accumulates_across_loop(self):
    def accum_kernel(out):
      ir = Kernel(
        "test_ir_register_accumulates_across_loop",
        (Arg("out"),),
        (
          Alloc("acc", 1, "float32", "register"),
          Set("acc", 0, 0),
          Range("ko", 2, (
            Range("k", 3, (
              Set("acc", 0, Add(Load("acc", 0), 1)),
            ), axis="reduce"),
          )),
          Set("out", 0, Load("acc", 0)),
        ),
      )
      return lower_kernel(ir, out)
    out = Tensor.empty(1)
    out = out.custom_kernel(fxn=accum_kernel)[0].realize()
    self.assertEqual(out.tolist(), [6.0])

  def test_range_returns_var(self):
    k = KernelBuilder("copy", ("out", "inp"))
    out = k.buffer("out")
    inp = k.buffer("inp")
    with k.range("i", 4) as i:
      out[i] = inp[i]
    kernel = k.build()
    assert kernel.body == (
      Range("i", 4, (
        Set("out", Var("i"), Load("inp", Var("i"))),
      )),
    )
  
  def test_buffer_ref_2d_indexing(self):
    k = KernelBuilder("copy2d", ("out", "inp"))
    out = k.buffer("out", shape=(2, 3))
    inp = k.buffer("inp", shape=(2, 3))
    with k.range("i", 2) as i:
      with k.range("j", 3) as j:
        out[i, j] = inp[i, j]
    kernel = k.build()
    assert kernel.body == (
      Range("i", 2, (
        Range("j", 3, (
          Set("out", Index2D(Var("i"), Var("j"), 3), Load("inp", Index2D(Var("i"), Var("j"), 3))),
        )),
      )),
    )
  
  def test_var_operator_expressions(self):
    i = Var("i")
    j = Var("j")
    assert i * 4 + j == Add(Mul(i, 4), j)

  def test_parallel_2d_buffer_refs(self):
    k = KernelBuilder("vecadd_2d", ("out", "a", "b"))
    out = k.buffer("out", shape=(2, 2))
    a = k.buffer("a", shape=(2, 2))
    b = k.buffer("b", shape=(2, 2))
    with k.parallel(2, 2) as (i, j):
      out[i, j] = a[i, j] + b[i, j]
    self.assertEqual(k.build().body, (
      Range("_p0_i0", 2, (
        Range("_p0_i1", 2, (
          Set(
            "out",
            Index2D(Var("_p0_i0"), Var("_p0_i1"), 2),
            Add(
              Load("a", Index2D(Var("_p0_i0"), Var("_p0_i1"), 2)),
              Load("b", Index2D(Var("_p0_i0"), Var("_p0_i1"), 2)),
            ),
          ),
        )),
      )),
    ))
  
  def test_builder_methods_accept_buffer_refs(self):
    k = KernelBuilder("copy", ("out", "inp"))
    out = k.buffer("out")
    inp = k.buffer("inp")
    with k.range("i", 4) as i: k.set(out, i, k.load(inp, i))
    self.assertEqual(k.build().body, (
      Range("i", 4, (
        Set("out", Var("i"), Load("inp", Var("i"))),
      )),
    ))

  def test_copy_accepts_buffer_refs(self):
    k = KernelBuilder("copy", ("out", "inp"))
    out = k.buffer("out")
    inp = k.buffer("inp")
    k.copy(inp, out, shape=(4,))
    self.assertEqual(k.build().body, (
      Range("_c0_i0", 4, (
        Store("out", "_c0_i0", Load("inp", "_c0_i0")),
      )),
    ))

if __name__ == "__main__":
  unittest.main()
