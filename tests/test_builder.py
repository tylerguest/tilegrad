import unittest
from tinygrad import Tensor
from tilegrad.builder import KernelBuilder
from tilegrad.ir import *
from tilegrad.ir import Arg, Range, FragmentGemm, FragmentClear, FragmentAlloc, FragmentStore, TileCopy, TileMMA, Alloc, Set
from tilegrad.lowerer import lower_kernel
from tilegrad.tiles import expand_tile_copies

class TestBuilder(unittest.TestCase):
  def test_builder_fragment_alloc_ir(self):
    k = KernelBuilder("fragment_alloc", ("out",))
    acc = k.fragment("acc", (2, 2), "float32")
    self.assertEqual(acc.name, "acc")
    self.assertEqual(acc.shape, (2, 2))
    self.assertEqual(acc.dtype, "float32")
    expected = Kernel(
      "fragment_alloc",
      (Arg("out"),),
      (FragmentAlloc("acc", (2, 2), "float32"),),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_fragment_clear_ir(self):
    k = KernelBuilder("fragment_clear", ("out",))
    acc = k.fragment("acc", (2, 2), "float32")
    with k.range("i", 1):
      k.clear(acc)
    expected = Kernel(
      "fragment_clear",
      (Arg("out"),),
      (
        FragmentAlloc("acc", (2, 2), "float32"),
        Range("i", 1, (FragmentClear("acc"),)),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_fragment_gemm_ir(self):
    k = KernelBuilder("fragment_gemm", ("out",))
    k.alloc("as", 6, "float32")
    k.alloc("bs", 6, "float32")
    as_tile = k.buffer("as", shape=(2, 3))
    bs_tile = k.buffer("bs", shape=(3, 2))
    acc = k.fragment("acc", (2, 2), "float32")
    with k.range("i", 1):
      k.gemm(as_tile, bs_tile, acc)
    expected = Kernel(
      "fragment_gemm",
      (Arg("out"),),
      (
        Alloc("as", 6, "float32", "shared"),
        Alloc("bs", 6, "float32", "shared"),
        FragmentAlloc("acc", (2, 2), "float32"),
        Range("i", 1, (FragmentGemm("as", "bs", "acc", (2, 3), (3, 2), (2, 2)),)),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_fragment_gemm_transpose_flags_ir(self):
    k = KernelBuilder("fragment_gemm_transpose", ("out",))
    k.alloc("as", 6, "float32")
    k.alloc("bs", 6, "float32")
    as_tile = k.buffer("as", shape=(3, 2))
    bs_tile = k.buffer("bs", shape=(2, 3))
    acc = k.fragment("acc", (2, 2), "float32")
    with k.range("i", 1):
      k.gemm(as_tile, bs_tile, acc, trans_a=True, trans_b=True)
    expected = Kernel(
      "fragment_gemm_transpose",
      (Arg("out"),),
      (
        Alloc("as", 6, "float32", "shared"),
        Alloc("bs", 6, "float32", "shared"),
        FragmentAlloc("acc", (2, 2), "float32"),
        Range("i", 1, (FragmentGemm("as", "bs", "acc", (3, 2), (2, 3), (2, 2), True, True),)),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_fragment_store_ir(self):
    k = KernelBuilder("fragment_store", ("out",))
    out = k.buffer("out", shape=(3, 3))
    acc = k.fragment("acc", (2, 2), "float32")
    with k.range("bi", 2) as bi:
      with k.range("bj", 2) as bj:
        k.store_fragment(acc, out, (bi * 2, bj * 2), guard=(bi < 2) & (bj < 2))
    expected = Kernel(
      "fragment_store",
      (Arg("out"),),
      (
        FragmentAlloc("acc", (2, 2), "float32"),
        Range("bi", 2, (
          Range("bj", 2, (
            FragmentStore("acc", "out", Mul(Var("bi"), 2), Mul(Var("bj"), 2), 3, And(Lt(Var("bi"), 2), Lt(Var("bj"), 2))),
          )),
        )),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_fragment_store_bounds_ir(self):
    k = KernelBuilder("fragment_store_bounds", ("out",))
    out = k.buffer("out", shape=(3, 3))
    acc = k.fragment("acc", (2, 2), "float32")
    with k.range("bi", 2) as bi:
      with k.range("bj", 2) as bj:
        k.store_fragment(acc, out, (bi * 2, bj * 2), bounds=(3, 3))
    expected = Kernel(
      "fragment_store_bounds",
      (Arg("out"),),
      (
        FragmentAlloc("acc", (2, 2), "float32"),
        Range("bi", 2, (
          Range("bj", 2, (
            FragmentStore("acc", "out", Mul(Var("bi"), 2), Mul(Var("bj"), 2), 3, None, (3, 3)),
          )),
        )),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_fragment_inside_range_fails(self):
    k = KernelBuilder("bad_fragment", ("out",))
    with k.range("i", 1):
      with self.assertRaisesRegex(ValueError, "fragment must be top-level"):
        k.fragment("acc", (2, 2), "float32")

  def test_fragment_shape_must_be_2d_tuple(self):
    k = KernelBuilder("bad_fragment_shape", ("out",))
    with self.assertRaisesRegex(ValueError, "fragment shape must be a 2D tuple"):
      k.fragment("acc", 4, "float32")

  def test_fragment_shape_must_be_positive_ints(self):
    k = KernelBuilder("bad_fragment_shape", ("out",))
    with self.assertRaisesRegex(ValueError, "fragment shape must contain positive integers"):
      k.fragment("acc", (2, 0), "float32")

  def test_gemm_requires_refs(self):
    k = KernelBuilder("bad_gemm", ("out",))
    acc = k.fragment("acc", (2, 2), "float32")
    with self.assertRaisesRegex(TypeError, "gemm A must be a buffer reference"):
      k.gemm("as", "bs", acc)

  def test_gemm_inputs_require_shapes(self):
    k = KernelBuilder("bad_gemm_shape", ("out",))
    a = k.buffer("as")
    b = k.buffer("bs", shape=(3, 2))
    acc = k.fragment("acc", (2, 2), "float32")
    with self.assertRaisesRegex(ValueError, "gemm inputs require shapes"):
      k.gemm(a, b, acc)

  def test_builder_tile_mma_ir(self):
    k = KernelBuilder("tile_mma", ("out",))
    as_tile = k.shared("as", shape=(2, 3), dtype="float32")
    bs_tile = k.shared("bs", shape=(3, 2), dtype="float32")
    acc = k.register("acc", shape=(2, 2), dtype="float32")
    with k.range("i", 1):
      k.gemm(as_tile, bs_tile, acc)
    expected = Kernel(
      "tile_mma",
      (Arg("out"),),
      (
        Alloc("as", 6, "float32", "shared"),
        Alloc("bs", 6, "float32", "shared"),
        Alloc("acc", 4, "float32", "register"),
        Range("i", 1, (TileMMA("as", "bs", "acc", (2, 3), (3, 2), (2, 2)),)),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_tile_mma_transpose_flags_ir(self):
    k = KernelBuilder("tile_mma_transpose", ("out",))
    as_tile = k.shared("as", shape=(3, 2), dtype="float32")
    bs_tile = k.shared("bs", shape=(2, 3), dtype="float32")
    acc = k.register("acc", shape=(2, 2), dtype="float32")
    with k.range("i", 1):
      k.gemm(as_tile, bs_tile, acc, trans_a=True, trans_b=True)
    expected = Kernel(
      "tile_mma_transpose",
      (Arg("out"),),
      (
        Alloc("as", 6, "float32", "shared"),
        Alloc("bs", 6, "float32", "shared"),
        Alloc("acc", 4, "float32", "register"),
        Range("i", 1, (TileMMA("as", "bs", "acc", (3, 2), (2, 3), (2, 2), True, True),)),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_clear_register_buffer_ir(self):
    k = KernelBuilder("clear_register_tile", ("out",))
    acc = k.register("acc", shape=(2, 2), dtype="float32")
    with k.range("i", 1):
      k.clear(acc)
    expected = Kernel(
      "clear_register_tile",
      (Arg("out"),),
      (
        Alloc("acc", 4, "float32", "register"),
        Range("i", 1, (
          Set("acc", 0, 0),
          Set("acc", 1, 0),
          Set("acc", 2, 0),
          Set("acc", 3, 0),
        )),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_store_fragment_requires_fragment_src(self):
    k = KernelBuilder("bad_store_fragment", ("out",))
    out = k.buffer("out", shape=(3, 3))
    with self.assertRaisesRegex(TypeError, "store_fragment src must be a fragment reference"):
      k.store_fragment("acc", out, (0, 0))

  def test_store_fragment_requires_2d_dst_shape(self):
    k = KernelBuilder("bad_store_fragment_dst", ("out",))
    out = k.buffer("out")
    acc = k.fragment("acc", (2, 2), "float32")
    with self.assertRaisesRegex(ValueError, "store_fragment dst must be a 2D buffer reference"):
      k.store_fragment(acc, out, (0, 0))

  def test_store_fragment_requires_2d_origin(self):
    k = KernelBuilder("bad_store_fragment_origin", ("out",))
    out = k.buffer("out", shape=(3, 3))
    acc = k.fragment("acc", (2, 2), "float32")
    with self.assertRaisesRegex(ValueError, "store_fragment dst_origin must be a 2D tuple"):
      k.store_fragment(acc, out, 0)

  def test_store_fragment_requires_2d_bounds(self):
    k = KernelBuilder("bad_store_fragment_bounds", ("out",))
    out = k.buffer("out", shape=(3, 3))
    acc = k.fragment("acc", (2, 2), "float32")
    with self.assertRaisesRegex(ValueError, "store_fragment bounds must be a 2D tuple"):
      k.store_fragment(acc, out, (0, 0), bounds=3)

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
    self.assertEqual(expand_tile_copies(k.build()), expected)
  
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
    self.assertEqual(expand_tile_copies(k.build()), expected)

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
    self.assertEqual(expand_tile_copies(k.build()), expected)

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
    self.assertEqual(expand_tile_copies(k.build()), expected)

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
    self.assertEqual(expand_tile_copies(k.build()), expected)

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
    self.assertEqual(expand_tile_copies(k.build()), expected)
  
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
  
  def test_buffer_ref_2d_indexing_uses_explicit_stride(self):
    k = KernelBuilder("copy2d_stride", ("out", "inp"))
    out = k.buffer("out", shape=(2,3), stride=5)
    inp = k.buffer("inp", shape=(2,3), stride=5)
    with k.range("i", 2) as i:
      with k.range("j", 3) as j:
        out[i, j] = inp[i, j]
    self.assertEqual(k.build().body, (
      Range("i", 2, (
        Range("j", 3, (
          Set("out", Index2D(Var("i"), Var("j"), 5), Load("inp", Index2D(Var("i"), Var("j"), 5))),
        )),
      )),
    ))
  
  def test_copy_tile_view_uses_explicit_buffer_stride(self):
    k = KernelBuilder("tile_copy_stride", ("out", "inp"))
    out = k.buffer("out", shape=(2,3), dtype="float32", stride=5)
    inp = k.buffer("inp", shape=(2,3), dtype="float32", stride=7)
    k.copy(inp.tile(), out.tile())
    self.assertEqual(k.build().body, (
      TileCopy(
        src="inp",
        dst="out",
        shape=(2,3),
        src_origin=(0,0),
        dst_origin=(0,0),
        src_stride=7,
        dst_stride=5,
        index_names=("_c0_i0", "_c0_i1"),
      ),
    ))

  def test_buffer_ref_3d_indexing(self):
    k = KernelBuilder("copy3d", ("out", "inp"))
    out = k.buffer("out", shape=(2, 3, 4))
    inp = k.buffer("inp", shape=(2, 3, 4))
    with k.range("b", 2) as b:
      with k.range("i", 3) as i:
        with k.range("j", 4) as j:
          out[b, i, j] = inp[b, i, j]
    flat = Add(Mul(Var("b"), 12), Add(Mul(Var("i"), 4), Var("j")))
    self.assertEqual(k.build().body, (
      Range("b", 2, (
        Range("i", 3, (
          Range("j", 4, (
            Set("out", flat, Load("inp", flat)),
          )),
        )),
      )),
    ))

  def test_builder_methods_accept_3d_buffer_refs(self):
    k = KernelBuilder("copy3d_methods", ("out", "inp"))
    out = k.buffer("out", shape=(2, 3, 4))
    inp = k.buffer("inp", shape=(2, 3, 4))
    with k.range("b", 2) as b:
      with k.range("i", 3) as i:
        with k.range("j", 4) as j:
          k.set(out, (b, i, j), k.load(inp, (b, i, j)))
    flat = Add(Mul(Var("b"), 12), Add(Mul(Var("i"), 4), Var("j")))
    self.assertEqual(k.build().body, (
      Range("b", 2, (
        Range("i", 3, (
          Range("j", 4, (
            Set("out", flat, Load("inp", flat)),
          )),
        )),
      )),
    ))

  def test_tuple_indexing_requires_shape(self):
    k = KernelBuilder("bad", ("out",))
    out = k.buffer("out")
    with self.assertRaisesRegex(ValueError, "tuple indexing requires shape"):
      _ = out[0, 0]

  def test_tuple_index_rank_mismatch_fails(self):
    k = KernelBuilder("bad_rank", ("out",))
    out = k.buffer("out", shape=(2, 3, 4))
    with self.assertRaisesRegex(ValueError, "2D index does not match 3D shape"):
      _ = out[0, 0]

  def test_buffer_ref_1d_tuple_indexing(self):
    k = KernelBuilder("copy1d_tuple", ("out", "inp"))
    out = k.buffer("out", shape=(4,))
    inp = k.buffer("inp", shape=(4,))
    with k.range("i", 4) as i:
      out[(i,)] = inp[(i,)]
    self.assertEqual(k.build().body, (
      Range("i", 4, (
        Set("out", Var("i"), Load("inp", Var("i"))),
      )),
    ))

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
      Range("_t0_i0", 2, (
        Range("_t0_i1", 2, (
          Set(
            "out",
            Index2D(Var("_t0_i0"), Var("_t0_i1"), 2),
            Add(
              Load("a", Index2D(Var("_t0_i0"), Var("_t0_i1"), 2)),
              Load("b", Index2D(Var("_t0_i0"), Var("_t0_i1"), 2)),
            ),
          ),
        ), "local"),
      ), "local"),
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
      TileCopy(
        src="inp",
        dst="out",
        shape=(4,),
        src_origin=(0,),
        dst_origin=(0,),
        index_names=("_c0_i0",),
      ),
    ))

  def test_set_if_ir(self):
    k = KernelBuilder("set_if", ("out", "inp"))
    out = k.buffer("out")
    inp = k.buffer("inp")
    with k.range("i", 4) as i: k.set_if(i < 3, out, i, inp[i])
    self.assertEqual(k.build().body, (
      Range("i", 4, (
        SetIf(Lt(Var("i"), 3), "out", Var("i"), Load("inp", Var("i"))),
      )),
    ))

  def test_set_if_accepts_buffer_ref_tuple_index(self):
    k = KernelBuilder("set_if_2d", ("out", "inp"))
    out = k.buffer("out", shape=(2, 3))
    inp = k.buffer("inp", shape=(2, 3))
    with k.range("i", 2) as i:
      with k.range("j", 3) as j:
        k.set_if(i < 1, out, (i, j), inp[i, j])
    self.assertEqual(k.build().body, (
      Range("i", 2, (
        Range("j", 3, (
          SetIf(
            Lt(Var("i"), 1),
            "out",
            Index2D(Var("i"), Var("j"), 3),
            Load("inp", Index2D(Var("i"), Var("j"), 3)),
          ),
        )),
      )),
    ))

  def test_load_if_ir(self):
    k = KernelBuilder("load_if", ("out", "inp"))
    out = k.buffer("out")
    inp = k.buffer("inp")
    with k.range("i", 5) as i: out[i] = k.load_if(i < 4, inp, i)
    self.assertEqual(k.build().body, (
      Range("i", 5, (
        Set("out", Var("i"), LoadIf(Lt(Var("i"), 4), "inp", Var("i"))),
      )),
    ))

  def test_load_if_accepts_2d_buffer_ref_index(self):
    k = KernelBuilder("load_if_2d", ("out", "inp"))
    out = k.buffer("out", shape=(4, 4))
    inp = k.buffer("inp", shape=(3, 2))
    with k.range("i", 4) as i:
      with k.range("j", 4) as j:
        out[i, j] = k.load_if((i < 3) & (j < 2), inp, (i, j))
    self.assertEqual(k.build().body, (
      Range("i", 4, (
        Range("j", 4, (
          Set(
            "out",
            Index2D(Var("i"), Var("j"), 4),
            LoadIf(And(Lt(Var("i"), 3), Lt(Var("j"), 2)), "inp", Index2D(Var("i"), Var("j"), 2)),
          ),
        )),
      )),
    ))

  def test_store_if_masked_copy_kernel(self):
    def copy_kernel(out, inp):
      k = KernelBuilder("store_if_masked_copy", ("out", "inp"))
      out_ref = k.buffer("out")
      inp_ref = k.buffer("inp")
      with k.range("i", 4) as i: k.store_if(i < 3, out_ref, i, inp_ref[i])
      return lower_kernel(k.build(), out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor([0.0, 0.0, 0.0, 9.0])
    out = out.custom_kernel(inp, fxn=copy_kernel)[0].realize()
    self.assertEqual(out.tolist(), [1.0, 2.0, 3.0, 9.0])

  def test_store_if_masks_out_of_bounds_store(self):
    def copy_kernel(out, inp):
      k = KernelBuilder("store_if_oob_masked_copy", ("out", "inp"))
      out_ref = k.buffer("out")
      inp_ref = k.buffer("inp")
      with k.range("i", 5) as i: k.store_if(i < 4, out_ref, i, inp_ref[i])
      return lower_kernel(k.build(), out, inp)
    inp = Tensor([1.0, 2.0, 3.0, 4.0, 99.0])
    out = Tensor.empty(4)
    out = out.custom_kernel(inp, fxn=copy_kernel)[0].realize()
    self.assertEqual(out.tolist(), [1.0, 2.0, 3.0, 4.0])

  def test_store_if_2d_edge_tile_vecadd_kernel(self):
    def vecadd_kernel(out, a, b):
      k = KernelBuilder("store_if_2d_edge_tile_vecadd", ("out", "a", "b"))
      out_ref = k.buffer("out", shape=(3, 2))
      a_ref = k.buffer("a", shape=(3, 2))
      b_ref = k.buffer("b", shape=(3, 2))
      with k.parallel(4, 4) as (i, j): k.store_if((i < 3) & (j < 2), out_ref, (i, j), a_ref[i, j] + b_ref[i, j])
      return lower_kernel(k.build(), out, a, b)
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = Tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    out = Tensor.empty(6)
    out = out.custom_kernel(a, b, fxn=vecadd_kernel)[0].realize()
    self.assertEqual(out.tolist(), [11.0, 22.0, 33.0, 44.0, 55.0, 66.0])

  def test_builder_grid_ir(self):
    k = KernelBuilder("grid", ("out",))
    with k.grid(2, 3) as (bx, by):
      k.store("out", Add(Mul(bx, 3), by), 1)
    expected = Kernel(
      "grid",
      (Arg("out"),),
      (
        Range("_g0_i0", 2, (
          Range("_g0_i1", 3, (
            Store("out", Add(Mul(Var("_g0_i0"), 3), Var("_g0_i1")), 1),
          ), "global"),
        ), "global"),
      ),
    )
    self.assertEqual(k.build(), expected)
  
  def test_builder_threads_ir(self):
    k = KernelBuilder("threads", ("out",))
    with k.threads(4) as tid:
      k.store("out", tid, 1)
    expected = Kernel(
      "threads",
      (Arg("out"),),
      (
        Range("_t0_i0", 4, (
          Store("out", Var("_t0_i0"), 1),
        ), "local"),
      ),
    )
    self.assertEqual(k.build(), expected)
  
  def test_builder_parallel_aliases_threads_ir(self):
    k = KernelBuilder("parallel", ("out",))
    with k.parallel(4) as tid:
      k.store("out", tid, 1)
    expected = Kernel(
      "parallel",
      (Arg("out"),),
      (
        Range("_t0_i0", 4, (
          Store("out", Var("_t0_i0"), 1),
        ), "local"),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_unroll_range_ir(self):
    k = KernelBuilder("unroll_range", ("out",))
    with k.range("u", 4, axis="unroll") as u:
      k.store("out", u, u)
    expected = Kernel(
      "unroll_range",
      (Arg("out"),),
      (
        Range("u", 4, (
          Store("out", Var("u"), Var("u")),
        ), "unroll"),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_builder_blocks_aliases_grid_ir(self):
    k = KernelBuilder("blocks", ("out",))
    with k.blocks(2, 3) as (bx, by):
      k.store("out", Add(Mul(bx, 3), by), 1)
    expected = Kernel(
      "blocks",
      (Arg("out"),),
      (
        Range("_g0_i0", 2, (
          Range("_g0_i1", 3, (
            Store("out", Add(Mul(Var("_g0_i0"), 3), Var("_g0_i1")), 1),
          ), "global"),
        ), "global"),
      ),
    )
    self.assertEqual(k.build(), expected)
  
  def test_empty_grid_fails(self):
    k = KernelBuilder("bad_empty_grid", ("out",))
    with self.assertRaisesRegex(ValueError, "axis context requires at least one extent"):
      k.grid()
  
  def test_empty_blocks_fails(self):
    k = KernelBuilder("bad_empty_blocks", ("out",))
    with self.assertRaisesRegex(ValueError, "axis context requires at least one extent"):
      k.blocks()

  def test_empty_threads_fails(self):
    k = KernelBuilder("bad_empty_threads", ("out",))
    with self.assertRaisesRegex(ValueError, "axis context requires at least one extent"):
      k.threads()
  
  def test_empty_parallel_fails(self):
    k = KernelBuilder("bad_empty_parallel", ("out",))
    with self.assertRaisesRegex(ValueError, "axis context requires at least one extent"):
      k.parallel()
  
  def test_builder_copy_1d_dst_origin_ir(self):
    k = KernelBuilder("copy_1d_dst_origin", ("out", "inp"))
    k.copy("inp", "out", shape=(3,), dst_origin=(1,))
    expected = Kernel(
      "copy_1d_dst_origin",
      (Arg("out"), Arg("inp")),
      (Range("_c0_i0", 3, (
        Store("out", Add(1, "_c0_i0"), Load("inp", "_c0_i0")),
      )),),
    )
    self.assertEqual(expand_tile_copies(k.build()), expected)
  
  def test_builder_copy_2d_origins_ir(self):
    k = KernelBuilder("copy_2d_origins", ("out", "inp"))
    k.copy("inp", "out", shape=(2, 3), src_origin=(1, 2), dst_origin=(3, 4), src_stride=8, dst_stride=10)
    expected = Kernel(
      "copy_2d_origins",
      (Arg("out"), Arg("inp")),
      (Range("_c0_i0", 2, (
        Range("_c0_i1", 3, (
          Store(
            "out",
            Index2D(Add(3, "_c0_i0"), Add(4, "_c0_i1"), 10),
            Load("inp", Index2D(Add(1, "_c0_i0"), Add(2, "_c0_i1"), 8)),
          ),
        )),
      )),),
    )
    self.assertEqual(expand_tile_copies(k.build()), expected)

  def test_builder_copy_3d_ir(self):
    k = KernelBuilder("copy_3d", ("out", "inp"))
    k.copy("inp", "out", shape=(2, 2, 3))
    tile = k.build()
    self.assertEqual(tile.body, (
      TileCopy(
        src="inp",
        dst="out",
        shape=(2, 2, 3),
        src_origin=(0, 0, 0),
        dst_origin=(0, 0, 0),
        index_names=("_c0_i0", "_c0_i1", "_c0_i2"),
      ),
    ))
    expected = Kernel(
      "copy_3d",
      (Arg("out"), Arg("inp")),
      (Range("_c0_i0", 2, (
        Range("_c0_i1", 2, (
          Range("_c0_i2", 3, (
            Store(
              "out",
              Add(Mul("_c0_i0", 6), Add(Mul("_c0_i1", 3), "_c0_i2")),
              Load("inp", Add(Mul("_c0_i0", 6), Add(Mul("_c0_i1", 3), "_c0_i2"))),
            ),
          )),
        )),
      )),),
    )
    self.assertEqual(expand_tile_copies(tile), expected)

  def test_builder_copy_infers_shape_from_dst_ref(self):
    k = KernelBuilder("copy_infer_shape", ("out", "inp"))
    out = k.buffer("out", shape=(4,))
    inp = k.buffer("inp")
    k.copy(inp, out)
    expected = Kernel(
      "copy_infer_shape",
      (Arg("out"), Arg("inp")),
      (Range("_c0_i0", 4, (
        Store("out", "_c0_i0", Load("inp", "_c0_i0")),
      )),),
    )
    self.assertEqual(expand_tile_copies(k.build()), expected)

  def test_builder_copy_guard_fill_zero_ir(self):
    k = KernelBuilder("copy_guard_fill_zero", ("out", "inp"))
    with k.range("base", 1) as base:
      k.copy("inp", "out", shape=(4,), src_origin=(base,), guard=Var("_c0_i0") < 3, fill=0)
    expected = Kernel(
      "copy_guard_fill_zero",
      (Arg("out"), Arg("inp")),
      (Range("base", 1, (
        Range("_c0_i0", 4, (
          Store("out", "_c0_i0", LoadIf(Lt(Var("_c0_i0"), 3), "inp", Add(Var("base"), "_c0_i0"))),
        )),
      )),),
    )
    self.assertEqual(expand_tile_copies(k.build()), expected)

  def test_builder_copy_nonzero_fill_fails(self):
    k = KernelBuilder("copy_bad_fill", ("out", "inp"))
    with self.assertRaisesRegex(NotImplementedError, "copy only supports fill=0"):
      k.copy("inp", "out", shape=(4,), guard=True, fill=1)

  def test_builder_pipelined_ir_matches_loop_range(self):
    k = KernelBuilder("pipelined", ("out",))
    with k.pipelined("ko", 2, stages=2) as ko:
      k.store("out", ko, ko)
    expected = Kernel(
      "pipelined",
      (Arg("out"),),
      (
        Range("ko", 2, (
          Store("out", Var("ko"), Var("ko")),
        )),
      ),
    )
    self.assertEqual(k.build(), expected)

  def test_pipelined_rejects_zero_stages(self):
    k = KernelBuilder("bad_pipelined", ("out",))
    with self.assertRaisesRegex(ValueError, "pipelined stages must be a positive integer"):
      k.pipelined("ko", 2, stages=0)

  def test_pipelined_rejects_non_int_stages(self):
    k = KernelBuilder("bad_pipelined", ("out",))
    with self.assertRaisesRegex(ValueError, "pipelined stages must be a positive integer"):
      k.pipelined("ko", 2, stages="2")
  
  def test_tile_view_constructs_metadata(self):
    k = KernelBuilder("tile_view", ("out", "inp"))
    inp = k.buffer("inp", shape=(4, 5), dtype="float32")
    tile = inp.tile(origin=(1,2), shape=(2,3), bounds=(4,5))
    self.assertEqual(tile.buffer, inp)
    self.assertEqual(tile.origin, (1,2))
    self.assertEqual(tile.shape, (2,3))
    self.assertEqual(tile.bounds, (4,5))
  
  def test_copy_accepts_tile_views_ir(self):
    k = KernelBuilder("tile_copy", ("out", "inp"))
    out = k.buffer("out", shape=(2, 3), dtype="float32")
    inp = k.buffer("inp", shape=(4, 5), dtype="float32")
    k.copy(inp.tile(origin=(1, 2), shape=(2, 3), bounds=(4, 5)), out.tile())
    expected = Kernel(
      "tile_copy",
      (Arg("out"), Arg("inp")),
      (
        TileCopy(
          src="inp",
          dst="out",
          shape=(2,3),
          src_origin=(1,2),
          dst_origin=(0,0),
          src_stride=5,
          dst_stride=3,
          src_bounds=(4,5),
          index_names=("_c0_i0", "_c0_i1"),
        ),
      ),
    )
    self.assertEqual(k.build(), expected)
  
  def test_tile_copy_expands_to_scalar_ir(self):
    k = KernelBuilder("tile_copy", ("out", "inp"))
    out = k.buffer("out", shape=(2, 3), dtype="float32")
    inp = k.buffer("inp", shape=(4, 5), dtype="float32")
    k.copy(inp.tile(origin=(1, 2), shape=(2, 3), bounds=(4, 5)), out.tile())
    expected = Kernel(
      "tile_copy",
      (Arg("out"), Arg("inp")),
      (
        Range("_c0_i0", 2, (
          Range("_c0_i1", 3, (
            Store(
              "out",
              Index2D("_c0_i0", "_c0_i1", 3),
              LoadIf(
                And(Lt(Add(1, Var("_c0_i0")), 4), Lt(Add(2, Var("_c0_i1")), 5)),
                "inp",
                Index2D(Add(1, "_c0_i0"), Add(2, "_c0_i1"), 5),
              ),
            ),
          )),
        )),
      ),
    )
    self.assertEqual(expand_tile_copies(k.build()), expected)
  
  def test_copy_tile_view_shape_mismatch_fails(self):
    k = KernelBuilder("tile_copy_bad_shape", ("out", "inp"))
    out = k.buffer("out", shape=(2,3), dtype="float32")
    inp = k.buffer("inp", shape=(3,2), dtype="float32")
    with self.assertRaisesRegex(ValueError, "tile copy shape mismatch"): k.copy(inp.tile(), out.tile())
  
  def test_copy_tile_view_explicit_shape_mismatch_fails(self):
    k = KernelBuilder("tile_copy_bad_explicit_shape", ("out", "inp"))
    out = k.buffer("out", shape=(2,2), dtype="float32")
    inp = k.buffer("inp", shape=(2,3), dtype="float32")
    with self.assertRaisesRegex(ValueError, "tile copy shape mismatch"): k.copy(inp.tile(), out.tile(), shape=(2,2))
  
  def test_copy_preserves_explicit_empty_shape_error(self):
    k = KernelBuilder("copy_empty_shape", ("out", "inp"))
    out = k.buffer("out", shape=(4,), dtype="float32")
    inp = k.buffer("inp", shape=(4,), dtype="float32")
    with self.assertRaisesRegex(ValueError, "copy shape must not be empty"): k.copy(inp, out, shape=())
      
if __name__ == "__main__":
  unittest.main()
