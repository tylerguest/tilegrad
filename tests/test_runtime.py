import unittest
from tinygrad import Tensor
from tilegrad import KernelBuilder, run
from tilegrad.ir import Add, Index2D, Kernel, Mul

class TestRuntime(unittest.TestCase):
  def test_run_copy(self):
    k = KernelBuilder("copy", ("out", "inp"))
    with k.range("i", "out.numel"): k.store("out", "i", k.load("inp", "i"))
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(4)
    self.assertEqual(run(k, out, inp).tolist(), [1.0, 2.0, 3.0, 4.0])
  
  def test_run_zero_no_inputs(self):
    k = KernelBuilder("zero", ("out",))
    with k.range("i", "out.numel"): k.store("out", "i", 0)
    out = Tensor.empty(4)
    self.assertEqual(run(k, out).tolist(), [0.0, 0.0, 0.0, 0.0])
  
  def test_run_shared_copy(self):
    k = KernelBuilder("shared_copy", ("out", "inp"))
    k.alloc("smem", "out.numel", "float32")
    with k.range("i", "out.numel"): k.store("smem", "i", k.load("inp", "i"))
    k.barrier()
    with k.range("j", "out.numel"): k.store("out", "j", k.load("smem", "j"))
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(4)
    self.assertEqual(run(k, out, inp).tolist(), [1.0, 2.0, 3.0, 4.0])
  
  def test_run_naive_gemm_2x2(self):
    k = KernelBuilder("naive_gemm_2x2", ("out", "a", "b"))
    with k.range("i", 2):
      with k.range("j", 2):
        k.set("out", Index2D("i", "j", 2), 0)
        with k.range("k", 2, axis="reduce"):
          k.set(
            "out", Index2D("i", "j", 2),
            Add(
              k.load("out", Index2D("i", "j", 2)),
              Mul(k.load("a", Index2D("i", "k", 2)), k.load("b", Index2D("k", "j", 2))),
            ),
          )
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([5.0, 6.0, 7.0, 8.0])
    out = Tensor.empty(4)
    # A=[[1,2],[3,4]] B=[[5,6],[7,8]] -> C=[[19,22],[43,50]]
    self.assertEqual(run(k, out, a, b).tolist(), [19.0, 22.0, 43.0, 50.0])
  
  def test_run_set_if_preserves_false_branch(self):
    k = KernelBuilder("set_if_preserve", ("out", "inp"))
    out = k.buffer("out")
    inp = k.buffer("inp")
    with k.range("i", 4) as i: k.set_if(i < 3, out, i, inp[i])
    inp_t = Tensor([1.0, 2.0, 3.0, 4.0])
    out_t = Tensor([9.0, 9.0, 9.0, 9.0])
    self.assertEqual(run(k, out_t, inp_t).tolist(), [1.0, 2.0, 3.0, 9.0])

  def test_run_set_if_guarded_register_reduce(self):
    k = KernelBuilder("set_if_guarded_row_sum", ("out", "inp"))
    k.alloc("acc", 1, "float32", "register")
    out = k.buffer("out")
    inp = k.buffer("inp", shape=(2, 4))
    acc = k.buffer("acc")
    with k.range("i", 2) as i:
      acc[0] = 0
      with k.range("j", 4, axis="reduce") as j:
        k.set_if(j < 3, acc, 0, Add(acc[0], inp[i, j]))
      out[i] = acc[0]
    inp_t = Tensor([1.0, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0, 99.0])
    out_t = Tensor.empty(2)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [6.0, 15.0])

  def test_run_copy_3d(self):
    k = KernelBuilder("copy_3d", ("out", "inp"))
    out = k.buffer("out", shape=(2, 2, 3))
    inp = k.buffer("inp", shape=(2, 2, 3))
    with k.range("b", 2) as b:
      with k.range("i", 2) as i:
        with k.range("j", 3) as j:
          out[b, i, j] = inp[b, i, j]
    inp_t = Tensor([
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0,
      10.0, 11.0, 12.0,
    ])
    out_t = Tensor.empty(12)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0,
      10.0, 11.0, 12.0,
    ])

  def test_run_load_if_masks_oob_read(self):
    k = KernelBuilder("load_if_masked_copy", ("out", "inp"))
    out = k.buffer("out")
    inp = k.buffer("inp")
    with k.range("i", 5) as i:
      out[i] = k.load_if(i < 4, inp, i)
    inp_t = Tensor([1.0, 2.0, 3.0, 4.0])
    out_t = Tensor.empty(5)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [1.0, 2.0, 3.0, 4.0, 0.0])

  def test_run_load_if_guarded_sum_reduce(self):
    k = KernelBuilder("load_if_sum_reduce", ("out", "inp"))
    k.alloc("acc", 1, "float32", "register")
    k.set("acc", 0, 0)
    out = k.buffer("out")
    inp = k.buffer("inp")
    acc = k.buffer("acc")
    with k.range("i", 5, axis="reduce") as i:
      acc[0] = Add(acc[0], k.load_if(i < 4, inp, i))
    out[0] = acc[0]
    inp_t = Tensor([1.0, 2.0, 3.0, 4.0])
    out_t = Tensor.empty(1)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [10.0])

  def test_run_load_if_2d_edge_tile_zero_fills(self):
    k = KernelBuilder("load_if_2d_edge_tile", ("out", "inp"))
    out = k.buffer("out", shape=(4, 4))
    inp = k.buffer("inp", shape=(3, 2))
    with k.parallel(4, 4) as (i, j):
      out[i, j] = k.load_if((i < 3) & (j < 2), inp, (i, j))
    inp_t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out_t = Tensor.empty(16)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [
      1.0, 2.0, 0.0, 0.0,
      3.0, 4.0, 0.0, 0.0,
      5.0, 6.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
    ])

  def test_run_tiled_gemm_k_tail(self):
    k = KernelBuilder("tiled_gemm_k_tail", ("out", "a", "b"))
    k.alloc("as", 3, "float32")
    k.alloc("bs", 3, "float32")
    k.alloc("acc", 1, "float32", "register")
    out = k.buffer("out", shape=(2, 2))
    a = k.buffer("a", shape=(2, 5))
    b = k.buffer("b", shape=(5, 2))
    as_tile = k.buffer("as")
    bs_tile = k.buffer("bs")
    acc = k.buffer("acc")
    with k.range("i", 2) as i:
      with k.range("j", 2) as j:
        acc[0] = 0
        with k.range("ko", 2) as ko:
          with k.range("kk", 3) as kk:
            gk = ko * 3 + kk
            k.store(as_tile, kk, k.load_if(gk < 5, a, (i, gk)))
            k.store(bs_tile, kk, k.load_if(gk < 5, b, (gk, j)))
          k.barrier()
          with k.range("kk", 3, axis="reduce") as kk:
            acc[0] = acc[0] + as_tile[kk] * bs_tile[kk]
        out[i, j] = acc[0]
    a_t = Tensor([
      1.0, 2.0, 3.0, 4.0, 5.0,
      6.0, 7.0, 8.0, 9.0, 10.0,
    ])
    b_t = Tensor([
      11.0, 12.0,
      13.0, 14.0,
      15.0, 16.0,
      17.0, 18.0,
      19.0, 20.0,
    ])
    out_t = Tensor.empty(4)
    self.assertEqual(run(k, out_t, a_t, b_t).tolist(), [245.0, 260.0, 620.0, 660.0])

  def test_run_tiled_gemm_full_edge_tile(self):
    k = KernelBuilder("tiled_gemm_full_edge_tile", ("out", "a", "b"))
    k.alloc("as", 3, "float32")
    k.alloc("bs", 3, "float32")
    k.alloc("acc", 1, "float32", "register")
    out = k.buffer("out", shape=(3, 3))
    a = k.buffer("a", shape=(3, 5))
    b = k.buffer("b", shape=(5, 3))
    as_tile = k.buffer("as")
    bs_tile = k.buffer("bs")
    acc = k.buffer("acc")
    with k.range("i", 4) as i:
      with k.range("j", 4) as j:
        acc[0] = 0
        with k.range("ko", 2) as ko:
          with k.range("kk", 3) as kk:
            gk = ko * 3 + kk
            k.store(as_tile, kk, k.load_if((i < 3) & (gk < 5), a, (i, gk)))
            k.store(bs_tile, kk, k.load_if((gk < 5) & (j < 3), b, (gk, j)))
          k.barrier()
          with k.range("kk", 3, axis="reduce") as kk:
            acc[0] = acc[0] + as_tile[kk] * bs_tile[kk]
        k.store_if((i < 3) & (j < 3), out, (i, j), acc[0])
    a_t = Tensor([
      1.0, 2.0, 3.0, 4.0, 5.0,
      6.0, 7.0, 8.0, 9.0, 10.0,
      11.0, 12.0, 13.0, 14.0, 15.0,
    ])
    b_t = Tensor([
      16.0, 17.0, 18.0,
      19.0, 20.0, 21.0,
      22.0, 23.0, 24.0,
      25.0, 26.0, 27.0,
      28.0, 29.0, 30.0,
    ])
    out_t = Tensor.empty(9)
    self.assertEqual(run(k, out_t, a_t, b_t).tolist(), [
      360.0, 375.0, 390.0,
      910.0, 950.0, 990.0,
      1460.0, 1525.0, 1590.0,
    ])

  @unittest.skip("2D register accumulator indexed by loop vars (acc[ii, jj]) is not supported by tinygrad's PTX renderer (requires constant register indices)")
  def test_run_tiled_gemm_bm_bn_accum_tile(self):
    k = KernelBuilder("tiled_gemm_bm_bn_accum_tile", ("out", "a", "b"))
    k.alloc("as", 6, "float32")
    k.alloc("bs", 6, "float32")
    k.alloc("acc", 4, "float32", "register")
    out = k.buffer("out", shape=(3, 3))
    a = k.buffer("a", shape=(3, 5))
    b = k.buffer("b", shape=(5, 3))
    as_tile = k.buffer("as", shape=(2, 3))
    bs_tile = k.buffer("bs", shape=(3, 2))
    acc = k.buffer("acc", shape=(2, 2))
    with k.range("bi", 2) as bi:
      with k.range("bj", 2) as bj:
        with k.range("ko", 2) as ko:
          with k.range("ii", 2) as ii:
            gi = bi * 2 + ii
            with k.range("kk", 3) as kk:
              gk = ko * 3 + kk
              k.store(as_tile, (ii, kk), k.load_if((gi < 3) & (gk < 5), a, (gi, gk)))
          with k.range("kk", 3) as kk:
            gk = ko * 3 + kk
            with k.range("jj", 2) as jj:
              gj = bj * 2 + jj
              k.store(bs_tile, (kk, jj), k.load_if((gk < 5) & (gj < 3), b, (gk, gj)))
          k.barrier()
          with k.range("ii", 2) as ii:
            with k.range("jj", 2) as jj:
              k.set_if(ko < 1, acc, (ii, jj), 0)
              with k.range("kk", 3, axis="reduce") as kk:
                acc[ii, jj] = acc[ii, jj] + as_tile[ii, kk] * bs_tile[kk, jj]
        with k.range("ii", 2) as ii:
          gi = bi * 2 + ii
          with k.range("jj", 2) as jj:
            gj = bj * 2 + jj
            k.store_if((gi < 3) & (gj < 3), out, (gi, gj), acc[ii, jj])
    a_t = Tensor([
      1.0, 2.0, 3.0, 4.0, 5.0,
      6.0, 7.0, 8.0, 9.0, 10.0,
      11.0, 12.0, 13.0, 14.0, 15.0,
    ])
    b_t = Tensor([
      16.0, 17.0, 18.0,
      19.0, 20.0, 21.0,
      22.0, 23.0, 24.0,
      25.0, 26.0, 27.0,
      28.0, 29.0, 30.0,
    ])
    out_t = Tensor.empty(9)
    self.assertEqual(run(k, out_t, a_t, b_t).tolist(), [
      360.0, 375.0, 390.0,
      910.0, 950.0, 990.0,
      1460.0, 1525.0, 1590.0,
    ])

  def test_run_guarded_vecadd_2d(self):
    k = KernelBuilder("guarded_vecadd_2d", ("out", "a", "b"))
    out_ref = k.buffer("out", shape=(3, 2))
    a_ref = k.buffer("a", shape=(3, 2))
    b_ref = k.buffer("b", shape=(3, 2))
    with k.parallel(4, 4) as (i, j):
      k.store_if((i < 3) & (j < 2), out_ref, (i, j), a_ref[i, j] + b_ref[i, j])
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = Tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    out = Tensor.empty(6)
    self.assertEqual(run(k, out, a, b).tolist(), [11.0, 22.0, 33.0, 44.0, 55.0, 66.0])
  
  def test_run_accepts_built_kernel(self):
    k = KernelBuilder("copy", ("out", "inp"))
    with k.range("i", "out.numel"): k.store("out", "i", k.load("inp", "i"))
    ir = k.build()
    self.assertIsInstance(ir, Kernel)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(4)
    self.assertEqual(run(ir, out, inp).tolist(), [1.0, 2.0, 3.0, 4.0])
  def test_run_lazy_no_realize(self):
    k = KernelBuilder("copy", ("out", "inp"))
    with k.range("i", "out.numel"): k.store("out", "i", k.load("inp", "i"))
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor.empty(4)
    lazy = run(k, out, inp, realize=False)
    self.assertIsInstance(lazy, Tensor)
    self.assertEqual(lazy.realize().tolist(), [1.0, 2.0, 3.0, 4.0])
  def test_run_arg_count_mismatch_raises(self):
    k = KernelBuilder("copy", ("out", "inp"))
    with k.range("i", "out.numel"): k.store("out", "i", k.load("inp", "i"))
    out = Tensor.empty(4)
    with self.assertRaises(ValueError):
      run(k, out)  # missing inp -> lower_kernel sees 1 arg vs 2
  def test_run_rejects_bad_kernel_type(self):
    out = Tensor.empty(4)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    with self.assertRaises(TypeError):
      run("not a kernel", out, inp)
  def test_run_rejects_no_tensors(self):
    k = KernelBuilder("zero", ("out",))
    with k.range("i", "out.numel"): k.store("out", "i", 0)
    with self.assertRaises(ValueError):
      run(k)

if __name__ == "__main__":
  unittest.main()