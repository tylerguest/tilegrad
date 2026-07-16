import unittest
from tinygrad import Tensor
from tilegrad import KernelBuilder, run
from tilegrad.ir import Add, Index2D, Kernel, Mul, Var
from tilegrad.kernels import grid_thread_fragment_gemm, tiled_gemm

def seq(n): return [float(i + 1) for i in range(n)]

def ref_matmul(a, b, M, N, K):
  return [
    sum(a[i * K + kk] * b[kk * N + j] for kk in range(K))
    for i in range(M)
    for j in range(N)
  ]

def run_tiled_gemm_case(M, N, K, BM=2, BN=2, BK=3):
  k = tiled_gemm(M, N, K, BM=BM, BN=BN, BK=BK)
  a_vals = seq(M * K)
  b_vals = seq(K * N)
  out = run(k, Tensor.empty(M * N), Tensor(a_vals), Tensor(b_vals)).tolist()
  return out, ref_matmul(a_vals, b_vals, M, N, K)

def run_grid_thread_fragment_gemm_case(M, N, K, BM=2, BN=2, BK=3):
  k = grid_thread_fragment_gemm(M, N, K, BM=BM, BN=BN, BK=BK)
  a_vals = seq(M * K)
  b_vals = seq(K * N)
  out = run(k, Tensor.empty(M * N), Tensor(a_vals), Tensor(b_vals)).tolist()
  return out, ref_matmul(a_vals, b_vals, M, N, K)

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
        for ko in range(2):
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
        for ko in range(2):
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

  def test_run_register_tile_unroll(self):
    k = KernelBuilder("register_tile_unroll", ("out",))
    k.alloc("acc", 4, "float32", "register")
    out = k.buffer("out", shape=(2, 2))
    acc = k.buffer("acc", shape=(2, 2))

    with k.range("ii", 2) as ii:
      with k.range("jj", 2) as jj:
        acc[ii, jj] = ii * 10 + jj

    with k.range("ii", 2) as ii:
      with k.range("jj", 2) as jj:
        out[ii, jj] = acc[ii, jj]

    out_t = Tensor.empty(4)
    self.assertEqual(run(k, out_t).tolist(), [0.0, 1.0, 10.0, 11.0])

  def test_run_tiled_gemm_bm_bn_accum_tile(self):
    k = KernelBuilder("tiled_gemm_bm_bn_accum_tile", ("out", "a", "b"))
    k.alloc("as", 6, "float32")
    k.alloc("bs", 6, "float32")
    out = k.buffer("out", shape=(3, 3))
    a = k.buffer("a", shape=(3, 5))
    b = k.buffer("b", shape=(5, 3))
    as_tile = k.buffer("as", shape=(2, 3))
    bs_tile = k.buffer("bs", shape=(3, 2))
    acc = k.fragment("acc", (2, 2), "float32")
    with k.range("bi", 2) as bi:
      with k.range("bj", 2) as bj:
        k.clear(acc)
        for ko in range(2):
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
          k.gemm(as_tile, bs_tile, acc)
        k.store_fragment(acc, out, (bi * 2, bj * 2), bounds=(3, 3))
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

  def test_run_fragment_store_bounds_2x2_into_3x3_edge(self):
    k = KernelBuilder("fragment_store_bounds_2x2_into_3x3_edge", ("out",))
    out = k.buffer("out", shape=(3, 3))
    acc = k.fragment("acc", (2, 2), "float32")
    k.clear(acc)
    k.store_fragment(acc, out, (2, 2), bounds=(3, 3))
    out_t = Tensor([9.0] * 9)
    result = run(k, out_t).tolist()
    self.assertEqual(result, [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.0])

  def test_run_tiled_gemm_bm_bn_accum_fragment(self):
    k = KernelBuilder("tiled_gemm_bm_bn_accum_fragment", ("out", "a", "b"))
    k.alloc("as", 6, "float32")
    k.alloc("bs", 6, "float32")
    out = k.buffer("out", shape=(3, 3))
    a = k.buffer("a", shape=(3, 5))
    b = k.buffer("b", shape=(5, 3))
    as_tile = k.buffer("as", shape=(2, 3))
    bs_tile = k.buffer("bs", shape=(3, 2))
    acc = k.fragment("acc", (2, 2), "float32")

    with k.range("bi", 2) as bi:
      with k.range("bj", 2) as bj:
        k.clear(acc)
        for ko in range(2):
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
          k.gemm(as_tile, bs_tile, acc)
        k.store_fragment(acc, out, (bi * 2, bj * 2), bounds=(3, 3))

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

  def test_run_fragment_clear_store_2x2(self):
    k = KernelBuilder("fragment_clear_store_2x2", ("out",))
    out = k.buffer("out", shape=(2, 2))
    acc = k.fragment("acc", (2, 2), "float32")
    k.clear(acc)
    k.store_fragment(acc, out, (0, 0))
    out_t = Tensor.empty(4)
    self.assertEqual(run(k, out_t).tolist(), [0.0, 0.0, 0.0, 0.0])

  def test_run_fragment_gemm_2x2x3(self):
    k = KernelBuilder("fragment_gemm_2x2x3", ("out", "a", "b"))
    k.alloc("as", 6, "float32")
    k.alloc("bs", 6, "float32")
    out = k.buffer("out", shape=(2, 2))
    a = k.buffer("a", shape=(2, 3))
    b = k.buffer("b", shape=(3, 2))
    as_tile = k.buffer("as", shape=(2, 3))
    bs_tile = k.buffer("bs", shape=(3, 2))
    acc = k.fragment("acc", (2, 2), "float32")
    k.copy(a, as_tile, shape=(2, 3), stride=3)
    k.copy(b, bs_tile, shape=(3, 2), stride=2)
    k.barrier()
    k.clear(acc)
    k.gemm(as_tile, bs_tile, acc)
    k.store_fragment(acc, out, (0, 0))
    a_t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b_t = Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    out_t = Tensor.empty(4)
    self.assertEqual(run(k, out_t, a_t, b_t).tolist(), [58.0, 64.0, 139.0, 154.0])

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

  def test_run_threads_vecadd(self):
    k = KernelBuilder("threads_vecadd", ("out", "a", "b"))
    out = k.buffer("out")
    a = k.buffer("a")
    b = k.buffer("b")
    with k.threads(4) as i: out[i] = a[i] + b[i]
    a_t = Tensor([1.0, 2.0, 3.0, 4.0])
    b_t = Tensor([10.0, 20.0, 30.0, 40.0])
    out_t = Tensor.empty(4)
    self.assertEqual(run(k, out_t, a_t, b_t).tolist(), [11.0, 22.0, 33.0, 44.0])
  
  def test_run_grid_threads_copy(self):
    k = KernelBuilder("grid_threads_copy", ("out", "inp"))
    out = k.buffer("out")
    inp = k.buffer("inp")
    with k.grid(2) as block:
      with k.threads(4) as tid:
        i = block * 4 + tid
        out[i] = inp[i]
    inp_t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    out_t = Tensor.empty(8)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
  
  def test_run_unroll_range(self):
    k = KernelBuilder("unroll_range", ("out", "inp"))
    out = k.buffer("out")
    inp = k.buffer("inp")

    with k.range("u", 4, axis="unroll") as u:
      out[u] = inp[u]
    
    inp_t = Tensor([0.0, 1.0, 2.0, 3.0])
    out_t = Tensor.empty(4)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [0.0, 1.0, 2.0, 3.0])

  def test_run_grid_threads_unroll_fill(self):
    k = KernelBuilder("grid_threads_unroll_fill", ("out", "inp"))
    out = k.buffer("out")
    inp = k.buffer("inp")

    with k.grid(2) as block:
      with k.threads(2) as tid:
        base = block * 4 + tid * 2
        with k.range("u", 2, axis="unroll") as u:
          i = base + u 
          out[i] = inp[i]
    
    inp_t = Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    out_t = Tensor.empty(8)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

  def test_run_copy_1d_dst_origin(self):
    k = KernelBuilder("copy_1d_dst_origin", ("out", "inp"))
    k.copy("inp", "out", shape=(3,), dst_origin=(1,))
    out = Tensor([0.0, 0.0, 0.0, 0.0])
    inp = Tensor([1.0, 2.0, 3.0])
    self.assertEqual(run(k, out, inp).tolist(), [0.0, 1.0, 2.0, 3.0])

  def test_run_copy_2d_src_origin(self):
    k = KernelBuilder("copy_2d_src_origin", ("out", "inp"))
    k.copy("inp", "out", shape=(2, 2), src_origin=(1, 1), src_stride=4, dst_stride=2)
    inp = Tensor([
      1.0, 2.0, 3.0, 4.0,
      5.0, 6.0, 7.0, 8.0,
      9.0, 10.0, 11.0, 12.0,
    ])
    out = Tensor.empty(4)
    self.assertEqual(run(k, out, inp).tolist(), [6.0, 7.0, 10.0, 11.0])

  def test_run_copy_2d_dst_origin(self):
    k = KernelBuilder("copy_2d_dst_origin", ("out", "inp"))
    k.copy("inp", "out", shape=(2, 2), src_stride=2, dst_origin=(1, 1), dst_stride=4)
    inp = Tensor([1.0, 2.0, 3.0, 4.0])
    out = Tensor([0.0] * 12)
    self.assertEqual(run(k, out, inp).tolist(), [
      0.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 2.0, 0.0,
      0.0, 3.0, 4.0, 0.0,
    ])

  def test_run_copy_guard_fill_zero(self):
    k = KernelBuilder("copy_guard_fill_zero", ("out", "inp"))
    with k.range("i", 1):
      k.copy("inp", "out", shape=(4,), guard=Var("_c0_i0") < 3, fill=0)
    inp = Tensor([1.0, 2.0, 3.0, 99.0])
    out = Tensor.empty(4)
    self.assertEqual(run(k, out, inp).tolist(), [1.0, 2.0, 3.0, 0.0])

  def test_run_copy_3d(self):
    k = KernelBuilder("copy_3d", ("out", "inp"))
    out = k.buffer("out", shape=(2, 2, 3))
    inp = k.buffer("inp", shape=(2, 2, 3))
    k.copy(inp, out)
    inp_t = Tensor([
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0,
      10.0, 11.0, 12.0,
    ])
    out_t = Tensor.empty(12)
    self.assertEqual(run(k, out_t, inp_t).tolist(), inp_t.tolist())

  def test_run_canonical_tiled_gemm_exact_tile(self):
    out, expected = run_tiled_gemm_case(2, 2, 3, BM=2, BN=2, BK=3)
    self.assertEqual(out, expected)

  def test_run_canonical_tiled_gemm_mn_edge(self):
    out, expected = run_tiled_gemm_case(3, 3, 3, BM=2, BN=2, BK=3)
    self.assertEqual(out, expected)
  
  def test_run_canonical_tiled_gemm_k_tail(self):
    out, expected = run_tiled_gemm_case(2, 2, 5, BM=2, BN=2, BK=3)
    self.assertEqual(out, expected)
  
  def test_run_canonical_tiled_gemm_non_square(self):
    out, expected = run_tiled_gemm_case(3, 4, 5, BM=2, BN=2, BK=3)
    self.assertEqual(out, expected)
  
  def test_run_canonical_tiled_gemm_flagship(self):
    M, N, K = 3, 3, 5
    k = tiled_gemm(M, N, K, BM=2, BN=2, BK=3)
    a_vals = seq(M * K)
    b_vals = [float(i + 16) for i in range(K * N)]
    out = run(k, Tensor.empty(M * N), Tensor(a_vals), Tensor(b_vals)).tolist()
    self.assertEqual(out, [
      360.0, 375.0, 390.0,
      910.0, 950.0, 990.0,
      1460.0, 1525.0, 1590.0,
    ])

  def test_run_grid_thread_fragment_gemm_exact_tile(self):
    out, expected = run_grid_thread_fragment_gemm_case(2, 2, 3, BM=2, BN=2, BK=3)
    self.assertEqual(out, expected)

  def test_run_grid_thread_fragment_gemm_mn_edge(self):
    out, expected = run_grid_thread_fragment_gemm_case(3, 3, 3, BM=2, BN=2, BK=3)
    self.assertEqual(out, expected)

  def test_run_grid_thread_fragment_gemm_k_tail(self):
    out, expected = run_grid_thread_fragment_gemm_case(2, 2, 5, BM=2, BN=2, BK=3)
    self.assertEqual(out, expected)

  def test_run_grid_thread_fragment_gemm_non_square(self):
    out, expected = run_grid_thread_fragment_gemm_case(3, 4, 5, BM=2, BN=2, BK=3)
    self.assertEqual(out, expected)

  def test_run_grid_thread_fragment_gemm_flagship(self):
    M, N, K = 3, 3, 5
    k = grid_thread_fragment_gemm(M, N, K, BM=2, BN=2, BK=3)
    a_vals = seq(M * K)
    b_vals = [float(i + 16) for i in range(K * N)]
    out = run(k, Tensor.empty(M * N), Tensor(a_vals), Tensor(b_vals)).tolist()
    self.assertEqual(out, [
      360.0, 375.0, 390.0,
      910.0, 950.0, 990.0,
      1460.0, 1525.0, 1590.0,
    ])

  def test_run_pipelined_copy(self):
    k = KernelBuilder("pipelined_copy", ("out", "inp"))
    out = k.buffer("out")
    inp = k.buffer("inp")
    with k.pipelined("i", 4, stages=2) as i:
      out[i] = inp[i]
    inp_t = Tensor([1.0, 2.0, 3.0, 4.0])
    out_t = Tensor.empty(4)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [1.0, 2.0, 3.0, 4.0])

  def test_run_tile_view_copy_2d_src_origin(self):
    k = KernelBuilder("tile_view_copy_2d_src_origin", ("out", "inp"))
    out = k.buffer("out", shape=(2,2), dtype="float32")
    inp = k.buffer("inp", shape=(3,4), dtype="float32")
    k.copy(inp.tile(origin=(1,1), shape=(2,2), bounds=(3,4)), out.tile())
    inp_t = Tensor([
      1.0, 2.0, 3.0, 4.0,
      5.0, 6.0, 7.0, 8.0,
      9.0, 10.0, 11.0, 12.0,
    ])
    out_t = Tensor.empty(4)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [6.0, 7.0, 10.0, 11.0])
  
  def test_run_tile_view_copy_honors_explicit_strides(self):
    k = KernelBuilder("tile_view_copy_explicit_stride", ("out", "inp"))
    out = k.buffer("out", shape=(2,3), dtype="float32", stride=5)
    inp = k.buffer("inp", shape=(2,3), dtype="float32", stride=7)
    k.copy(inp.tile(), out.tile())
    inp_t = Tensor([
      1.0, 2.0, 3.0, 99.0, 99.0, 99.0, 99.0,
      4.0, 5.0, 6.0, 99.0, 99.0, 99.0, 99.0,
    ])
    out_t = Tensor([0.0] * 10)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [
      1.0, 2.0, 3.0, 0.0, 0.0,
      4.0, 5.0, 6.0, 0.0, 0.0,
    ])
  
  def test_run_tile_view_copy_src_bounds_zero_fills(self):
    k = KernelBuilder("tile_view_copy_src_bounds_zero_fill", ("out", "inp"))
    out = k.buffer("out", shape=(2,2), dtype="float32")
    inp = k.buffer("inp", shape=(3,4), dtype="float32")
    k.copy(inp.tile(origin=(2,2), shape=(2,2), bounds=(3,4)), out.tile())
    inp_t = Tensor([
      1.0, 2.0, 3.0, 4.0,
      5.0, 6.0, 7.0, 8.0,
      9.0, 10.0, 11.0, 12.0,
    ])
    out_t = Tensor.empty(4)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [11.0, 12.0, 0.0, 0.0])
  
  def test_run_tile_view_copy_src_mask_zero_fills(self):
    k = KernelBuilder("tile_view_copy_src_mask_zero_fill", ("out", "inp"))
    out = k.buffer("out", shape=(4,), dtype="float32")
    inp = k.buffer("inp", shape=(4,), dtype="float32")
    k.copy(inp.tile(mask=Var("_c0_i0") < 3), out.tile())
    inp_t = Tensor([1.0, 2.0, 3.0, 99.0])
    out_t = Tensor.empty(4)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [1.0, 2.0, 3.0, 0.0])
  
  def test_run_tile_view_copy_dst_bounds_guards_store(self):
    k = KernelBuilder("tile_view_copy_dst_bounds_guard", ("out", "inp"))
    out = k.buffer("out", shape=(3,3), dtype="float32")
    inp = k.buffer("inp", shape=(2,2), dtype="float32")
    k.copy(inp.tile(), out.tile(origin=(2,2), shape=(2,2), bounds=(3,3)))
    inp_t = Tensor([1.0, 2.0, 3.0, 4.0])
    out_t = Tensor([9.0] * 9)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [
      9.0, 9.0, 9.0,
      9.0, 9.0, 9.0,
      9.0, 9.0, 1.0,
    ])
  
  def test_run_tile_view_copy_dst_mask_guards_store(self):
    k = KernelBuilder("tile_view_copy_dst_mask_guard", ("out", "inp"))
    out = k.buffer("out", shape=(4,), dtype="float32")
    inp = k.buffer("inp", shape=(4,), dtype="float32")
    k.copy(inp.tile(), out.tile(mask=Var("_c0_i0") < 3))
    inp_t = Tensor([1.0, 2.0, 3.0, 4.0])
    out_t = Tensor([9.0, 9.0, 9.0, 9.0])
    self.assertEqual(run(k, out_t, inp_t).tolist(), [1.0, 2.0, 3.0, 9.0])

  def test_run_tile_view_copy_coalesced_width_metadata(self):
    k = KernelBuilder("tile_view_copy_coalesced_width", ("out", "inp"))
    out = k.buffer("out", shape=(4,), dtype="float32")
    inp = k.buffer("inp", shape=(4,), dtype="float32")
    k.copy(inp.tile(), out.tile(), coalesced_width=4)
    inp_t = Tensor([1.0, 2.0, 3.0, 4.0])
    out_t = Tensor.empty(4)
    self.assertEqual(run(k, out_t, inp_t).tolist(), [1.0, 2.0, 3.0, 4.0])

if __name__ == "__main__":
  unittest.main()
