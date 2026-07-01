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