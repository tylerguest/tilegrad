import unittest
from tilegrad.ir import Add, Alloc, Arg, Barrier, Const, Kernel, Load, Mul, Range, Store

class TestIR(unittest.TestCase):
  def test_arg(self):
    self.assertEqual(Arg("out").name, "out")

  def test_load(self):
    load = Load("inp", "i")
    self.assertEqual(load.buffer, "inp")
    self.assertEqual(load.index, "i")
  
  def test_store(self):
    store = Store("out", "i", 0)
    self.assertEqual(store.buffer, "out")
    self.assertEqual(store.index, "i")
    self.assertEqual(store.value, 0)

  def test_range(self):
    rng = Range("i", "out.numel", ())
    self.assertEqual(rng.name, "i")
    self.assertEqual(rng.extent, "out.numel")
    self.assertEqual(rng.body, ())
  
  def test_alloc(self):
    alloc = Alloc("smem", "out.numel", "float32", "shared")
    self.assertEqual(alloc.name, "smem")
    self.assertEqual(alloc.shape, "out.numel")
    self.assertEqual(alloc.dtype, "float32")
    self.assertEqual(alloc.space, "shared")
  
  def test_barrier(self):
    self.assertIsInstance(Barrier(), Barrier)
  
  def test_kernel(self):
    body = (Range("i", "out.numel", (Store("out", "i", 0),)),)
    kernel = Kernel("test", (Arg("out"),), body)
    self.assertEqual(kernel.name, "test")
    self.assertEqual(kernel.args, (Arg("out"),))
    self.assertEqual(kernel.body, body)

  def test_const(self):
    const = Const(3)
    self.assertEqual(const.value, 3)
  
  def test_add(self):
    expr = Add("i", 1)
    self.assertEqual(expr.lhs, "i")
    self.assertEqual(expr.rhs, 1)
  
  def test_mul(self):
    expr = Mul("i", 2)
    self.assertEqual(expr.lhs, "i")
    self.assertEqual(expr.rhs, 2)

if __name__ == "__main__":
  unittest.main()
