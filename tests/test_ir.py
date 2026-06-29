import unittest
from tilegrad.ir import Add, Alloc, Arg, Barrier, Const, FloorDiv, Kernel, Load, Mod, Mul, Range, Store, BinaryExpr, Expr, KernelOp, Stmt, Set, Sub, Index2D

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
  
  def test_floor_div(self):
    expr = FloorDiv("i", 3)
    self.assertEqual(expr.lhs, "i")
    self.assertEqual(expr.rhs, 3)
  
  def test_mod(self):
    expr = Mod("i", 3)
    self.assertEqual(expr.lhs, "i")
    self.assertEqual(expr.rhs, 3)
  
  def test_expr_markers(self):
    self.assertIsInstance(Const(1), Expr)
    self.assertIsInstance(Add("i", 1), Expr)
    self.assertIsInstance(Mul("i", 2), Expr)
    self.assertIsInstance(FloorDiv("i", 3), Expr)
    self.assertIsInstance(Mod("i", 3), Expr)
    self.assertIsInstance(Load("inp", "i"), Expr)
    self.assertIsInstance(Sub("i", 1), Expr)
    self.assertIsInstance(Index2D("row", "col", 4), Expr)

  def test_binary_expr_marker(self):
    self.assertIsInstance(Add("i", 1), BinaryExpr)
    self.assertIsInstance(Mul("i", 2), BinaryExpr)
    self.assertIsInstance(FloorDiv("i", 3), BinaryExpr)
    self.assertIsInstance(Mod("i", 3), BinaryExpr)
    self.assertIsInstance(Sub("i", 1), BinaryExpr)

  def test_stmt_markers(self):
    self.assertIsInstance(Store("out", "i", 0), Stmt)
    self.assertIsInstance(Range("i", 4, ()), Stmt)
    self.assertIsInstance(Barrier(), Stmt)
    self.assertIsInstance(Set("out", 0, 1), Stmt)
  
  def test_kernel_op_markers(self):
    self.assertIsInstance(Alloc("smem", 4, "float32", "shared"), KernelOp)
    self.assertIsInstance(Range("i", 4, ()), KernelOp)
    self.assertIsInstance(Barrier(), KernelOp)
  
  def test_sub(self):
    expr = Sub("i", 1)
    self.assertEqual(expr.lhs, "i")
    self.assertEqual(expr.rhs, 1)
  
  def test_index2d(self):
    expr = Index2D("row", "col", 4)
    self.assertEqual(expr.row, "row")
    self.assertEqual(expr.col, "col")
    self.assertEqual(expr.stride, 4)
  
  def test_set(self):
    stmt = Set("out", 0, 1)
    self.assertEqual(stmt.buffer, "out")
    self.assertEqual(stmt.index, 0)
    self.assertEqual(stmt.value, 1)

if __name__ == "__main__":
  unittest.main()
