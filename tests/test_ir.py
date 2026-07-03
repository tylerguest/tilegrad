import unittest
from tilegrad.ir import Add, Alloc, And, Arg, Barrier, Const, FloorDiv, FragmentAlloc, FragmentClear, FragmentGemm, FragmentStore, Kernel, Load, LoadIf, Lt, Mod, Mul, Range, Store, BinaryExpr, Expr, KernelOp, Stmt, Set, SetIf, Sub, Index2D, Var, and_, lt
from tilegrad.utils import ceildiv_expr

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
    self.assertIsInstance(LoadIf(Lt(Var("i"), 4), "inp", Var("i")), Expr)
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
    self.assertIsInstance(SetIf(Lt(Var("i"), 3), "out", Var("i"), 1), Stmt)
    self.assertIsInstance(FragmentClear("acc"), Stmt)
    self.assertIsInstance(FragmentGemm("as", "bs", "acc", (2, 3), (3, 2), (2, 2)), Stmt)
    self.assertIsInstance(FragmentStore("acc", "out", 0, 0, 3), Stmt)
  
  def test_kernel_op_markers(self):
    self.assertIsInstance(Alloc("smem", 4, "float32", "shared"), KernelOp)
    self.assertIsInstance(FragmentAlloc("acc", (2, 2), "float32"), KernelOp)
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

  def test_set_if(self):
    stmt = SetIf(Lt(Var("i"), 3), "out", Var("i"), 1)
    self.assertEqual(stmt.cond, Lt(Var("i"), 3))
    self.assertEqual(stmt.buffer, "out")
    self.assertEqual(stmt.index, Var("i"))
    self.assertEqual(stmt.value, 1)
    self.assertIsInstance(stmt, Stmt)

  def test_load_if(self):
    expr = LoadIf(Lt(Var("i"), 4), "inp", Var("i"))
    self.assertEqual(expr.cond, Lt(Var("i"), 4))
    self.assertEqual(expr.buffer, "inp")
    self.assertEqual(expr.index, Var("i"))
    self.assertIsInstance(expr, Expr)

  def test_predicate_helpers(self):
    self.assertEqual(lt(Var("i"), 4), Lt(Var("i"), 4))
    self.assertEqual(and_(lt(Var("i"), 4), lt(Var("j"), 3)), And(Lt(Var("i"), 4), Lt(Var("j"), 3)))

  def test_predicate_operators(self):
    i = Var("i")
    j = Var("j")
    self.assertEqual((i < 4) & (j < 3), And(Lt(i, 4), Lt(j, 3)))

  def test_fragment_alloc(self):
    alloc = FragmentAlloc("acc", (2, 2), "float32")
    self.assertEqual(alloc.name, "acc")
    self.assertEqual(alloc.shape, (2, 2))
    self.assertEqual(alloc.dtype, "float32")
    self.assertIsInstance(alloc, KernelOp)

  def test_fragment_clear(self):
    stmt = FragmentClear("acc")
    self.assertEqual(stmt.buffer, "acc")
    self.assertIsInstance(stmt, Stmt)

  def test_fragment_gemm(self):
    stmt = FragmentGemm("as", "bs", "acc", (2, 3), (3, 2), (2, 2))
    self.assertEqual(stmt.a, "as")
    self.assertEqual(stmt.b, "bs")
    self.assertEqual(stmt.c, "acc")
    self.assertEqual(stmt.a_shape, (2, 3))
    self.assertEqual(stmt.b_shape, (3, 2))
    self.assertEqual(stmt.c_shape, (2, 2))
    self.assertFalse(stmt.trans_a)
    self.assertFalse(stmt.trans_b)
    self.assertIsInstance(stmt, Stmt)

  def test_fragment_gemm_transpose_flags(self):
    stmt = FragmentGemm("as", "bs", "acc", (3, 2), (2, 3), (2, 2), trans_a=True, trans_b=True)
    self.assertTrue(stmt.trans_a)
    self.assertTrue(stmt.trans_b)

  def test_fragment_store(self):
    guard = Lt(Var("i"), 3)
    stmt = FragmentStore("acc", "out", Var("i"), Var("j"), 3, guard)
    self.assertEqual(stmt.src, "acc")
    self.assertEqual(stmt.dst, "out")
    self.assertEqual(stmt.dst_row, Var("i"))
    self.assertEqual(stmt.dst_col, Var("j"))
    self.assertEqual(stmt.dst_stride, 3)
    self.assertEqual(stmt.guard, guard)
    self.assertIsInstance(stmt, Stmt)

  def test_ceildiv_expr_var_by_int(self):
    self.assertEqual(ceildiv_expr(Var("M"), 8), FloorDiv(Add(Var("M"), 7), 8))

  def test_ceildiv_expr_shape_string_by_int(self):
    self.assertEqual(ceildiv_expr("inp.shape.0", 8), FloorDiv(Add("inp.shape.0", 7), 8))

  def test_ceildiv_expr_requires_int_divisor(self):
    with self.assertRaisesRegex(TypeError, "ceildiv_expr divisor must be an int"):
      ceildiv_expr(Var("M"), Var("B"))

if __name__ == "__main__":
  unittest.main()
