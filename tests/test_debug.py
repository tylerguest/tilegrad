import unittest

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp

from tilegrad import KernelBuilder
from tilegrad.debug import DebugArtifact, inspect_kernel, ir_stages, lowered_uops, scalar_ir, tile_ir, uops_text
from tilegrad.ir import Range, TileCopy, TileMMA
from tilegrad.kernels import tiled_gemm


def has_tile_copy(body):
  for op in body:
    if isinstance(op, TileCopy): return True
    if isinstance(op, Range) and has_tile_copy(op.body): return True
  return False

def has_tile_mma(body):
  for op in body:
    if isinstance(op, TileMMA): return True
    if isinstance(op, Range) and has_tile_mma(op.body): return True
  return False

def find_tile_copy(body):
  for op in body:
    if isinstance(op, TileCopy): return op
    if isinstance(op, Range):
      found = find_tile_copy(op.body)
      if found is not None: return found
  return None

def tile_copy_builder():
  k = KernelBuilder("debug_tile_copy", ("out", "inp"))
  out = k.buffer("out", shape=(2, 3), dtype="float32")
  inp = k.buffer("inp", shape=(4, 5), dtype="float32")
  k.copy(inp.tile(origin=(1, 2), shape=(2, 3), bounds=(4, 5)), out.tile())
  return k

def shared_copy_builder():
  k = KernelBuilder("debug_shared_copy", ("out", "inp"))
  out = k.buffer("out", shape=(4,), dtype="float32")
  inp = k.buffer("inp", shape=(4,), dtype="float32")
  smem = k.shared("smem", shape=(4,), dtype="float32")
  k.copy(inp.tile(), smem.tile())
  k.barrier()
  k.copy(smem.tile(), out.tile())
  return k

class TestDebug(unittest.TestCase):
  def test_ir_stages_have_expected_names(self):
    stages = ir_stages(tile_copy_builder())
    self.assertEqual(
      [stage.name for stage in stages],
      ["tile_ir", "expand_tile_copies", "expand_fragments", "unroll_register_tiles", "scalar_ir"],
    )

  def test_inspect_kernel_returns_debug_artifact(self):
    dbg = inspect_kernel(tile_copy_builder())
    self.assertIsInstance(dbg, DebugArtifact)
    self.assertTrue(has_tile_copy(dbg.tile_ir.body))
    self.assertFalse(has_tile_copy(dbg.scalar_ir.body))
    self.assertIsNone(dbg.uops)

  def test_convenience_wrappers_return_tile_and_scalar_ir(self):
    self.assertTrue(has_tile_copy(tile_ir(tile_copy_builder()).body))
    self.assertFalse(has_tile_copy(scalar_ir(tile_copy_builder()).body))

  def test_lowered_uops_returns_sink(self):
    out = UOp.placeholder((6,), dtypes.float, slot=-1)
    inp = UOp.placeholder((20,), dtypes.float, slot=0)
    sink = lowered_uops(tile_copy_builder(), out, inp)
    self.assertIs(sink.op, Ops.SINK)

  def test_debug_artifact_uops_text(self):
    out = UOp.placeholder((6,), dtypes.float, slot=-1)
    inp = UOp.placeholder((20,), dtypes.float, slot=0)
    dbg = inspect_kernel(tile_copy_builder(), out, inp)
    self.assertIsNotNone(dbg.uops)
    text = dbg.uops_text()
    self.assertIn("Ops.SINK", text)
    self.assertIn("Ops.STORE", text)

  def test_uops_text_rejects_non_uop(self):
    with self.assertRaisesRegex(TypeError, "expected UOp"):
      uops_text("not a uop")

  def test_debug_artifact_uops_text_requires_lowered_uops(self):
    dbg = inspect_kernel(tile_copy_builder())
    with self.assertRaisesRegex(ValueError, "no lowered UOps"):
      dbg.uops_text()

  def test_shared_copy_debug_lowers(self):
    out = UOp.placeholder((4,), dtypes.float, slot=-1)
    inp = UOp.placeholder((4,), dtypes.float, slot=0)
    dbg = inspect_kernel(shared_copy_builder(), out, inp)
    self.assertIsNotNone(dbg.uops)
    self.assertIs(dbg.uops.op, Ops.SINK)

  def test_canonical_tiled_gemm_stages(self):
    dbg = inspect_kernel(tiled_gemm(3, 3, 5, BM=2, BN=2, BK=3))
    self.assertEqual(dbg.stages[0].name, "tile_ir")
    self.assertEqual(dbg.stages[-1].name, "scalar_ir")

  def test_canonical_tiled_gemm_preserves_tile_ops_before_scalar_ir(self):
    dbg = inspect_kernel(tiled_gemm(3, 3, 5, BM=2, BN=2, BK=3))
    self.assertTrue(has_tile_copy(dbg.tile_ir.body))
    self.assertTrue(has_tile_mma(dbg.tile_ir.body))
    self.assertFalse(has_tile_copy(dbg.scalar_ir.body))
    self.assertFalse(has_tile_mma(dbg.scalar_ir.body))

  def test_debug_preserves_tile_copy_coalesced_width(self):
    k = KernelBuilder("debug_coalesced_width", ("out", "inp"))
    out = k.buffer("out", shape=(4,), dtype="float32")
    inp = k.buffer("inp", shape=(4,), dtype="float32")
    k.copy(inp.tile(), out.tile(), coalesced_width=4)
    dbg = inspect_kernel(k)
    tile_copy = find_tile_copy(dbg.tile_ir.body)
    self.assertIsNotNone(tile_copy)
    self.assertEqual(tile_copy.coalesced_width, 4)
    self.assertFalse(has_tile_copy(dbg.scalar_ir.body))


if __name__ == "__main__":
  unittest.main()
