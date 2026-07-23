import unittest
from unittest.mock import patch

from tilegrad import KernelBuilder, run
from tilegrad import tinygrad_compat as tg
from tilegrad.lowerer import lower_kernel


def portable_copy_kernel():
  k = KernelBuilder("compatibility_copy", ("out", "inp"))
  out = k.buffer("out", shape=(4,), dtype="float32")
  inp = k.buffer("inp", shape=(4,), dtype="float32")
  k.copy(inp, out)
  return k


class TestTinygradCompatibility(unittest.TestCase):
  def test_current_tinygrad_satisfies_contract(self):
    self.assertIsNone(tg.check_tinygrad_compatibility())

  def test_missing_required_api_has_clear_error(self):
    with patch.object(tg.Tensor, "custom_kernel", None):
      with self.assertRaisesRegex(tg.TinygradCompatibilityError, "Tensor.custom_kernel"):
        tg.check_tinygrad_compatibility()

  def test_missing_composition_api_has_clear_error(self):
    for name in ("TinyJit", "function"):
      with self.subTest(name=name):
        with patch.object(tg, name, None):
          with self.assertRaisesRegex(tg.TinygradCompatibilityError, name):
            tg.check_tinygrad_compatibility()

  def test_kernel_info_disables_additional_opts(self):
    info = tg.kernel_info("compatibility_contract")
    self.assertEqual(info.name, "compatibility_contract")
    self.assertEqual(info.opts_to_apply, ())

  def test_schedule_storage_and_barrier_contract(self):
    k = KernelBuilder("compatibility_schedule", ("out", "inp"))
    out = k.buffer("out", shape=(8,), dtype="float32")
    inp = k.buffer("inp", shape=(8,), dtype="float32")
    smem = k.shared("smem", shape=(4,), dtype="float32")
    acc = k.register("acc", shape=(1,), dtype="float32")

    with k.grid(2) as block:
      with k.threads(4) as tid:
        i = block * 4 + tid
        acc[0] = inp[i]
        k.store(smem, tid, acc[0])
        k.barrier()
        k.store(out, i, smem[tid])

    out_uop = tg.placeholder((8,), tg.dtypes.float32, slot=-1)
    inp_uop = tg.placeholder((8,), tg.dtypes.float32, slot=0)
    sink = lower_kernel(k.build(), out_uop, inp_uop)
    uops = sink.toposort()

    axis_types = {u.arg[1] for u in uops if u.op is tg.Ops.RANGE}
    buffer_spaces = {u.addrspace for u in uops if u.op is tg.Ops.BUFFER}

    self.assertIs(sink.op, tg.Ops.SINK)
    self.assertIsInstance(sink.arg, tg.KernelInfo)
    self.assertEqual(sink.arg.opts_to_apply, ())
    self.assertIn(tg.AxisType.GLOBAL, axis_types)
    self.assertIn(tg.AxisType.LOCAL, axis_types)
    self.assertIn(tg.AddrSpace.LOCAL, buffer_spaces)
    self.assertIn(tg.AddrSpace.REG, buffer_spaces)
    self.assertTrue(any(u.op is tg.Ops.BARRIER for u in uops))

  def test_custom_kernel_executes_portable_copy(self):
    result = run(
      portable_copy_kernel(),
      tg.Tensor.empty(4),
      tg.Tensor([1.0, 2.0, 3.0, 4.0]),
    )
    self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 4.0])

  def test_enclosing_tinyjit_captures_and_replays(self):
    kernel = portable_copy_kernel()

    @tg.TinyJit
    def jit_copy(inp):
      out = tg.Tensor.empty(*inp.shape, dtype=inp.dtype, device=inp.device)
      return run(kernel, out, inp)

    cases = (
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
    )
    for values in cases:
      inp = tg.Tensor(values).contiguous().realize()
      self.assertEqual(jit_copy(inp).tolist(), values)

  def test_precompiled_function_executes_custom_kernel(self):
    kernel = portable_copy_kernel()

    @tg.function(precompile=True)
    def precompiled_copy(inp: tg.Tensor) -> tg.Tensor:
      out = tg.Tensor.invalids(*inp.shape, dtype=inp.dtype, device=inp.device)
      return run(kernel, out, inp, realize=False)

    inp = tg.Tensor([1.0, 2.0, 3.0, 4.0]).contiguous().realize()
    result = precompiled_copy(inp).realize()
    self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
  unittest.main()
