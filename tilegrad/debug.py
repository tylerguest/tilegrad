from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from tilegrad import tinygrad_compat as tg
from tilegrad.builder import KernelBuilder
from tilegrad.ir import Kernel
from tilegrad.lowerer import lower_kernel, prepare_kernel_stages

@dataclass(frozen=True)
class IRStage:
  name: str
  kernel: Kernel

@dataclass(frozen=True)
class DebugArtifact:
  tile_ir: Kernel
  stages: tuple[IRStage, ...]
  scalar_ir: Kernel
  uops: tg.UOp | None = None

  def uops_text(self) -> str:
    if self.uops is None:
      raise ValueError("DebugArtifact has no lowered UOps; call inspect_kernel with tinygrad placeholder UOps")
    return uops_text(self.uops)

def _as_kernel(kernel):
  if isinstance(kernel, KernelBuilder):
    if kernel._range_stack: raise ValueError("cannot inspect a KernelBuilder with open ranges")
    return kernel.build()
  if isinstance(kernel, Kernel): return kernel
  raise TypeError(f"expected Kernel or KernelBuilder, got {type(kernel).__name__}")

def tile_ir(kernel) -> Kernel: return _as_kernel(kernel)

def ir_stages(kernel) -> tuple[IRStage, ...]: return tuple(IRStage(name, ir) for name, ir in prepare_kernel_stages(_as_kernel(kernel)))

def scalar_ir(kernel) -> Kernel: return ir_stages(kernel)[-1].kernel

def lowered_uops(kernel, *args: tg.UOp) -> tg.UOp: return lower_kernel(_as_kernel(kernel), *args)

def inspect_kernel(kernel, *args: tg.UOp) -> DebugArtifact:
  tile = _as_kernel(kernel)
  stages = ir_stages(tile)
  uops = lowered_uops(tile, *args) if args else None
  return DebugArtifact(tile, stages, stages[-1].kernel, uops)

def uops_text(uop: tg.UOp) -> str:
  if not isinstance(uop, tg.UOp): raise TypeError(f"expected UOp, got {type(uop).__name__}")
  out = StringIO()
  with redirect_stdout(out):
    tg.print_uops(list(uop.toposort()))
  return out.getvalue()
