from tilegrad import tinygrad_compat as tg
from tilegrad.builder import KernelBuilder
from tilegrad.ir import Kernel
from tilegrad.lowerer import lower_kernel

def run(kernel, *tensors, realize: bool=True):
  if isinstance(kernel, KernelBuilder):
    if kernel._range_stack: raise ValueError("cannot run a KernelBuilder with open ranges")
    kernel = kernel.build()
  if not isinstance(kernel, Kernel): raise TypeError(f"expected Kernel or KernelBuilder, got {type(kernel).__name__}")
  if not tensors: raise ValueError("run requires at least one tensor (the output)")
  out, inputs = tensors[0], tensors[1:]
  if not isinstance(out, tg.Tensor): raise TypeError("first tensor must be the output Tensor")
  for i, t in enumerate(inputs):
    if not isinstance(t, tg.Tensor): raise TypeError(f"input tensor {i} is not a Tensor")
  def fxn(*uops): return lower_kernel(kernel, *uops)
  result = out.custom_kernel(*inputs, fxn=fxn)
  return result[0].realize() if realize else result[0]
