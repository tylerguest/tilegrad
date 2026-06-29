from tilegrad.ir import Alloc, Arg, Barrier, Kernel, Load, Range, Set, Store

class KernelBuilder:
  def __init__(self, name, args):
    self.name = name
    self.args = tuple(Arg(arg) for arg in args)
    self._body = []
    self._range_stack = []

  def _current_body(self): return self._range_stack[-1] if self._range_stack else self._body 
  
  def load(self, buffer, index): return Load(buffer, index)

  def set(self, buffer, index, value):
    self._current_body().append(Set(buffer, index, value))

  def store(self, buffer, index, value):
    if not self._range_stack: raise ValueError("store requires an active range")
    self._current_body().append(Store(buffer, index, value))
  
  def alloc(self, name, shape, dtype, space="shared"):
    if self._range_stack: raise ValueError("alloc must be top-level")
    self._body.append(Alloc(name, shape, dtype, space))
  
  def barrier(self):
    if self._range_stack: raise ValueError("barrier must be top-level")
    self._body.append(Barrier())
  
  def range(self, name, extent, axis="loop"): return _RangeContext(self, name, extent, axis)

  def build(self): return Kernel(self.name, self.args, tuple(self._body))

class _RangeContext:
  def __init__(self, builder, name, extent, axis):
    self.builder = builder
    self.name = name
    self.extent = extent
    self.axis = axis

  def __enter__(self):
    self.builder._range_stack.append([])
    return self
  
  def __exit__(self, exc_type, exc, tb):
    body = self.builder._range_stack.pop()
    if exc_type is None: self.builder._current_body().append(Range(self.name, self.extent, tuple(body), self.axis))
    return False
