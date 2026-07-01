from tilegrad.ir import Add, Alloc, Arg, Barrier, Index2D, Kernel, Load, Range, Set, Store

class KernelBuilder:
  def __init__(self, name, args):
    self.name = name
    self.args = tuple(Arg(arg) for arg in args)
    self._body = []
    self._range_stack = []
    self._copy_counter = 0

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
    self._current_body().append(Barrier())
  
  def range(self, name, extent, axis="loop"): return _RangeContext(self, name, extent, axis)

  def copy(self, src, dst, shape, stride=None, src_row_off=0, src_col_off=0):
    n = self._copy_counter
    self._copy_counter += 1
    body = self._current_body()
    if len(shape) == 1:
      name_i0 = f"_c{n}_i0"
      if stride is None:
        src_idx = Add(src_col_off, name_i0) if src_col_off != 0 else name_i0
      else:
        src_row = Add(src_row_off, name_i0) if src_row_off != 0 else name_i0
        src_idx = Index2D(src_row, src_col_off, stride)
      body.append(Range(name_i0, shape[0], (
        Store(dst, name_i0, Load(src, src_idx)),
      )))
    elif len(shape) == 2:
      if stride is None: raise ValueError("stride required for 2D copy")
      name_i0 = f"_c{n}_i0"
      name_i1 = f"_c{n}_i1"
      src_row = Add(src_row_off, name_i0) if src_row_off != 0 else name_i0
      src_col = Add(src_col_off, name_i1) if src_col_off != 0 else name_i1
      body.append(Range(name_i0, shape[0], (
        Range(name_i1, shape[1], (
          Store(dst, Index2D(name_i0, name_i1, shape[1]), Load(src, Index2D(src_row, src_col, stride))),
        )),
      )))
    else: raise NotImplementedError(f"copy does not support {len(shape)}D")

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
