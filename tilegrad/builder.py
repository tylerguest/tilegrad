from tilegrad.ir import Add, Alloc, Arg, Barrier, Index2D, Kernel, Load, Mul, Range, Set, SetIf, Store, StoreIf, Var

class BufferRef:
  def __init__(self, builder, name, shape=None):
    self.builder = builder
    self.name = name
    self.shape = shape
  
  def _index(self, index):
    if not isinstance(index, tuple): return index
    if self.shape is None: raise ValueError("tuple indexing requires shape")
    return _flatten_nd_index(index, self.shape)
  
  def __getitem__(self, index): return Load(self.name, self._index(index))
  def __setitem__(self, index, value): self.builder.set(self.name, self._index(index), value)

def _buffer_name(x): return x.name if isinstance(x, BufferRef) else x
def _buffer_index(buffer, index): return buffer._index(index) if isinstance(buffer, BufferRef) else index

def _flatten_nd_index(indices, shape):
  if len(indices) != len(shape): raise ValueError(f"{len(indices)}D index does not match {len(shape)}D shape")
  if len(indices) == 1: return indices[0]
  if len(indices) == 2: return Index2D(indices[0], indices[1], shape[1])
  if not all(isinstance(dim, int) for dim in shape):
    raise TypeError("N-D tuple indexing requires integer buffer shapes")
  flat = indices[-1]
  stride = 1
  for idx, dim in zip(reversed(indices[:-1]), reversed(shape[1:])):
    stride *= dim
    flat = Add(Mul(idx, stride), flat)
  return flat

class KernelBuilder:
  def __init__(self, name, args):
    self.name = name
    self.args = tuple(Arg(arg) for arg in args)
    self._body = []
    self._range_stack = []
    self._copy_counter = 0
    self._parallel_counter = 0

  def _current_body(self): return self._range_stack[-1] if self._range_stack else self._body 

  def parallel(self, *extents): return _ParallelContext(self, extents)

  def buffer(self, name, shape=None): return BufferRef(self, name, shape)

  def buffers(self, *names): return tuple(self.buffer(name) for name in names)
  
  def load(self, buffer, index): return Load(_buffer_name(buffer), _buffer_index(buffer, index))

  def set(self, buffer, index, value): self._current_body().append(Set(_buffer_name(buffer), _buffer_index(buffer, index), value))

  def set_if(self, cond, buffer, index, value):
    self._current_body().append(SetIf(cond, _buffer_name(buffer), _buffer_index(buffer, index), value))

  def store(self, buffer, index, value):
    if not self._range_stack: raise ValueError("store requires an active range")
    self._current_body().append(Store(_buffer_name(buffer), _buffer_index(buffer, index), value))
  
  def store_if(self, cond, buffer, index, value):
    if not self._range_stack: raise ValueError("store_if requires an active range")
    self._current_body().append(StoreIf(cond, _buffer_name(buffer), _buffer_index(buffer, index), value))

  def alloc(self, name, shape, dtype, space="shared"):
    if self._range_stack: raise ValueError("alloc must be top-level")
    self._body.append(Alloc(_buffer_name(name), shape, dtype, space))

  def barrier(self): self._current_body().append(Barrier())
  
  def range(self, name, extent, axis="loop"): return _RangeContext(self, name, extent, axis)

  def copy(self, src, dst, shape, stride=None, src_row_off=0, src_col_off=0):
    src = _buffer_name(src)
    dst = _buffer_name(dst)
    n = self._copy_counter
    self._copy_counter += 1
    body = self._current_body()
    if len(shape) == 1:
      name_i0 = f"_c{n}_i0"
      if stride is None: src_idx = Add(src_col_off, name_i0) if src_col_off != 0 else name_i0
      else:
        src_row = Add(src_row_off, name_i0) if src_row_off != 0 else name_i0
        src_idx = Index2D(src_row, src_col_off, stride)
      body.append(Range(name_i0, shape[0], (Store(dst, name_i0, Load(src, src_idx)),)))
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
    return Var(self.name)
  
  def __exit__(self, exc_type, exc, tb):
    body = self.builder._range_stack.pop()
    if exc_type is None: self.builder._current_body().append(Range(self.name, self.extent, tuple(body), self.axis))
    return False

class _ParallelContext:
  def __init__(self, builder, extents):
    self.builder = builder
    self.extents = extents
    self.names = None
  
  def __enter__(self):
    n = self.builder._parallel_counter
    self.builder._parallel_counter += 1
    self.names = tuple(f"_p{n}_i{i}" for i in range(len(self.extents)))
    self.builder._range_stack.append([])
    vars = tuple(Var(name) for name in self.names)
    return vars[0] if len(vars) == 1 else vars
  
  def __exit__(self, exc_type, exc, tb):
    body = tuple(self.builder._range_stack.pop())
    if exc_type is None:
      for name, extent in reversed(tuple(zip(self.names, self.extents))):
        body = (Range(name, extent, body),)
      self.builder._current_body().extend(body)
    return False