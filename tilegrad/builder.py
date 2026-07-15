from dataclasses import dataclass
from tilegrad.ir import Add, Alloc, Arg, Barrier, FragmentAlloc, FragmentClear, FragmentGemm, FragmentStore, Index2D, Kernel, Load, LoadIf, Mul, Not, Range, Set, SetIf, Store, StoreIf, TileCopy, Var

@dataclass(frozen=True)
class TileView:
  buffer: object
  origin: tuple
  shape: tuple
  stride: object = None
  bounds: tuple | None = None
  mask: object | None = None
  layout: object | None = None

class BufferRef:
  def __init__(self, builder, name, shape=None, dtype=None, stride=None, scope="global"):
    self.builder = builder
    self.name = name
    self.shape = shape
    self.dtype = dtype
    self.stride = stride
    self.scope = scope
  
  def _index(self, index):
    if not isinstance(index, tuple): return index
    if self.shape is None: raise ValueError("tuple indexing requires shape")
    return _flatten_nd_index(index, self.shape)
  
  def tile(self, origin=None, shape=None, stride=None, bounds=None, mask=None, layout=None):
    shape = self.shape if shape is None else shape
    if shape is None: raise ValueError("tile shape is required unless buffer shape is set")
    if not isinstance(shape, tuple): raise TypeError("tile shape must be a tuple")
    origin = tuple(0 for _ in shape) if origin is None else origin
    if not isinstance(origin, tuple): origin = (origin,)
    if len(origin) != len(shape): raise ValueError(f"{len(origin)}D tile origin does not match {len(shape)}D tile shape")
    if bounds is not None and (not isinstance(bounds, tuple) or len(bounds) != len(shape)):
      raise ValueError(f"tile bounds must be a {len(shape)}D tuple")
    return TileView(self, origin, shape, self.stride if stride is None else stride, bounds, mask, layout)

  def __getitem__(self, index): return Load(self.name, self._index(index))
  def __setitem__(self, index, value): self.builder.set(self.name, self._index(index), value)

class FragmentRef:
  def __init__(self, builder, name, shape, dtype):
    self.builder = builder
    self.name = name
    self.shape = shape
    self.dtype = dtype

def _numel(shape):
  if isinstance(shape, int): return shape
  if not isinstance(shape, tuple): raise TypeError("shape must be an int or tuple")
  out = 1
  for dim in shape:
    if not isinstance(dim, int): raise TypeError("tuple allocation shapes must contain integers")
    out *= dim
  return out
def _default_stride(shape): return shape[1] if isinstance(shape, tuple) and len(shape) == 2 else None
def _buffer_name(x): 
  if isinstance(x, TileView): return x.buffer.name
  return x.name if isinstance(x, BufferRef) else x
def _buffer_ref(x): return x.buffer if isinstance(x, TileView) else x
def _buffer_index(buffer, index): return buffer._index(index) if isinstance(buffer, BufferRef) else index
def _fragment_name(x): return x.name if isinstance(x, FragmentRef) else x
def _and(lhs, rhs):
  if lhs is None: return rhs
  if rhs is None: return lhs
  return lhs & rhs 
def _tile_guard(tile, names, origin):
  if tile is None: return None
  guard = tile.mask
  if tile.bounds is not None:
    for i, bound in enumerate(tile.bounds):
      coord = _add_if_nonzero(_origin_at(origin, i), Var(names[i]))
      guard = _and(guard, coord < bound)
  return guard
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

def _add_if_nonzero(lhs, rhs):
  return Add(lhs, rhs) if lhs != 0 else rhs

def _origin_at(origin, dim):
  return origin[dim] if dim < len(origin) else 0

def _copy_src_index(indices, shape, origin, stride):
  shifted = tuple(_add_if_nonzero(_origin_at(origin, i), idx) for i, idx in enumerate(indices))
  if len(shape) == 1 and stride is not None and len(origin) >= 2: return Index2D(shifted[0], _origin_at(origin, 1), stride)
  if len(shape) == 1: return shifted[0]
  if len(shape) == 2: return Index2D(shifted[0], shifted[1], stride)
  return _flatten_nd_index(shifted, shape)

def _copy_dst_index(indices, shape, origin, stride):
  shifted = tuple(_add_if_nonzero(_origin_at(origin, i), idx) for i, idx in enumerate(indices))
  if len(shape) == 1: return shifted[0]
  if len(shape) == 2: return Index2D(shifted[0], shifted[1], stride)
  return _flatten_nd_index(shifted, shape)

def _copy_shape(src, dst, shape):
  if shape is not None: return shape
  if isinstance(dst, BufferRef) and dst.shape is not None: return dst.shape
  if isinstance(src, BufferRef) and src.shape is not None: return src.shape
  raise ValueError("copy shape is required unless a shaped buffer ref is provided")

def _infer_tile_copy_shape(src_tile, dst_tile):
  if dst_tile is not None: return dst_tile.shape
  if src_tile is not None: return src_tile.shape
  return None

def _validate_tile_copy_shape(tile, shape, role):
  if tile is not None and tile.shape != shape:
    raise ValueError(f"tile copy shape mismatch: {role} tile shape {tile.shape} != copy shape {shape}")

def _copy_stride(buffer, stride, shape, name):
  if len(shape) != 2: return stride
  if stride is not None: return stride
  if isinstance(buffer, BufferRef) and buffer.shape is not None and len(buffer.shape) == 2:
    return buffer.shape[1]
  if name == "dst": return shape[1]
  raise ValueError(f"{name}_stride required for 2D copy")

class KernelBuilder:
  def __init__(self, name, args):
    self.name = name
    self.args = tuple(Arg(arg) for arg in args)
    self._body = []
    self._range_stack = []
    self._copy_counter = 0
    self._axes_counter = 0

  def _current_body(self): return self._range_stack[-1] if self._range_stack else self._body 

  def grid(self, *extents): return _AxesContext(self, "_g", extents, "global")

  def blocks(self, *extents): return self.grid(*extents)

  def threads(self, *extents): return _AxesContext(self, "_t", extents, "local")

  def parallel(self, *extents): return self.threads(*extents)

  def buffer(self, name, shape=None, dtype=None, stride=None, scope="global"):
    return BufferRef(self, name, shape, dtype, _default_stride(shape) if stride is None else stride, scope)
  
  def shared(self, name, shape, dtype):
    self.alloc(name, _numel(shape), dtype, "shared")
    return self.buffer(name, shape=shape, dtype=dtype, scope="shared")
  
  def register(self, name, shape, dtype):
    self.alloc(name, _numel(shape), dtype, "register")
    return self.buffer(name, shape=shape, dtype=dtype, scope="register")

  def buffers(self, *names): return tuple(self.buffer(name) for name in names)

  def fragment(self, name, shape, dtype):
    if self._range_stack: raise ValueError("fragment must be top-level")
    if not isinstance(shape, tuple) or len(shape) != 2:
      raise ValueError("fragment shape must be a 2D tuple")
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
      raise ValueError(f"fragment shape must contain positive integers: {shape}")
    self._body.append(FragmentAlloc(_fragment_name(name), shape, dtype))
    return FragmentRef(self, _fragment_name(name), shape, dtype)
  
  def load(self, buffer, index): return Load(_buffer_name(buffer), _buffer_index(buffer, index))

  def load_if(self, cond, buffer, index): return LoadIf(cond, _buffer_name(buffer), _buffer_index(buffer, index))

  def set(self, buffer, index, value): self._current_body().append(Set(_buffer_name(buffer), _buffer_index(buffer, index), value))

  def set_if(self, cond, buffer, index, value):
    self._current_body().append(SetIf(cond, _buffer_name(buffer), _buffer_index(buffer, index), value))

  def clear(self, fragment):
    self._current_body().append(FragmentClear(_fragment_name(fragment)))

  def gemm(self, a, b, c, trans_a=False, trans_b=False):
    if not isinstance(a, BufferRef): raise TypeError("gemm A must be a buffer reference")
    if not isinstance(b, BufferRef): raise TypeError("gemm B must be a buffer reference")
    if not isinstance(c, FragmentRef): raise TypeError("gemm C must be a fragment reference")
    if a.shape is None or b.shape is None: raise ValueError("gemm inputs require shapes")
    self._current_body().append(FragmentGemm(a.name, b.name, c.name, a.shape, b.shape, c.shape, trans_a, trans_b))

  def store_fragment(self, src, dst, dst_origin, guard=None, bounds=None):
    if not isinstance(src, FragmentRef): raise TypeError("store_fragment src must be a fragment reference")
    if not isinstance(dst, BufferRef): raise TypeError("store_fragment dst must be a buffer reference")
    if dst.shape is None or not isinstance(dst.shape, tuple) or len(dst.shape) != 2:
      raise ValueError("store_fragment dst must be a 2D buffer reference")
    if not isinstance(dst_origin, tuple) or len(dst_origin) != 2:
      raise ValueError("store_fragment dst_origin must be a 2D tuple")
    if bounds is not None and (not isinstance(bounds, tuple) or len(bounds) != 2):
      raise ValueError("store_fragment bounds must be a 2D tuple")
    self._current_body().append(FragmentStore(src.name, dst.name, dst_origin[0], dst_origin[1], dst.shape[1], guard, bounds))

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

  def pipelined(self, name, extent, stages=2):
    if not isinstance(stages, int) or stages <= 0:
      raise ValueError("pipelined stages must be a positive integer")
    return _RangeContext(self, name, extent, "loop")

  def copy(self, src, dst, shape=None, stride=None, src_row_off=0, src_col_off=0, src_origin=None, dst_origin=None, src_stride=None, dst_stride=None, guard=None, fill=None,):
    src_tile = src if isinstance(src, TileView) else None
    dst_tile = dst if isinstance(dst, TileView) else None
    src_ref = _buffer_ref(src)
    dst_ref = _buffer_ref(dst)
    if shape is None: shape = _infer_tile_copy_shape(src_tile, dst_tile)
    shape = _copy_shape(src_ref, dst_ref, shape)
    if not isinstance(shape, tuple): raise TypeError("copy shape must be a tuple")
    if len(shape) == 0: raise ValueError("copy shape must not be empty")
    if len(shape) > 3: raise NotImplementedError(f"copy does not support {len(shape)}D")
    if fill not in (None, 0): raise NotImplementedError("copy only supports fill=0")
    _validate_tile_copy_shape(src_tile, shape, "source")
    _validate_tile_copy_shape(dst_tile, shape, "destination")

    if src_origin is None:
      src_origin = src_tile.origin if src_tile is not None else (src_row_off, src_col_off) if len(shape) >= 2 or stride is not None else (src_col_off,) 
    if dst_origin is None:
      dst_origin = dst_tile.origin if dst_tile is not None else tuple(0 for _ in shape)

    if src_stride is None:
      src_stride = src_tile.stride if src_tile is not None and src_tile.stride is not None else stride
    if dst_stride is None:
      dst_stride = dst_tile.stride if dst_tile is not None and dst_tile.stride is not None else None
    
    src_stride = _copy_stride(src_ref, src_stride, shape, "src")
    dst_stride = _copy_stride(dst_ref, dst_stride, shape, "dst")

    src_name = _buffer_name(src_ref)
    dst_name = _buffer_name(dst_ref)
    n = self._copy_counter
    self._copy_counter += 1

    names = tuple(f"_c{n}_i{i}" for i in range(len(shape)))

    if src_tile is not None or dst_tile is not None:
      self._current_body().append(TileCopy(
        src=src_name,
        dst=dst_name,
        shape=shape,
        src_origin=src_origin,
        dst_origin=dst_origin,
        src_stride=src_stride,
        dst_stride=dst_stride,
        src_bounds=src_tile.bounds if src_tile is not None else None,
        dst_bounds=dst_tile.bounds if dst_tile is not None else None,
        src_mask=src_tile.mask if src_tile is not None else None,
        dst_mask=dst_tile.mask if dst_tile is not None else None,
        guard=guard,
        fill=fill,
        src_layout=src_tile.layout if src_tile is not None else None,
        dst_layout=dst_tile.layout if dst_tile is not None else None,
        index_names=names,
      ))
      return

    src_idx = _copy_src_index(names, shape, src_origin, src_stride)
    dst_idx = _copy_dst_index(names, shape, dst_origin, dst_stride)

    load_guard = _tile_guard(src_tile, names, src_origin)
    store_guard = _tile_guard(dst_tile, names, dst_origin)

    if guard is not None and fill == 0:
      load_guard = _and(load_guard, guard)
    elif guard is not None:
      store_guard = _and(store_guard, guard) 
    
    value = Load(src_name, src_idx)
    if load_guard is not None: value = LoadIf(load_guard, src_name, src_idx)

    stmt = StoreIf(store_guard, dst_name, dst_idx, value) if store_guard is not None else Store(dst_name, dst_idx, value)
    
    body = (stmt,)
    for name, extent in reversed(tuple(zip(names, shape))):
      body = (Range(name, extent, body),)
    self._current_body().extend(body)
  
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

class _AxesContext:
  def __init__(self, builder, prefix, extents, axis):
    if not extents: raise ValueError("axis context requires at least one extent")
    self.builder = builder
    self.prefix = prefix
    self.extents = extents
    self.axis = axis
    self.names = None

  def __enter__(self):
    n = self.builder._axes_counter
    self.builder._axes_counter += 1
    self.names = tuple(f"{self.prefix}{n}_i{i}" for i in range(len(self.extents)))
    self.builder._range_stack.append([])
    vars = tuple(Var(name) for name in self.names)
    return vars[0] if len(vars) == 1 else vars
  
  def __exit__(self, exc_type, exc, tb):
    body = tuple(self.builder._range_stack.pop())
    if exc_type is None:
      for name, extent in reversed(tuple(zip(self.names, self.extents))):
        body = (Range(name, extent, body, self.axis),)
      self.builder._current_body().extend(body)
    return False
