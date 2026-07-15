from tilegrad.ir import Add, And, Index2D, Kernel, Load, LoadIf, Mul, Range, Store, StoreIf, TileCopy, Var

def _and(lhs, rhs):
  if lhs is None: return rhs
  if rhs is None: return lhs 
  return And(lhs, rhs)

def _add_if_nonzero(lhs, rhs): return Add(lhs, rhs) if lhs != 0 else rhs

def _origin_at(origin, dim): return origin[dim] if dim < len(origin) else 0

def _flatten_nd_index(indices, shape):
  if len(indices) != len(shape): raise ValueError(f"{len(indices)}D index does not match {len(shape)}D shape")
  if len(indices) == 1: return indices[0]
  if len(indices) == 2: return Index2D(indices[0], indices[1], shape[1])
  if not all(isinstance(dim, int) for dim in shape): raise TypeError("N-D tuple indexing requires integer buffer shapes")
  
  flat = indices[-1]
  stride = 1
  for idx, dim in zip(reversed(indices[:-1]), reversed(shape[1:])):
    stride *= dim
    flat = Add(Mul(idx, stride), flat)
  return flat

def _copy_src_index(indices, shape, origin, stride):
  shifted = tuple(_add_if_nonzero(_origin_at(origin, i), idx) for i, idx in enumerate(indices))
  if len(shape) == 1 and stride is not None and len(origin) >= 2:
    return Index2D(shifted[0], _origin_at(origin, 1), stride)
  if len(shape) == 1: return shifted[0]
  if len(shape) == 2: return Index2D(shifted[0], shifted[1], stride)
  return _flatten_nd_index(shifted, shape)

def _copy_dst_index(indices, shape, origin, stride):
  shifted = tuple(_add_if_nonzero(_origin_at(origin, i), idx) for i, idx in enumerate(indices))
  if len(shape) == 1: return shifted[0]
  if len(shape) == 2: return Index2D(shifted[0], shifted[1], stride)
  return _flatten_nd_index(shifted, shape)

def _tile_guard(bounds, mask, names, origin):
  guard = mask
  if bounds is not None:
    for i, bound in enumerate(bounds):
      coord = _add_if_nonzero(_origin_at(origin, i), Var(names[i]))
      guard = _and(guard, coord < bound)
  return guard

def _expand_tile_copy(op):
  if not isinstance(op.shape, tuple): raise TypeError("tile copy shape must be a tuple")
  if len(op.shape) == 0: raise ValueError("tile copy shape must not be empty")
  if len(op.shape) > 3: raise NotImplementedError(f"tile copy does not support {len(op.shape)}D")
  names = op.index_names or tuple(f"_tc_i{i}" for i in range(len(op.shape)))
  if len(names) != len(op.shape):
    raise ValueError (f"tile copy index name count {len(names)} does not match shape rank {len(op.shape)}")
  src_idx = _copy_src_index(names, op.shape, op.src_origin, op.src_stride)
  dst_idx = _copy_dst_index(names, op.shape, op.dst_origin, op.dst_stride)
  load_guard = _tile_guard(op.src_bounds, op.src_mask, names, op.src_origin)
  store_guard = _tile_guard(op.dst_bounds, op.dst_mask, names, op.dst_origin)

  if op.guard is not None and op.fill == 0: load_guard = _and(load_guard, op.guard)
  elif op.guard is not None: store_guard = _and(store_guard, op.guard)

  value = Load(op.src, src_idx)
  if load_guard is not None: value = LoadIf(load_guard, op.src, src_idx)
  stmt = StoreIf(store_guard, op.dst, dst_idx, value) if store_guard is not None else Store(op.dst, dst_idx, value)
  body = (stmt,)
  for name, extent in reversed(tuple(zip(names, op.shape))): body = (Range(name, extent, body),)
  return body

def _expand_body(body):
  out = []
  for op in body:
    if isinstance(op, TileCopy): out.extend(_expand_tile_copy(op))
    elif isinstance(op, Range): out.append(Range(op.name, op.extent, _expand_body(op.body), op.axis))
    else: out.append(op)
  return tuple(out)

def expand_tile_copies(kernel): return Kernel(kernel.name, kernel.args, _expand_body(kernel.body))