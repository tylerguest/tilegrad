from tilegrad.ir import Add, Alloc, And, FragmentAlloc, FragmentClear, FragmentGemm, FragmentStore, Index2D, Kernel, Load, Lt, Mul, Range, Set, Store, StoreIf, TileMMA

def _add_const(expr, value):
  return expr if value == 0 else Add(expr, value)

def _and(lhs, rhs):
  if lhs is None: return rhs
  if rhs is None: return lhs
  return And(lhs, rhs)

def _flatten(row, col, stride):
  return row * stride + col

class _FragmentExpander:
  def __init__(self):
    self.fragments = {}
    self.gemm_counter = 0

  def expand_kernel(self, kernel):
    body = []
    for op in kernel.body:
      body.extend(self.expand_op(op))
    return Kernel(kernel.name, kernel.args, tuple(body))

  def expand_body(self, body):
    out = []
    for op in body:
      out.extend(self.expand_op(op))
    return tuple(out)

  def fragment_shape(self, name):
    if name not in self.fragments: raise ValueError(f"unknown fragment: {name}")
    return self.fragments[name].shape

  @staticmethod
  def _expand_gemm(op, m, n):
    k = op.a_shape[0] if op.trans_a else op.a_shape[1]
    out = []
    for kk in range(k):
      for i in range(m):
        for j in range(n):
          acc_idx = _flatten(i, j, n)
          a_idx = Index2D(kk, i, op.a_shape[1]) if op.trans_a else Index2D(i, kk, op.a_shape[1])
          b_idx = Index2D(j, kk, op.b_shape[1]) if op.trans_b else Index2D(kk, j, op.b_shape[1])
          out.append(Set(op.c, acc_idx, Add(Load(op.c, acc_idx), Mul(Load(op.a, a_idx), Load(op.b, b_idx)))))
    return tuple(out)

  def expand_op(self, op):
    if isinstance(op, FragmentAlloc):
      self.fragments[op.name] = op
      m, n = op.shape
      return (Alloc(op.name, m * n, op.dtype, "register"),)
    if isinstance(op, Range):
      return (Range(op.name, op.extent, self.expand_body(op.body), op.axis),)
    if isinstance(op, FragmentClear):
      m, n = self.fragment_shape(op.buffer)
      return tuple(Set(op.buffer, _flatten(i, j, n), 0) for i in range(m) for j in range(n))
    if isinstance(op, FragmentGemm):
      self.gemm_counter += 1
      return self._expand_gemm(op, *self.fragment_shape(op.c))
    if isinstance(op, TileMMA):
      return self._expand_gemm(op, *op.c_shape)
    if isinstance(op, FragmentStore):
      m, n = self.fragment_shape(op.src)
      out = []
      for i in range(m):
        for j in range(n):
          row = _add_const(op.dst_row, i)
          col = _add_const(op.dst_col, j)
          idx = Index2D(row, col, op.dst_stride)
          val = Load(op.src, _flatten(i, j, n))
          guard = op.guard
          if op.bounds is not None:
            guard = _and(guard, And(Lt(row, op.bounds[0]), Lt(col, op.bounds[1])))
          out.append(StoreIf(guard, op.dst, idx, val) if guard is not None else Store(op.dst, idx, val))
      return tuple(out)
    return (op,)

def expand_fragments(kernel):
  return _FragmentExpander().expand_kernel(kernel)

def can_lower_fragment_gemm_intrinsic(dtype, a_shape, b_shape, c_shape):
  return False
