from tilegrad.ir import Alloc, BinaryExpr, Const, Index2D, Kernel, Load, LoadIf, Not, Range, Set, SetIf, Store, StoreIf, Var

def _vars_in_expr(expr, subs):
  if isinstance(expr, (int, float)): return set()
  if isinstance(expr, str): return set() if expr in subs else {expr}
  if isinstance(expr, Var): return set() if expr.name in subs else {expr.name}
  if isinstance(expr, Const): return set()
  if isinstance(expr, Not): return _vars_in_expr(expr.x, subs)
  if isinstance(expr, BinaryExpr): return _vars_in_expr(expr.lhs, subs) | _vars_in_expr(expr.rhs, subs)
  if isinstance(expr, Index2D): return _vars_in_expr(expr.row, subs) | _vars_in_expr(expr.col, subs) | _vars_in_expr(expr.stride, subs)
  if isinstance(expr, Load): return _vars_in_expr(expr.index, subs)
  if isinstance(expr, LoadIf): return _vars_in_expr(expr.cond, subs) | _vars_in_expr(expr.index, subs)
  return set()

def _reg_index_vars_expr(expr, register_buffers, subs):
  if isinstance(expr, (int, float, str, Var, Const)): return set()
  if isinstance(expr, Not): return _reg_index_vars_expr(expr.x, register_buffers, subs)
  if isinstance(expr, BinaryExpr):
    return _reg_index_vars_expr(expr.lhs, register_buffers, subs) | _reg_index_vars_expr(expr.rhs, register_buffers, subs)
  if isinstance(expr, Index2D):
    return (_reg_index_vars_expr(expr.row, register_buffers, subs) |
            _reg_index_vars_expr(expr.col, register_buffers, subs) |
            _reg_index_vars_expr(expr.stride, register_buffers, subs))
  if isinstance(expr, Load):
    ret = _reg_index_vars_expr(expr.index, register_buffers, subs)
    if expr.buffer in register_buffers: ret |= _vars_in_expr(expr.index, subs)
    return ret
  if isinstance(expr, LoadIf):
    ret = _reg_index_vars_expr(expr.cond, register_buffers, subs) | _reg_index_vars_expr(expr.index, register_buffers, subs)
    if expr.buffer in register_buffers: ret |= _vars_in_expr(expr.index, subs)
    return ret
  return set()

def _reg_index_vars_stmt(stmt, register_buffers, subs):
  if isinstance(stmt, (Set, Store)):
    ret = _reg_index_vars_expr(stmt.index, register_buffers, subs) | _reg_index_vars_expr(stmt.value, register_buffers, subs)
    if stmt.buffer in register_buffers: ret |= _vars_in_expr(stmt.index, subs)
    return ret
  if isinstance(stmt, (SetIf, StoreIf)):
    ret = (_reg_index_vars_expr(stmt.cond, register_buffers, subs) |
           _reg_index_vars_expr(stmt.index, register_buffers, subs) |
           _reg_index_vars_expr(stmt.value, register_buffers, subs))
    if stmt.buffer in register_buffers: ret |= _vars_in_expr(stmt.index, subs)
    return ret
  if isinstance(stmt, Range): return _reg_index_vars_ops(stmt.body, register_buffers, subs)
  return set()

def _reg_index_vars_ops(ops, register_buffers, subs):
  ret = set()
  for op in ops: ret |= _reg_index_vars_stmt(op, register_buffers, subs)
  return ret

def _subst_expr(expr, subs):
  if isinstance(expr, (int, float)): return expr
  if isinstance(expr, str): return subs.get(expr, expr)
  if isinstance(expr, Var): return subs.get(expr.name, expr)
  if isinstance(expr, Const): return expr
  if isinstance(expr, Not): return Not(_subst_expr(expr.x, subs))
  if isinstance(expr, BinaryExpr): return type(expr)(_subst_expr(expr.lhs, subs), _subst_expr(expr.rhs, subs))
  if isinstance(expr, Index2D): return Index2D(_subst_expr(expr.row, subs), _subst_expr(expr.col, subs), _subst_expr(expr.stride, subs))
  if isinstance(expr, Load): return Load(expr.buffer, _subst_expr(expr.index, subs))
  if isinstance(expr, LoadIf): return LoadIf(_subst_expr(expr.cond, subs), expr.buffer, _subst_expr(expr.index, subs))
  return expr

def _rewrite_stmt(stmt, register_buffers, subs, unroll_product, max_unroll):
  if isinstance(stmt, Set):
    return (Set(stmt.buffer, _subst_expr(stmt.index, subs), _subst_expr(stmt.value, subs)),)
  if isinstance(stmt, SetIf):
    return (SetIf(_subst_expr(stmt.cond, subs), stmt.buffer, _subst_expr(stmt.index, subs), _subst_expr(stmt.value, subs)),)
  if isinstance(stmt, Store):
    return (Store(stmt.buffer, _subst_expr(stmt.index, subs), _subst_expr(stmt.value, subs)),)
  if isinstance(stmt, StoreIf):
    return (StoreIf(_subst_expr(stmt.cond, subs), stmt.buffer, _subst_expr(stmt.index, subs), _subst_expr(stmt.value, subs)),)
  if isinstance(stmt, Range): return _rewrite_range(stmt, register_buffers, subs, unroll_product, max_unroll)
  return (stmt,)

def _rewrite_ops(ops, register_buffers, subs, unroll_product, max_unroll):
  out = []
  for op in ops: out.extend(_rewrite_stmt(op, register_buffers, subs, unroll_product, max_unroll))
  return tuple(out)

def _rewrite_range(op, register_buffers, subs, unroll_product, max_unroll):
  reg_vars = _reg_index_vars_ops(op.body, register_buffers, subs)
  needs_unroll = op.name in reg_vars

  if needs_unroll and op.axis == "loop" and isinstance(op.extent, int) and 0 < op.extent and unroll_product * op.extent <= max_unroll:
    out = []
    for i in range(op.extent):
      out.extend(_rewrite_ops(op.body, register_buffers, subs | {op.name: i}, unroll_product * op.extent, max_unroll))
    return tuple(out)

  body = _rewrite_ops(op.body, register_buffers, subs, unroll_product, max_unroll)
  return (Range(op.name, op.extent, body, op.axis),)

def unroll_register_tiles(kernel, max_unroll=16):
  register_buffers = {op.name for op in kernel.body if isinstance(op, Alloc) and op.space == "register"}
  if not register_buffers: return kernel
  body = []
  for op in kernel.body:
    if isinstance(op, Range): body.extend(_rewrite_range(op, register_buffers, {}, 1, max_unroll))
    elif isinstance(op, (Set, SetIf, Store, StoreIf)): body.extend(_rewrite_stmt(op, register_buffers, {}, 1, max_unroll))
    else: body.append(op)
  return Kernel(kernel.name, kernel.args, tuple(body))
