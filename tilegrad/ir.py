from dataclasses import dataclass

class Expr:
  def __add__(self, other): return Add(self, other)
  def __radd__(self, other): return Add(other, self)
  def __sub__(self, other): return Sub(self, other)
  def __rsub__(self, other): return Sub(other, self)
  def __mul__(self, other): return Mul(self, other)
  def __rmul__(self, other): return Mul(other, self)
  def __floordiv__(self, other): return FloorDiv(self, other)
  def __rfloordiv__(self, other): return FloorDiv(other, self)
  def __mod__(self, other): return Mod(self, other)
  def __rmod__(self, other): return Mod(other, self)
  def __lt__(self, other): return Lt(self, other)
  def __le__(self, other): return Le(self, other)
  def __gt__(self, other): return Gt(self, other)
  def __ge__(self, other): return Ge(self, other)
  def __and__(self, other): return And(self, other)
  def __rand__(self, other): return And(other, self)
  def __or__(self, other): return Or(self, other)
  def __ror__(self, other): return Or(other, self)

class Stmt: pass
class KernelOp: pass

@dataclass(frozen=True)
class Arg:
  name: str

@dataclass(frozen=True)
class Const(Expr):
  value: int | float

@dataclass(frozen=True)
class Var(Expr):
  name: str

@dataclass(frozen=True)
class BinaryExpr(Expr):
  lhs: object
  rhs: object

@dataclass(frozen=True)
class Lt(BinaryExpr): pass

@dataclass(frozen=True)
class Le(BinaryExpr): pass

@dataclass(frozen=True)
class Gt(BinaryExpr): pass

@dataclass(frozen=True)
class Ge(BinaryExpr): pass

@dataclass(frozen=True)
class Eq(BinaryExpr): pass

@dataclass(frozen=True)
class Ne(BinaryExpr): pass

@dataclass(frozen=True)
class And(BinaryExpr): pass

@dataclass(frozen=True)
class Or(BinaryExpr): pass

@dataclass(frozen=True)
class Not(Expr):
  x: object

@dataclass(frozen=True)
class Add(BinaryExpr): pass

@dataclass(frozen=True)
class Sub(BinaryExpr): pass

@dataclass(frozen=True)
class Mul(BinaryExpr): pass

@dataclass(frozen=True)
class FloorDiv(BinaryExpr): pass

@dataclass(frozen=True)
class Mod(BinaryExpr): pass

@dataclass(frozen=True)
class Set(Stmt):
  buffer: str
  index: object
  value: object

@dataclass(frozen=True)
class SetIf(Stmt):
  cond: object
  buffer: str
  index: object
  value: object

@dataclass(frozen=True)
class Index2D(Expr):
  row: object
  col: object
  stride: object

@dataclass(frozen=True)
class Load(Expr):
  buffer: str
  index: object

@dataclass(frozen=True)
class LoadIf(Expr):
  cond: object
  buffer: str
  index: object

@dataclass(frozen=True)
class Store(Stmt):
  buffer: str
  index: object
  value: object

@dataclass(frozen=True)
class StoreIf(Stmt):
  cond: object
  buffer: str
  index: object
  value: object

@dataclass(frozen=True)
class FragmentAlloc(KernelOp):
  name: str
  shape: tuple[int, int]
  dtype: str

@dataclass(frozen=True)
class FragmentClear(Stmt):
  buffer: str

@dataclass(frozen=True)
class FragmentGemm(Stmt):
  a: str
  b: str
  c: str
  a_shape: tuple[int, int]
  b_shape: tuple[int, int]
  c_shape: tuple[int, int]
  trans_a: bool = False
  trans_b: bool = False

@dataclass(frozen=True)
class FragmentStore(Stmt):
  src: str
  dst: str
  dst_row: object
  dst_col: object
  dst_stride: object
  guard: object | None = None
  bounds: tuple[object, object] | None = None

@dataclass(frozen=True)
class TileCopy(Stmt, KernelOp):
  src: str
  dst: str
  shape: tuple
  src_origin: tuple
  dst_origin: tuple
  src_stride: object | None = None
  dst_stride: object | None = None
  src_bounds: tuple | None = None
  dst_bounds: tuple | None = None
  src_mask: object | None = None
  dst_mask: object | None = None
  guard: object | None = None
  fill: object | None = None
  coalesced_width: int | None = None
  src_layout: object | None = None
  dst_layout: object | None = None
  index_names: tuple[str, ...] = ()

@dataclass(frozen=True)
class Range(Stmt, KernelOp):
  name: str
  extent: int | str
  body: tuple
  axis: str = "loop"

@dataclass(frozen=True)
class Alloc(KernelOp):
  name: str
  shape: int | str
  dtype: str
  space: str

@dataclass(frozen=True)
class Barrier(Stmt, KernelOp): pass

@dataclass(frozen=True)
class Kernel:
  name: str
  args: tuple[Arg, ...]
  body: tuple

def var(name:str): return Var(name)
def add(lhs:object, rhs:object): return Add(lhs, rhs)
def sub(lhs:object, rhs:object): return Sub(lhs, rhs)
def mul(lhs:object, rhs:object): return Mul(lhs, rhs)
def floordiv(lhs:object, rhs:object): return FloorDiv(lhs, rhs)
def mod(lhs:object, rhs:object): return Mod(lhs, rhs)
def idx2(row:object, col:object, stride:object): return Index2D(row, col, stride)
def lt(lhs:object, rhs:object): return Lt(lhs, rhs)
def le(lhs:object, rhs:object): return Le(lhs, rhs)
def gt(lhs:object, rhs:object): return Gt(lhs, rhs)
def ge(lhs:object, rhs:object): return Ge(lhs, rhs)
def eq(lhs:object, rhs:object): return Eq(lhs, rhs)
def ne(lhs:object, rhs:object): return Ne(lhs, rhs)
def and_(lhs:object, rhs:object): return And(lhs, rhs)
def or_(lhs:object, rhs:object): return Or(lhs, rhs)
def not_(x:object): return Not(x)
