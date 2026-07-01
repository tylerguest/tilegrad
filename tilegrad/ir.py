from dataclasses import dataclass

class Expr: pass
class Stmt: pass
class KernelOp: pass

@dataclass(frozen=True)
class Arg:
  name: str

@dataclass(frozen=True)
class Const(Expr):
  value: int | float

@dataclass(frozen=True)
class BinaryExpr(Expr):
  lhs: object
  rhs: object

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
class Index2D(Expr):
  row: object
  col: object
  stride: object

@dataclass(frozen=True)
class Load(Expr):
  buffer: str
  index: object

@dataclass(frozen=True)
class Store(Stmt):
  buffer: str
  index: object
  value: object

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

def add(lhs:object, rhs:object): return Add(lhs, rhs)
def sub(lhs:object, rhs:object): return Sub(lhs, rhs)
def mul(lhs:object, rhs:object): return Mul(lhs, rhs)
def floordiv(lhs:object, rhs:object): return FloorDiv(lhs, rhs)
def mod(lhs:object, rhs:object): return Mod(lhs, rhs)
def idx2(row:object, col:object, stride:object): return Index2D(row, col, stride)