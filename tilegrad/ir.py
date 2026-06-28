from dataclasses import dataclass

@dataclass(frozen=True)
class Arg:
  name: str 

@dataclass(frozen=True)
class Const:
  value: int | float

@dataclass(frozen=True)
class Add:
  lhs: object
  rhs: object

@dataclass(frozen=True)
class Mul:
  lhs: object
  rhs: object

@dataclass(frozen=True)
class Load:
  buffer: str
  index: object

@dataclass(frozen=True)
class Store:
  buffer: str 
  index: object
  value: object

@dataclass(frozen=True)
class Range:
  name: str
  extent: int | str 
  body: tuple

@dataclass(frozen=True)
class Alloc:
  name: str
  shape: int | str
  dtype: str
  space: str

@dataclass(frozen=True)
class Barrier:
  pass

@dataclass(frozen=True)
class Kernel:
  name: str 
  args: tuple[Arg, ...]
  body: tuple 
