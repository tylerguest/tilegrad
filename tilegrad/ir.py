from dataclasses import dataclass

@dataclass(frozen=True)
class Arg:
  name: str 

@dataclass(frozen=True)
class Load:
  buffer: str
  index: str

@dataclass(frozen=True)
class Store:
  buffer: str 
  index: str 
  value: int | float | Load

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
