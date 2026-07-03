from tilegrad.ir import Add, FloorDiv

def ceildiv(a, b): return (a + b - 1) // b

def ceildiv_expr(a, b):
  if not isinstance(b, int): raise TypeError("ceildiv_expr divisor must be an int")
  return FloorDiv(Add(a, b - 1), b)
