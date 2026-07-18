from tinygrad import Tensor
from tilegrad import run
from tilegrad.debug import inspect_kernel
from tilegrad.ir import Range, TileCopy, TileMMA
from tilegrad.kernels import tiled_gemm

M = 3
N = 3
K = 5
BM = 2
BN = 2
BK = 3

k = tiled_gemm(M, N, K, BM, BN, BK)


def ref_matmul(a, b, m, n, k_dim):
  return [
    sum(a[i * k_dim + kk] * b[kk * n + j] for kk in range(k_dim))
    for i in range(m)
    for j in range(n)
  ]


def has_op(body, op_type):
  for op in body:
    if isinstance(op, op_type): return True
    if isinstance(op, Range) and has_op(op.body, op_type): return True
  return False


if __name__ == "__main__":
  a_vals = [
    1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0,
  ]
  b_vals = [
    16.0, 17.0, 18.0,
    19.0, 20.0, 21.0, 
    22.0, 23.0, 24.0, 
    25.0, 26.0, 27.0, 
    28.0, 29.0, 30.0,
  ]
  a_t = Tensor(a_vals)
  b_t = Tensor(b_vals)
  out_t = Tensor.empty(M * N)
  out = run(k, out_t, a_t, b_t).tolist()
  expected = ref_matmul(a_vals, b_vals, M, N, K)
  if out != expected:
    raise AssertionError(f"canonical tiled GEMM mismatch: got={out} expected={expected}")

  dbg = inspect_kernel(k)
  print("Result:")
  print(out)
  print("Matches reference:", out == expected)
  print("TileGrad IR stages:")
  print([stage.name for stage in dbg.stages])
  print("tile_ir has TileCopy:", has_op(dbg.tile_ir.body, TileCopy))
  print("tile_ir has TileMMA:", has_op(dbg.tile_ir.body, TileMMA))
  print("scalar_ir has TileCopy:", has_op(dbg.scalar_ir.body, TileCopy))
  print("scalar_ir has TileMMA:", has_op(dbg.scalar_ir.body, TileMMA))
