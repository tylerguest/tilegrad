from tinygrad import Tensor
from tilegrad import KernelBuilder, run

# Computes C[M, N] = A[M, K] @ B[K, N]
#
# Shape:
#   A: 3 x 5
#   B: 5 x 3
#   C: 3 x 3
#
# Tiling:
#   BM = 2
#   BN = 2
#   BK = 3
#
# Grid:
#   2 x 2 output tiles
#
# Threads:
#   3 local lanes used to cooperatively load each K tile.
#
# This intentionally stresses:
#   - grid axes
#   - local/thread axes
#   - shared memory
#   - barriers
#   - guarded edge loads
#   - guarded edge stores
#   - K tail masking

k = KernelBuilder("tilegrad_grid_threads_tiled_gemm_edge", ("out", "a", "b"))

BM = 2
BN = 2
BK = 3
M = 3
N = 3
K = 5
KTILES = 2

k.alloc("as", BM * BK, "float32")
k.alloc("bs", BK * BN, "float32")
k.alloc("acc", 1, "float32", "register")
out = k.buffer("out", shape=(M, N))
a = k.buffer("a", shape=(M, K))
b = k.buffer("b", shape=(K, N))
as_tile = k.buffer("as", shape=(BM, BK))
bs_tile = k.buffer("bs", shape=(BK, BN))
acc = k.buffer("acc")

with k.grid(2, 2) as (bi, bj):
  with k.range("ii", BM) as ii:
    with k.range("jj", BN) as jj:
      gi = bi * BM + ii
      gj = bj * BN + jj
      acc[0] = 0
      with k.range("ko", KTILES) as ko:
        # Cooperative local copy. Each local lane copies one K column/row
        # for all BM/BN positions in the tile.
        with k.threads(BK) as tk:
          gk = ko * BK + tk
          with k.range("li", BM) as li:
            row = bi * BM + li
            k.store(
              as_tile,
              (li, tk),
              k.load_if((row < M) & (gk < K), a, (row, gk)),
            )
          with k.range("lj", BN) as lj:
            col = bj * BN + lj
            k.store(
              bs_tile,
              (tk, lj),
              k.load_if((gk < K) & (col < N), b, (gk, col)),
            )
        k.barrier()
        with k.range("kk", BK, axis="reduce") as kk:
          acc[0] = acc[0] + as_tile[ii, kk] * bs_tile[kk, jj]
      k.store_if((gi < M) & (gj < N), out, (gi, gj), acc[0])

if __name__ == "__main__":
  a_t = Tensor([
    1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0,
  ])
  b_t = Tensor([
    16.0, 17.0, 18.0,
    19.0, 20.0, 21.0,
    22.0, 23.0, 24.0,
    25.0, 26.0, 27.0,
    28.0, 29.0, 30.0,
  ])
  out_t = Tensor.empty(M * N)
  print(run(k, out_t, a_t, b_t).tolist())