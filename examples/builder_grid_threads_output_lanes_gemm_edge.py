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
#   2 x 2 local lanes.
#   Each local lane computes one C element in the output tile.
#
# This is a stronger boundary test than cooperative-copy-only threading because
# the final output store depends on local thread axes.
k = KernelBuilder("tilegrad_grid_threads_output_lanes_gemm_edge", ("out", "a", "b"))
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
  with k.threads(BM, BN) as (ti, tj):
    gi = bi * BM + ti
    gj = bj * BN + tj
    acc[0] = 0
    with k.range("ko", KTILES) as ko:
      # Cooperative shared-memory fill using the 2x2 output lanes.
      #
      # as_tile is 2x3. Four local lanes fill six elements by looping over kk.
      # Each thread loads A[gi, gk] for its own row ti and all K columns.
      with k.range("kk_a", BK) as kk_a:
        gk_a = ko * BK + kk_a
        k.store(
          as_tile,
          (ti, kk_a),
          k.load_if((gi < M) & (gk_a < K), a, (gi, gk_a)),
        )
      # bs_tile is 3x2. Four local lanes fill six elements by looping over kk.
      # Each thread loads B[gk, gj] for its own column tj and all K rows.
      with k.range("kk_b", BK) as kk_b:
        gk_b = ko * BK + kk_b
        k.store(
          bs_tile,
          (kk_b, tj),
          k.load_if((gk_b < K) & (gj < N), b, (gk_b, gj)),
        )
      k.barrier()
      with k.range("kk", BK, axis="reduce") as kk:
        acc[0] = acc[0] + as_tile[ti, kk] * bs_tile[kk, tj]
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