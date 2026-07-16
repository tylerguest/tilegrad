from tilegrad.builder import KernelBuilder
from tilegrad.utils import ceildiv

def tiled_gemm(M, N, K, BM=2, BN=2, BK=3):
  k = KernelBuilder("tilegrad_tiled_gemm", ("out", "a", "b"))
  KTILES = ceildiv(K, BK)

  out = k.buffer("out", shape=(M, N), dtype="float32")
  a = k.buffer("a", shape=(M, K), dtype="float32")
  b = k.buffer("b", shape=(K, N), dtype="float32")

  as_tile = k.shared("as", shape=(BM, BK), dtype="float32")
  bs_tile = k.shared("bs", shape=(BK, BN), dtype="float32")
  acc = k.register("acc", shape=(BM, BN), dtype="float32")

  with k.grid(ceildiv(M, BM), ceildiv(N, BN)) as (bi, bj):
    with k.threads(1) as _:
      k.clear(acc)
      for ko in range(KTILES):
        k.copy(a.tile(origin=(bi * BM, ko * BK), shape=(BM, BK), bounds=(M, K)), as_tile.tile(),)
        k.copy(b.tile(origin=(ko * BK, bj * BN), shape=(BK, BN), bounds=(K, N)), bs_tile.tile(),)
        k.barrier()
        k.gemm(as_tile, bs_tile, acc)
      for i in range(BM):
        for j in range(BN):
          row = bi * BM + i
          col = bj * BN + j
          k.store_if((row < M) & (col < N), out, (row, col), acc[i, j])
  return k

def fragment_gemm(M, N, K, BM=8, BN=8, BK=8):
  k = KernelBuilder("tilegrad_fragment_gemm", ("out", "a", "b"))
  KTILES = ceildiv(K, BK)

  k.alloc("as", BM * BK, "float32")
  k.alloc("bs", BK * BN, "float32")

  out = k.buffer("out", shape=(M, N))
  a = k.buffer("a", shape=(M, K))
  b = k.buffer("b", shape=(K, N))
  as_tile = k.buffer("as", shape=(BM, BK))
  bs_tile = k.buffer("bs", shape=(BK, BN))
  acc = k.fragment("acc", (BM, BN), "float32")

  with k.range("bi", ceildiv(M, BM)) as bi:
    with k.range("bj", ceildiv(N, BN)) as bj:
      k.clear(acc)
      for ko in range(KTILES):
        with k.range("ii", BM) as ii:
          gi = bi * BM + ii
          with k.range("kk", BK) as kk:
            gk = ko * BK + kk
            k.store(as_tile, (ii, kk), k.load_if((gi < M) & (gk < K), a, (gi, gk)))
        with k.range("kk", BK) as kk:
          gk = ko * BK + kk
          with k.range("jj", BN) as jj:
            gj = bj * BN + jj
            k.store(bs_tile, (kk, jj), k.load_if((gk < K) & (gj < N), b, (gk, gj)))
        k.barrier()
        k.gemm(as_tile, bs_tile, acc)
      k.store_fragment(acc, out, (bi * BM, bj * BN), bounds=(M, N))

  return k

def grid_thread_fragment_gemm(M, N, K, BM=2, BN=2, BK=3):
  k = KernelBuilder("tilegrad_grid_thread_fragment_gemm", ("out", "a", "b"))
  KTILES = ceildiv(K, BK)

  k.alloc("as", BM * BK, "float32")
  k.alloc("bs", BK * BN, "float32")

  out = k.buffer("out", shape=(M, N))
  a = k.buffer("a", shape=(M, K))
  b = k.buffer("b", shape=(K, N))
  as_tile = k.buffer("as", shape=(BM, BK))
  bs_tile = k.buffer("bs", shape=(BK, BN))
  acc = k.fragment("acc", (BM, BN), "float32")

  with k.grid(ceildiv(M, BM), ceildiv(N, BN)) as (bi, bj):
    with k.threads(1) as _:
      k.clear(acc)
      for ko in range(KTILES):
        with k.range("ii", BM) as ii:
          gi = bi * BM + ii
          with k.range("kk", BK) as kk:
            gk = ko * BK + kk
            k.store(as_tile, (ii, kk), k.load_if((gi < M) & (gk < K), a, (gi, gk)))
        with k.range("kk", BK) as kk:
          gk = ko * BK + kk
          with k.range("jj", BN) as jj:
            gj = bj * BN + jj
            k.store(bs_tile, (kk, jj), k.load_if((gk < K) & (gj < N), b, (gk, gj)))
        k.barrier()
        k.gemm(as_tile, bs_tile, acc)
      k.store_fragment(acc, out, (bi * BM, bj * BN), bounds=(M, N))

  return k
