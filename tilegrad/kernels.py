from tilegrad.builder import KernelBuilder
from tilegrad.utils import ceildiv

def tiled_gemm(M, N, K, BM=2, BN=2, BK=3):
  k = KernelBuilder("tilegrad_tiled_gemm", ("out", "a", "b"))
  KTILES = ceildiv(K, BK)

  k.alloc("as", BM * BK, "float32")
  k.alloc("bs", BK * BN, "float32")
  k.alloc("acc", 1, "float32", "register")

  out = k.buffer("out", shape=(M, N))
  a = k.buffer("a", shape=(M, K))
  b = k.buffer("b", shape=(K, N))
  as_tile = k.buffer("as", shape=(BM, BK))
  bs_tile = k.buffer("bs", shape=(BK, BN))
  acc = k.buffer("acc")

  with k.grid(ceildiv(M, BM), ceildiv(N, BN)) as (bi, bj):
    with k.threads(BM, BN) as (ti, tj):
      gi = bi * BM + ti
      gj = bj * BN + tj
      acc[0] = 0
      with k.range("ko", KTILES) as ko:
        with k.range("kk_a", BK) as kk_a:
          gk_a = ko * BK + kk_a
          k.store(as_tile, (ti, kk_a), k.load_if((gi < M) & (gk_a < K), a, (gi, gk_a)),)
        with k.range("kk_b", BK) as kk_b:
          gk_b = ko * BK + kk_b
          k.store(bs_tile, (kk_b, tj), k.load_if((gk_b < K) & (gj < N), b, (gk_b, gj)),)
        k.barrier()
        with k.range("kk", BK, axis="reduce") as kk:
          acc[0] = acc[0] + as_tile[ti, kk] * bs_tile[kk, tj]
      k.store_if((gi < M) & (gj < N), out, (gi, gj), acc[0])
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
