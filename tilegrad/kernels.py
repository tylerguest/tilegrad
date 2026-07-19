from tilegrad.builder import KernelBuilder
from tilegrad.utils import ceildiv

def tiled_gemm(M, N, K, BM=16, BN=64, BK=32):
  for name, value in (("M", M), ("N", N), ("K", K), ("BM", BM), ("BN", BN), ("BK", BK)):
    if not isinstance(value, int) or value <= 0: raise ValueError(f"{name} must be a positive integer")

  TM = 2 if BM % 2 == 0 else 1
  TN = 2 if BN % 2 == 0 else 1
  THREAD_ROWS = BM // TM
  THREAD_COLS = BN // TN
  THREADS = THREAD_ROWS * THREAD_COLS
  if THREADS > 1024: raise ValueError(f"tiled_gemm requires at most 1024 threads per block, got {THREADS}")

  k = KernelBuilder("tilegrad_tiled_gemm", ("out", "a", "b"))
  KTILES = ceildiv(K, BK)

  out = k.buffer("out", shape=(M, N), dtype="float32")
  a = k.buffer("a", shape=(M, K), dtype="float32")
  b = k.buffer("b", shape=(K, N), dtype="float32")

  as_tile = k.shared("as", shape=(BM, BK), dtype="float32")
  bs_tile = k.shared("bs", shape=(BK, BN), dtype="float32")
  acc = k.register("acc", shape=(TM, TN), dtype="float32")

  with k.grid(ceildiv(M, BM), ceildiv(N, BN)) as (bi, bj):
    # Keep output columns on threadIdx.x so warp loads and stores are contiguous.
    with k.threads(THREAD_COLS, THREAD_ROWS) as (tj, ti):
      row = bi * BM + ti * TM
      col = bj * BN + tj * TN
      tid = ti * THREAD_COLS + tj
      k.clear(acc)

      for ko in range(KTILES):
        for load in range(ceildiv(BM * BK, THREADS)):
          idx = tid + load * THREADS
          ai = idx // BK
          ak = idx % BK
          g_row = bi * BM + ai
          g_k = ko * BK + ak
          valid = (idx < BM * BK) & (g_row < M) & (g_k < K)
          k.store_if(idx < BM * BK, as_tile, (ai, ak), k.load_if(valid, a, (g_row, g_k)))

        for load in range(ceildiv(BK * BN, THREADS)):
          idx = tid + load * THREADS
          bk = idx // BN
          bj_local = idx % BN
          g_k = ko * BK + bk
          g_col = bj * BN + bj_local
          valid = (idx < BK * BN) & (g_k < K) & (g_col < N)
          k.store_if(idx < BK * BN, bs_tile, (bk, bj_local), k.load_if(valid, b, (g_k, g_col)))

        k.barrier()
        for mi in range(TM):
          for nj in range(TN):
            with k.range(f"kk_{mi}_{nj}", BK, axis="reduce") as kk:
              acc[mi, nj] = acc[mi, nj] + as_tile[ti * TM + mi, kk] * bs_tile[kk, tj * TN + nj]
        k.barrier()

      for mi in range(TM):
        for nj in range(TN):
          out_row = row + mi
          out_col = col + nj
          k.store_if((out_row < M) & (out_col < N), out, (out_row, out_col), acc[mi, nj])
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
