import statistics
import time

from tinygrad import Tensor

from tilegrad import run
from tilegrad.kernels import fragment_gemm, grid_thread_fragment_gemm, tiled_gemm
from tilegrad.utils import ceildiv


SIZES = (
  (64, 64, 64),
  (128, 128, 128),
  (256, 256, 256),
)

BM = 8
BN = 8
BK = 8

FRAGMENT_BM = 2
FRAGMENT_BN = 2
FRAGMENT_BK = 8
# FragmentGemm currently expands to scalar operations, so keep this baseline small
# enough that the benchmark remains usable.
FRAGMENT_SIZES = {(64, 64, 64)}

GRID_THREAD_FRAGMENT_BM = 2
GRID_THREAD_FRAGMENT_BN = 2
GRID_THREAD_FRAGMENT_BK = 8
GRID_THREAD_FRAGMENT_SIZES = {(64, 64, 64)}


def gflops(M, N, K, seconds):
  return (2 * M * N * K) / seconds / 1e9


def sync(t):
  return t.realize()


def bench(fn, warmup=3, iters=10):
  for _ in range(warmup):
    sync(fn())
  times = []
  for _ in range(iters):
    start = time.perf_counter()
    sync(fn())
    times.append(time.perf_counter() - start)
  return statistics.median(times)


def max_abs_diff(a, b):
  a_vals = a.reshape(a.numel()).tolist()
  b_vals = b.reshape(b.numel()).tolist()
  return max(abs(x - y) for x, y in zip(a_vals, b_vals))


def tinygrad_matmul_case(M, N, K, a, b):
  return lambda: a @ b


def tilegrad_tiled_gemm_case(M, N, K, a, b):
  k = tiled_gemm(M, N, K, BM=BM, BN=BN, BK=BK)
  a_flat = a.reshape(M * K)
  b_flat = b.reshape(K * N)

  def run_case():
    out = Tensor.empty(M * N)
    return run(k, out, a_flat, b_flat, realize=False).reshape(M, N)

  return run_case


def tilegrad_fragment_gemm_case(M, N, K, a, b):
  k = fragment_gemm(M, N, K, BM=FRAGMENT_BM, BN=FRAGMENT_BN, BK=FRAGMENT_BK)
  a_flat = a.reshape(M * K)
  b_flat = b.reshape(K * N)

  def run_case():
    out = Tensor.empty(M * N)
    return run(k, out, a_flat, b_flat, realize=False).reshape(M, N)

  return run_case


def tilegrad_grid_thread_fragment_gemm_case(M, N, K, a, b):
  k = grid_thread_fragment_gemm(
    M, N, K,
    BM=GRID_THREAD_FRAGMENT_BM,
    BN=GRID_THREAD_FRAGMENT_BN,
    BK=GRID_THREAD_FRAGMENT_BK,
  )
  a_flat = a.reshape(M * K)
  b_flat = b.reshape(K * N)

  def run_case():
    out = Tensor.empty(M * N)
    return run(k, out, a_flat, b_flat, realize=False).reshape(M, N)

  return run_case


def correctness_check(M, N, K, a, b, got_fn):
  expected = (a @ b).realize()
  got = got_fn().realize()
  diff = max_abs_diff(got, expected)
  if diff > 1e-4:
    raise AssertionError(f"correctness failed for {M}x{N}x{K}: max_abs_diff={diff}")
  return diff


def bench_case(name, M, N, K, fn, launch="-"):
  first_start = time.perf_counter()
  sync(fn())
  first_ms = (time.perf_counter() - first_start) * 1000

  median_s = bench(fn)
  median_ms = median_s * 1000
  return {
    "backend": name,
    "shape": f"{M}x{N}x{K}",
    "first_ms": first_ms,
    "median_ms": median_ms,
    "gflops": gflops(M, N, K, median_s),
    "launch": launch,
  }


def print_rows(rows):
  print(f"{'backend':<24} {'shape':<12} {'first_ms':>10} {'median_ms':>10} {'gflops':>10} launch")
  for row in rows:
    print(
      f"{row['backend']:<24} "
      f"{row['shape']:<12} "
      f"{row['first_ms']:>10.3f} "
      f"{row['median_ms']:>10.3f} "
      f"{row['gflops']:>10.3f} "
      f"{row['launch']}"
    )


def seq(n):
  return [float(i + 1) for i in range(n)]


def main():
  rows = []
  for M, N, K in SIZES:
    a = Tensor(seq(M * K)).reshape(M, K).realize()
    b = Tensor(seq(K * N)).reshape(K, N).realize()

    tinygrad_fn = tinygrad_matmul_case(M, N, K, a, b)
    tilegrad_fn = tilegrad_tiled_gemm_case(M, N, K, a, b)

    diff = correctness_check(M, N, K, a, b, tilegrad_fn)
    print(f"correctness tilegrad.tiled_gemm {M}x{N}x{K}: max_abs_diff={diff}")

    rows.append(bench_case(
      "tinygrad.matmul",
      M, N, K,
      tinygrad_fn,
    ))

    rows.append(bench_case(
      "tilegrad.tiled_gemm",
      M, N, K,
      tilegrad_fn,
      launch=f"grid=({ceildiv(M, BM)},{ceildiv(N, BN)}) threads=({BN // 2},{BM // 2}) microtile=(2,2)",
    ))

    if (M, N, K) in FRAGMENT_SIZES:
      fragment_fn = tilegrad_fragment_gemm_case(M, N, K, a, b)
      fragment_diff = correctness_check(M, N, K, a, b, fragment_fn)
      print(f"correctness tilegrad.fragment_gemm {M}x{N}x{K}: max_abs_diff={fragment_diff}")
      rows.append(bench_case(
        "tilegrad.fragment_gemm",
        M, N, K,
        fragment_fn,
        launch=f"tiles=({ceildiv(M, FRAGMENT_BM)},{ceildiv(N, FRAGMENT_BN)}) fragment=({FRAGMENT_BM},{FRAGMENT_BN})",
      ))

    if (M, N, K) in GRID_THREAD_FRAGMENT_SIZES:
      grid_thread_fragment_fn = tilegrad_grid_thread_fragment_gemm_case(M, N, K, a, b)
      grid_thread_fragment_diff = correctness_check(M, N, K, a, b, grid_thread_fragment_fn)
      print(f"correctness tilegrad.grid_thread_fragment_gemm {M}x{N}x{K}: max_abs_diff={grid_thread_fragment_diff}")
      rows.append(bench_case(
        "tilegrad.grid_thread_frag",
        M, N, K,
        grid_thread_fragment_fn,
        launch=(
          f"grid=({ceildiv(M, GRID_THREAD_FRAGMENT_BM)},{ceildiv(N, GRID_THREAD_FRAGMENT_BN)}) "
          f"threads=(1) fragment=({GRID_THREAD_FRAGMENT_BM},{GRID_THREAD_FRAGMENT_BN})"
        ),
      ))

  print()
  print_rows(rows)


if __name__ == "__main__":
  main()
