import argparse
import statistics
import subprocess
import time

from tinygrad import Device, Tensor, TinyJit

from tilegrad import run
from tilegrad.kernels import tiled_gemm


def parse_triplet(value, name):
  try:
    dims = tuple(int(x) for x in value.lower().split("x"))
  except ValueError as exc:
    raise argparse.ArgumentTypeError(f"{name} must contain integers: {value}") from exc
  if len(dims) != 3 or any(dim <= 0 for dim in dims):
    raise argparse.ArgumentTypeError(f"{name} must be three positive integers: {value}")
  return dims


def parse_shapes(value):
  # Accept semicolons for compatibility, but commas are safer to use in a shell.
  return tuple(parse_triplet(item.strip(), "shape") for item in value.replace(";", ",").split(",") if item.strip())


def gpu_model():
  try:
    out = subprocess.check_output(
      ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
      text=True,
      timeout=5,
    )
    return out.strip().splitlines()[0]
  except (FileNotFoundError, subprocess.SubprocessError, IndexError):
    return "unknown"


def make_matrix(rows, cols, offset=0):
  vals = [float(((i + offset) % 17) - 8) / 8.0 for i in range(rows * cols)]
  return Tensor(vals).reshape(rows, cols).realize()


def synchronize():
  Device[Device.DEFAULT].synchronize()


def prepare(fn):
  start = time.perf_counter()
  fn()  # compile
  fn()  # capture
  fn()  # first JIT replay
  synchronize()
  return time.perf_counter() - start


def time_call(fn):
  synchronize()
  start = time.perf_counter()
  fn()
  synchronize()
  return time.perf_counter() - start


def bench_pair(baseline_fn, tilegrad_fn, warmup, iters):
  for _ in range(warmup):
    baseline_fn()
    tilegrad_fn()
  synchronize()

  baseline_times = []
  tilegrad_times = []
  for _ in range(iters):
    calls = ((baseline_fn, baseline_times), (tilegrad_fn, tilegrad_times))
    if len(baseline_times) % 2: calls = tuple(reversed(calls))
    for fn, times in calls: times.append(time_call(fn))
  return statistics.median(baseline_times), statistics.median(tilegrad_times)


def gflops(M, N, K, seconds):
  return (2 * M * N * K) / seconds / 1e9


def max_abs_diff(a, b):
  a_vals = a.reshape(a.numel()).tolist()
  b_vals = b.reshape(b.numel()).tolist()
  return max(abs(x - y) for x, y in zip(a_vals, b_vals))


def tinygrad_case(a, b):
  @TinyJit
  def run_case(a, b):
    return (a @ b).realize()

  return lambda: run_case(a, b)


def tilegrad_case(M, N, K, BM, BN, BK, a, b):
  kernel = tiled_gemm(M, N, K, BM=BM, BN=BN, BK=BK)
  a_flat = a.reshape(M * K).realize()
  b_flat = b.reshape(K * N).realize()

  @TinyJit
  def run_case(a, b):
    out = Tensor.empty(M * N)
    return run(kernel, out, a, b).reshape(M, N)

  return lambda: run_case(a_flat, b_flat)


def print_rows(rows):
  print()
  print(f"{'backend':<22} {'shape':<12} {'prepare_s':>10} {'median_ms':>10} {'GFLOP/s':>10} {'vs base':>10} {'max error':>12}")
  print("-" * 94)
  for row in rows:
    print(
      f"{row['backend']:<22} "
      f"{row['shape']:<12} "
      f"{row['prepare_s']:>10.3f} "
      f"{row['median_ms']:>10.3f} "
      f"{row['gflops']:>10.3f} "
      f"{row['speedup']:>10.3f} "
      f"{row['max_abs_diff']:>12.2e}"
    )


def main():
  parser = argparse.ArgumentParser(description="Benchmark Tilegrad tiled GEMM against tinygrad.Tensor.matmul.")
  parser.add_argument(
    "--shapes",
    default="128x128x128,256x256x256,512x512x512",
    help="comma-separated MxNxK shapes",
  )
  parser.add_argument("--tile", default="16x64x32", help="Tilegrad tile as BMxBNxBK")
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--iters", type=int, default=10)
  args = parser.parse_args()

  try:
    shapes = parse_shapes(args.shapes)
    tile = parse_triplet(args.tile, "tile")
  except argparse.ArgumentTypeError as exc:
    parser.error(str(exc))
  if not shapes: parser.error("--shapes must contain at least one shape")
  if args.warmup < 0: parser.error("--warmup must be non-negative")
  if args.iters <= 0: parser.error("--iters must be positive")

  BM, BN, BK = tile
  gpu = gpu_model()

  print(f"GPU: {gpu}")
  print(f"tinygrad device: {Device.DEFAULT}")
  print(f"shapes: {shapes}")
  print(f"tile: {tile}")
  print(f"warmup: {args.warmup}, iterations: {args.iters}", flush=True)

  rows = []
  for M, N, K in shapes:
    shape_name = f"{M}x{N}x{K}"
    print(f"\nPreparing {shape_name}...", flush=True)
    a = make_matrix(M, K)
    b = make_matrix(K, N, offset=7)
    expected = (a @ b).realize()

    tiny_fn = tinygrad_case(a, b)
    tiny_prepare = prepare(tiny_fn)
    tiny_result = tiny_fn()
    tiny_error = max_abs_diff(tiny_result, expected)
    print(f"  tinygrad.matmul ready in {tiny_prepare:.3f}s", flush=True)

    tile_fn = tilegrad_case(M, N, K, BM, BN, BK, a, b)
    tile_prepare = prepare(tile_fn)
    tile_result = tile_fn()
    tile_error = max_abs_diff(tile_result, expected)
    print(f"  tilegrad.tiled_gemm ready in {tile_prepare:.3f}s", flush=True)
    tiny_median, tile_median = bench_pair(tiny_fn, tile_fn, args.warmup, args.iters)

    rows.extend((
      {
        "backend": "tinygrad.matmul",
        "shape": shape_name,
        "prepare_s": tiny_prepare,
        "median_ms": tiny_median * 1000,
        "gflops": gflops(M, N, K, tiny_median),
        "speedup": 1.0,
        "max_abs_diff": tiny_error,
      },
      {
        "backend": "tilegrad.tiled_gemm",
        "shape": shape_name,
        "prepare_s": tile_prepare,
        "median_ms": tile_median * 1000,
        "gflops": gflops(M, N, K, tile_median),
        "speedup": tiny_median / tile_median,
        "max_abs_diff": tile_error,
      },
    ))

  print_rows(rows)
  tile_rows = [row for row in rows if row["backend"] == "tilegrad.tiled_gemm"]
  best = max(tile_rows, key=lambda row: row["gflops"])
  best_speedup = max(tile_rows, key=lambda row: row["speedup"])
  print("\nResume fill-ins:")
  print("  baseline: tinygrad.Tensor.matmul")
  print(f"  GPU: {gpu}")
  print(f"  best Tilegrad result: {best['gflops']:.3f} GFLOP/s on {best['shape']}")
  print(f"  best speedup: {best_speedup['speedup']:.3f}x on {best_speedup['shape']}")
  print(f"  max Tilegrad numerical error: {max(row['max_abs_diff'] for row in tile_rows):.2e}")


if __name__ == "__main__":
  main()
