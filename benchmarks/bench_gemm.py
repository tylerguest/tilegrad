import argparse
import subprocess
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from tinygrad import Device, Tensor, TinyJit

from benchmarks.runner import benchmark_pair, prepare, validate_protocol, write_json
from benchmarks.history import write_history_html
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
  return tuple(
    parse_triplet(item.strip(), "shape")
    for item in value.replace(";", ",").split(",")
    if item.strip()
  )


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
  values = [
    float(((i + offset) % 17) - 8) / 8.0
    for i in range(rows * cols)
  ]
  return Tensor(values).reshape(rows, cols).realize()


def synchronize():
  Device[Device.DEFAULT].synchronize()


def gflops(M, N, K, seconds):
  return (2 * M * N * K) / seconds / 1e9


def max_abs_diff(a, b):
  a_values = a.reshape(a.numel()).tolist()
  b_values = b.reshape(b.numel()).tolist()
  return max(abs(x - y) for x, y in zip(a_values, b_values))


def tinygrad_case(a, b):
  @TinyJit
  def run_case(a, b):
    return (a @ b).realize()

  return lambda: run_case(a, b)


def tilegrad_case(M, N, K, BM, BN, BK, a, b):
  kernel = tiled_gemm(M, N, K, BM=BM, BN=BN, BK=BK)
  a_flat = a.reshape(M * K).realize()
  b_flat = b.reshape(K * N).realize()
  out = Tensor.empty(M * N, dtype=a.dtype, device=a.device).realize()

  @TinyJit
  def run_case(out, a, b):
    return run(kernel, out, a, b).reshape(M, N)

  return lambda: run_case(out, a_flat, b_flat)


def print_case(case):
  shape = case["shape"]
  baseline = case["measurements"]["baseline"]
  tilegrad = case["measurements"]["tilegrad"]

  print(
    f"{shape['M']}x{shape['N']}x{shape['K']}: "
    f"baseline median={baseline['median_s'] * 1000:.3f} ms "
    f"(p10={baseline['p10_s'] * 1000:.3f}, "
    f"p90={baseline['p90_s'] * 1000:.3f}), "
    f"TileGrad median={tilegrad['median_s'] * 1000:.3f} ms "
    f"(p10={tilegrad['p10_s'] * 1000:.3f}, "
    f"p90={tilegrad['p90_s'] * 1000:.3f}), "
    f"ratio={case['measurements']['median_ratio']:.3f}x"
  )


def main():
  parser = argparse.ArgumentParser(
    description="Run the TileGrad GEMM benchmark and update the performance graph."
  )
  parser.add_argument(
    "--shapes",
    default="128x128x128,256x256x256,512x512x512",
    help="comma-separated MxNxK shapes",
  )
  parser.add_argument(
    "--tile",
    default="16x64x32",
    help="TileGrad tile as BMxBNxBK",
  )
  parser.add_argument("--warmup", type=int, default=10)
  parser.add_argument(
    "--samples",
    "--iters",
    dest="samples",
    type=int,
    default=30,
    help="timed samples per candidate",
  )
  parser.add_argument("--atol", type=float, default=1e-4)
  parser.add_argument(
    "--output",
    help="optional JSON output path",
  )
  parser.add_argument(
    "--no-open",
    action="store_true",
    help="do not open the performance graph after benchmarking",
  )
  args = parser.parse_args()

  try:
    shapes = parse_shapes(args.shapes)
    tile = parse_triplet(args.tile, "tile")
    validate_protocol(args.warmup, args.samples)
  except (argparse.ArgumentTypeError, ValueError) as exc:
    parser.error(str(exc))

  if not shapes:
    parser.error("--shapes must contain at least one shape")
  if args.atol < 0:
    parser.error("--atol must be non-negative")

  BM, BN, BK = tile
  model = gpu_model()
  now = datetime.now(timezone.utc)
  created_at = now.isoformat().replace("+00:00", "Z")
  label = now.strftime("%Y-%m-%d %H:%M:%S")
  stamp = now.strftime("%Y%m%d-%H%M%S-%f")
  output_path = (
    Path(args.output)
    if args.output
    else Path("benchmarks/results") / f"{stamp}.json"
  )

  record = {
    "record_type": "diagnostic",
    "claim_level": None,
    "created_at": created_at,
    "label": label,
    "suite": "gemm-diagnostic",
    "timing_method": "synchronized-host",
    "device": {
      "tinygrad": str(Device.DEFAULT),
      "model": model,
    },
    "candidates": {
      "baseline": "tinygrad.Tensor.matmul",
      "tilegrad": "tilegrad.tiled_gemm",
    },
    "schedule": {
      "BM": BM,
      "BN": BN,
      "BK": BK,
    },
    "protocol": {
      "warmups": args.warmup,
      "samples_per_candidate": args.samples,
      "permitted_variance": 0.05,
    },
    "correctness": {
      "reference": "tinygrad.Tensor.matmul",
      "absolute_tolerance": args.atol,
    },
    "cases": [],
  }

  print(f"device: {Device.DEFAULT}")
  print(f"model: {model}")
  print(f"tile: {tile}")
  print(f"warmups: {args.warmup}")
  print(f"samples: {args.samples}")

  for M, N, K in shapes:
    print(f"\nPreparing {M}x{N}x{K}...")

    a = make_matrix(M, K)
    b = make_matrix(K, N, offset=7)
    expected = (a @ b).realize()

    baseline_fn = tinygrad_case(a, b)
    tilegrad_fn = tilegrad_case(M, N, K, BM, BN, BK, a, b)

    baseline_prepare_s = prepare(baseline_fn, synchronize)
    tilegrad_prepare_s = prepare(tilegrad_fn, synchronize)

    baseline_error = max_abs_diff(baseline_fn(), expected)
    tilegrad_error = max_abs_diff(tilegrad_fn(), expected)

    if baseline_error > args.atol:
      raise AssertionError(
        f"baseline correctness failed for {M}x{N}x{K}: "
        f"max_abs_diff={baseline_error}"
      )
    if tilegrad_error > args.atol:
      raise AssertionError(
        f"TileGrad correctness failed for {M}x{N}x{K}: "
        f"max_abs_diff={tilegrad_error}"
      )

    measurements = benchmark_pair(
      baseline_fn,
      tilegrad_fn,
      synchronize=synchronize,
      warmups=args.warmup,
      samples=args.samples,
    )

    case = {
      "shape": {
        "M": M,
        "N": N,
        "K": K,
      },
      "preparation_s": {
        "baseline": baseline_prepare_s,
        "tilegrad": tilegrad_prepare_s,
      },
      "correctness": {
        "baseline_max_abs_diff": baseline_error,
        "tilegrad_max_abs_diff": tilegrad_error,
      },
      "measurements": measurements,
      "throughput_gflops": {
        "baseline": gflops(M, N, K, measurements["baseline"]["median_s"]),
        "tilegrad": gflops(M, N, K, measurements["tilegrad"]["median_s"]),
      },
    }
    record["cases"].append(case)
    print_case(case)

  output = write_json(output_path, record)
  print(f"\nWrote diagnostic record to {output}")

  graph = write_history_html(
    output.parent,
    output.parent / "performance.html",
  )
  print(f"Wrote performance graph to {graph}")

  if not args.no_open:
    webbrowser.open(graph.resolve().as_uri())


if __name__ == "__main__":
  main()