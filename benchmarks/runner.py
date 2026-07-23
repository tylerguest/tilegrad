import json
import math
import statistics
import time
from pathlib import Path


MIN_WARMUPS = 10
MIN_SAMPLES = 30


def validate_protocol(warmups, samples):
  if type(warmups) is not int or warmups < MIN_WARMUPS:
    raise ValueError(f"warmups must be an integer >= {MIN_WARMUPS}")
  if type(samples) is not int or samples < MIN_SAMPLES:
    raise ValueError(f"samples must be an integer >= {MIN_SAMPLES}")


def _percentile(samples, fraction):
  ordered = sorted(samples)
  position = (len(ordered) - 1) * fraction
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  weight = position - lower
  return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_samples(samples):
  values = tuple(float(sample) for sample in samples)
  if not values:
    raise ValueError("at least one timing sample is required")
  if any(not math.isfinite(sample) or sample <= 0 for sample in values):
    raise ValueError("timing samples must be finite and positive")

  p10 = _percentile(values, 0.10)
  median = statistics.median(values)
  p90 = _percentile(values, 0.90)

  return {
    "sample_count": len(values),
    "samples_s": list(values),
    "p10_s": p10,
    "median_s": median,
    "p90_s": p90,
    "interdecile_fraction": (p90 - p10) / median,
  }


def prepare(fn, synchronize, launches=3, clock=time.perf_counter):
  if type(launches) is not int or launches <= 0:
    raise ValueError("preparation launches must be a positive integer")

  start = clock()
  for _ in range(launches):
    fn()
    synchronize()
  return clock() - start


def synchronized_time(fn, synchronize, clock=time.perf_counter):
  synchronize()
  start = clock()
  fn()
  synchronize()
  elapsed = clock() - start
  if elapsed <= 0:
    raise RuntimeError("timer returned a non-positive duration")
  return elapsed


def benchmark_pair(
  baseline_fn,
  tilegrad_fn,
  *,
  synchronize,
  warmups=MIN_WARMUPS,
  samples=MIN_SAMPLES,
  timer=None,
):
  validate_protocol(warmups, samples)

  candidates = (
    ("baseline", baseline_fn),
    ("tilegrad", tilegrad_fn),
  )

  for iteration in range(warmups):
    ordered = candidates if iteration % 2 == 0 else tuple(reversed(candidates))
    for _, fn in ordered:
      fn()
      synchronize()

  if timer is None:
    timer = lambda fn: synchronized_time(fn, synchronize)

  timings = {
    "baseline": [],
    "tilegrad": [],
  }

  for iteration in range(samples):
    ordered = candidates if iteration % 2 == 0 else tuple(reversed(candidates))
    for name, fn in ordered:
      timings[name].append(timer(fn))

  baseline = summarize_samples(timings["baseline"])
  tilegrad = summarize_samples(timings["tilegrad"])

  return {
    "baseline": baseline,
    "tilegrad": tilegrad,
    "median_ratio": baseline["median_s"] / tilegrad["median_s"],
  }


def write_json(path, record):
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
  target.write_text(payload + "\n", encoding="utf-8")
  return target