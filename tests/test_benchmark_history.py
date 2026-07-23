import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.history import load_history, write_history_html


def benchmark_record(created_at, latency):
  baseline = 0.010

  return {
    "record_type": "diagnostic",
    "created_at": created_at,
    "label": created_at,
    "cases": [{
      "shape": {"M": 128, "N": 128, "K": 128},
      "measurements": {
        "baseline": {
          "median_s": baseline,
          "interdecile_fraction": 0.02,
        },
        "tilegrad": {
          "median_s": latency,
          "interdecile_fraction": 0.03,
        },
        "median_ratio": baseline / latency,
      },
      "throughput_gflops": {
        "baseline": 100,
        "tilegrad": 80,
      },
      "correctness": {
        "tilegrad_max_abs_diff": 1e-6,
      },
    }],
  }


class TestBenchmarkHistory(unittest.TestCase):
  def test_history_orders_runs_and_computes_improvement(self):
    with TemporaryDirectory() as directory:
      root = Path(directory)

      (root / "after.json").write_text(
        json.dumps(benchmark_record(
          "2026-07-24T12:00:00Z",
          0.008,
        )),
        encoding="utf-8",
      )
      (root / "before.json").write_text(
        json.dumps(benchmark_record(
          "2026-07-23T12:00:00Z",
          0.010,
        )),
        encoding="utf-8",
      )

      runs = load_history(root)

      self.assertEqual(runs[0]["cases"][0]["improvement"], 1.0)
      self.assertEqual(runs[1]["cases"][0]["improvement"], 1.25)

  def test_write_history_html_embeds_chart_data(self):
    with TemporaryDirectory() as directory:
      root = Path(directory)
      results = root / "results"
      results.mkdir()

      (results / "baseline.json").write_text(
        json.dumps(benchmark_record(
          "2026-07-23T12:00:00Z",
          0.010,
        )),
        encoding="utf-8",
      )

      output = root / "performance.html"
      write_history_html(results, output)
      html = output.read_text(encoding="utf-8")

      self.assertIn("TileGrad", html)
      self.assertIn("chart.js", html)
      self.assertIn('"improvement":1.0', html)
      self.assertNotIn("__HISTORY_DATA__", html)


if __name__ == "__main__":
  unittest.main()