import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.runner import (
  benchmark_pair,
  summarize_samples,
  validate_protocol,
  write_json,
)


class TestBenchmarkRunner(unittest.TestCase):
  def test_protocol_requires_manifest_minimums(self):
    with self.assertRaisesRegex(ValueError, "warmups"):
      validate_protocol(9, 30)
    with self.assertRaisesRegex(ValueError, "samples"):
      validate_protocol(10, 29)

    self.assertIsNone(validate_protocol(10, 30))

  def test_summary_preserves_samples_and_reports_percentiles(self):
    samples = tuple(float(i) for i in range(1, 11))
    summary = summarize_samples(samples)

    self.assertEqual(summary["sample_count"], 10)
    self.assertEqual(summary["samples_s"], list(samples))
    self.assertAlmostEqual(summary["p10_s"], 1.9)
    self.assertAlmostEqual(summary["median_s"], 5.5)
    self.assertAlmostEqual(summary["p90_s"], 9.1)
    self.assertAlmostEqual(
      summary["interdecile_fraction"],
      (9.1 - 1.9) / 5.5,
    )

  def test_summary_rejects_invalid_samples(self):
    for samples in ((), (0.0,), (-1.0,), (float("nan"),)):
      with self.subTest(samples=samples):
        with self.assertRaises(ValueError):
          summarize_samples(samples)

  def test_benchmark_pair_alternates_candidate_order(self):
    events = []

    def baseline():
      events.append("baseline")

    def tilegrad():
      events.append("tilegrad")

    def timer(fn):
      fn()
      return 0.002 if fn is baseline else 0.001

    result = benchmark_pair(
      baseline,
      tilegrad,
      synchronize=lambda: None,
      warmups=10,
      samples=30,
      timer=timer,
    )

    timed_events = events[20:]
    self.assertEqual(
      timed_events[:4],
      ["baseline", "tilegrad", "tilegrad", "baseline"],
    )
    self.assertEqual(result["baseline"]["sample_count"], 30)
    self.assertEqual(result["tilegrad"]["sample_count"], 30)
    self.assertEqual(result["median_ratio"], 2.0)

  def test_write_json_creates_machine_readable_record(self):
    record = {
      "record_type": "diagnostic",
      "claim_level": None,
      "samples": [0.001, 0.002],
    }

    with TemporaryDirectory() as directory:
      path = Path(directory) / "nested" / "result.json"
      written = write_json(path, record)

      self.assertEqual(written, path)
      self.assertEqual(
        json.loads(path.read_text(encoding="utf-8")),
        record,
      )


if __name__ == "__main__":
  unittest.main()