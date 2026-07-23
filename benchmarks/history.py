import json
from datetime import datetime, timezone
from pathlib import Path


CHART_JS_URL = "https://cdn.jsdelivr.net/npm/chart.js"


def _created_at(record, path):
  value = record.get("created_at")
  if value is None:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)

  parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=timezone.utc)
  return parsed.astimezone(timezone.utc)


def _normalize_case(case):
  shape = case["shape"]
  measurements = case["measurements"]
  tilegrad = measurements["tilegrad"]
  throughput = case.get("throughput_gflops", {})
  correctness = case.get("correctness", {})

  return {
    "shape": f"{shape['M']}x{shape['N']}x{shape['K']}",
    "latency_ms": float(tilegrad["median_s"]) * 1000,
    "gflops": float(throughput.get("tilegrad", 0)),
    "baseline_ratio": float(measurements["median_ratio"]),
    "noise": float(tilegrad["interdecile_fraction"]),
    "max_abs_diff": float(correctness.get("tilegrad_max_abs_diff", 0)),
  }


def load_history(results_directory):
  results_directory = Path(results_directory)
  loaded = []

  for path in results_directory.glob("*.json"):
    try:
      record = json.loads(path.read_text(encoding="utf-8"))
      if record.get("record_type") != "diagnostic":
        continue

      created_at = _created_at(record, path)
      loaded.append((
        created_at,
        {
          "label": record.get("label") or path.stem,
          "created_at": created_at.isoformat().replace("+00:00", "Z"),
          "cases": [_normalize_case(case) for case in record["cases"]],
        },
      ))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ValueError(f"{path}: invalid benchmark record: {exc}") from exc

  if not loaded:
    raise ValueError(f"no benchmark records found in {results_directory}")

  runs = [run for _, run in sorted(loaded, key=lambda item: item[0])]

  first_latency = {}
  for run in runs:
    for case in run["cases"]:
      first_latency.setdefault(case["shape"], case["latency_ms"])
      case["improvement"] = first_latency[case["shape"]] / case["latency_ms"]

  return runs


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="10">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TileGrad Performance</title>
  <script src="__CHART_JS_URL__"></script>
  <style>
    :root {
      color-scheme: dark;
      --background: #090c13;
      --panel: #121925;
      --border: #273246;
      --text: #e7edf7;
      --muted: #8b98ad;
      --accent: #ff7557;
      --good: #58d6a1;
      --bad: #ff6b7a;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background:
        radial-gradient(circle at 10% 0%, #17243d 0, transparent 36rem),
        var(--background);
      color: var(--text);
      font-family: Inter, system-ui, sans-serif;
    }

    main {
      width: min(1400px, calc(100% - 32px));
      margin: auto;
      padding: 48px 0 80px;
    }

    .eyebrow {
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .15em;
      text-transform: uppercase;
    }

    h1 {
      margin: 8px 0;
      font-size: clamp(36px, 6vw, 68px);
      letter-spacing: -.05em;
    }

    .subtitle {
      color: var(--muted);
      margin-bottom: 28px;
    }

    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }

    .card, .panel {
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgb(18 25 37 / 94%);
      box-shadow: 0 20px 60px rgb(0 0 0 / 20%);
    }

    .card { padding: 18px; }

    .card-label {
      color: var(--muted);
      font-size: 11px;
      letter-spacing: .1em;
      text-transform: uppercase;
    }

    .card-value {
      margin-top: 7px;
      font-size: 24px;
      font-weight: 700;
    }

    .panel {
      height: 560px;
      padding: 24px;
    }

    .table-panel {
      height: auto;
      margin-top: 14px;
      overflow-x: auto;
    }

    canvas {
      width: 100% !important;
      height: 100% !important;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-variant-numeric: tabular-nums;
    }

    th, td {
      padding: 12px 10px;
      border-bottom: 1px solid var(--border);
      text-align: right;
    }

    th:first-child, td:first-child { text-align: left; }

    th {
      color: var(--muted);
      font-size: 11px;
      letter-spacing: .08em;
      text-transform: uppercase;
    }

    .good { color: var(--good); }
    .bad { color: var(--bad); }

    @media (max-width: 800px) {
      .cards { grid-template-columns: repeat(2, 1fr); }
      .panel { height: 430px; }
    }
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">Performance History</div>
    <h1>TileGrad</h1>
    <div class="subtitle">
      Speedup relative to the first recorded benchmark. Higher is better.
    </div>

    <section class="cards" id="cards"></section>

    <section class="panel">
      <canvas id="history"></canvas>
    </section>

    <section class="panel table-panel">
      <h2 id="latest-title">Latest Run</h2>
      <table>
        <thead>
          <tr>
            <th>Shape</th>
            <th>Latency</th>
            <th>Improvement</th>
            <th>vs tinygrad</th>
            <th>GFLOP/s</th>
            <th>Noise</th>
            <th>Error</th>
          </tr>
        </thead>
        <tbody id="latest-table"></tbody>
      </table>
    </section>
  </main>

  <script type="application/json" id="history-data">__HISTORY_DATA__</script>
  <script>
    const runs = JSON.parse(
      document.getElementById("history-data").textContent
    );

    const shapes = [...new Set(
      runs.flatMap(run => run.cases.map(item => item.shape))
    )];

    const colors = [
      "#ff7557", "#58d6a1", "#67a6ff", "#d987ff",
      "#ffd166", "#67d5e8", "#ff8fab", "#b8de6f"
    ];

    function pointsForShape(shape) {
      return runs.flatMap(run => {
        const item = run.cases.find(candidate => candidate.shape === shape);
        if (!item) return [];

        return [{
          x: run.label,
          y: item.improvement,
          latency: item.latency_ms,
          gflops: item.gflops,
          ratio: item.baseline_ratio,
          noise: item.noise,
          createdAt: run.created_at
        }];
      });
    }

    const baselineLine = {
      id: "baseline-line",
      afterDraw(chart) {
        const y = chart.scales.y.getPixelForValue(1);
        const {left, right} = chart.chartArea;
        const context = chart.ctx;

        context.save();
        context.strokeStyle = "#7d899d";
        context.setLineDash([6, 6]);
        context.beginPath();
        context.moveTo(left, y);
        context.lineTo(right, y);
        context.stroke();
        context.restore();
      }
    };

    new Chart(document.getElementById("history"), {
      type: "line",
      plugins: [baselineLine],
      data: {
        datasets: shapes.map((shape, index) => ({
          label: shape,
          data: pointsForShape(shape),
          borderColor: colors[index % colors.length],
          backgroundColor: colors[index % colors.length],
          borderWidth: 2.5,
          pointRadius: 4,
          pointHoverRadius: 7,
          tension: 0.18
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        parsing: false,
        interaction: {
          mode: "nearest",
          intersect: false
        },
        scales: {
          x: {
            type: "category",
            grid: {color: "#222d40"},
            ticks: {
              color: "#8b98ad",
              maxRotation: 30,
              minRotation: 30
            },
            title: {
              display: true,
              text: "Benchmark run",
              color: "#8b98ad"
            }
          },
          y: {
            grid: {color: "#222d40"},
            ticks: {
              color: "#8b98ad",
              callback: value => `${value.toFixed(2)}x`
            },
            title: {
              display: true,
              text: "Speedup vs first run",
              color: "#8b98ad"
            }
          }
        },
        plugins: {
          legend: {
            labels: {color: "#dfe7f3"}
          },
          tooltip: {
            callbacks: {
              label(context) {
                const point = context.raw;
                return [
                  `${context.dataset.label}: ${point.y.toFixed(3)}x`,
                  `Latency: ${point.latency.toFixed(4)} ms`,
                  `Throughput: ${point.gflops.toFixed(3)} GFLOP/s`,
                  `vs tinygrad: ${point.ratio.toFixed(3)}x`,
                  `Noise: ${(point.noise * 100).toFixed(2)}%`,
                  point.createdAt
                ];
              }
            }
          }
        }
      }
    });

    function geometricMean(values) {
      const valid = values.filter(value => value > 0);
      return Math.exp(
        valid.reduce((sum, value) => sum + Math.log(value), 0) /
        valid.length
      );
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    const latest = runs[runs.length - 1];
    const geomean = geometricMean(
      latest.cases.map(item => item.improvement)
    );

    const cards = [
      ["Runs", runs.length],
      ["Shapes", shapes.length],
      ["Latest", latest.label],
      ["Geomean improvement", `${geomean.toFixed(3)}x`]
    ];

    document.getElementById("cards").innerHTML = cards.map(([label, value]) => `
      <article class="card">
        <div class="card-label">${escapeHtml(label)}</div>
        <div class="card-value">${escapeHtml(value)}</div>
      </article>
    `).join("");

    document.getElementById("latest-title").textContent =
      `Latest Run: ${latest.label}`;

    document.getElementById("latest-table").innerHTML =
      latest.cases.map(item => `
        <tr>
          <td>${escapeHtml(item.shape)}</td>
          <td>${item.latency_ms.toFixed(4)} ms</td>
          <td class="${item.improvement >= 1 ? "good" : "bad"}">
            ${item.improvement.toFixed(3)}x
          </td>
          <td class="${item.baseline_ratio >= 1 ? "good" : "bad"}">
            ${item.baseline_ratio.toFixed(3)}x
          </td>
          <td>${item.gflops.toFixed(3)}</td>
          <td class="${item.noise <= 0.05 ? "good" : "bad"}">
            ${(item.noise * 100).toFixed(2)}%
          </td>
          <td>${item.max_abs_diff.toExponential(2)}</td>
        </tr>
      `).join("");
  </script>
</body>
</html>
"""


def build_history_html(runs):
  payload = json.dumps(runs, separators=(",", ":")).replace("<", "\\u003c")
  return (
    HTML_TEMPLATE
    .replace("__CHART_JS_URL__", CHART_JS_URL)
    .replace("__HISTORY_DATA__", payload)
  )


def write_history_html(results_directory, output_path):
  runs = load_history(results_directory)
  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(build_history_html(runs), encoding="utf-8")
  return output