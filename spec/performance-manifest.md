# TileGrad 1.0 Performance Manifest

Manifest version: 1

This manifest defines how TileGrad performance claims are scoped, measured, and
reported. It does not grant a claim by itself. A claim exists only when a
completed capability record and reproducible benchmark results are published.

## Scope

Performance is claimed for a capability tuple, not for TileGrad, tinygrad, a
vendor, or a renderer family in general. A tuple identifies:

- TileGrad revision
- tinygrad Git SHA
- target and renderer
- device model and architecture
- driver, compiler, and operating-system versions
- operation profile and profile version
- input, accumulator, and output dtypes
- admitted shape set, layouts, alignment, and participant geometry
- selected schedule and tuning-space version
- baseline implementation and revision

Core conformance carries no performance claim. Targets without the typed
operations required by a performance profile continue to use the portable Core
lowering.

## Claim Levels

Ratios are computed as baseline latency divided by TileGrad latency for each
case, then combined using the geometric mean.

| Level | Geometric mean | Per-case floor | Meaning |
| --- | ---: | ---: | --- |
| Competitive | 0.80 | 0.65 | TileGrad reaches at least 80% of baseline throughput geometrically and 65% on every case. |
| Parity | 0.90 | 0.80 | TileGrad reaches at least 90% of baseline throughput geometrically and 80% on every case. |

A claim may publish stricter thresholds. It must not use the words
"competitive" or "parity" with weaker gates. A claim against a non-TileLang
baseline must name that baseline in its description. Only a claim measured
against a pinned TileLang baseline may use "TileLang-like performance"; only the
Parity level against TileLang may use "TileLang parity."

## Baseline Rules

A TileLang comparison must pin the TileLang revision and use the same logical
operation, input and output dtypes, accumulator dtype, layouts visible to the
API, masking, epilogue, and numerical tolerance.

The baseline must not use a mechanism absent from the claimed TileGrad
capability tuple. For example, a synchronous native-matrix claim compares with
TileLang's ordinary MMA path with TMA, WGMMA, warp specialization, clusters, and
other unavailable mechanisms disabled. A separate claim is required when both
implementations support one of those mechanisms.

When TileLang does not support the target or capability tuple, results may be
reported against tinygrad or a vendor library, but they must not be described as
TileLang-like performance or TileLang parity.

## Measurement Protocol

Each published result must follow this protocol:

1. Build TileGrad, tinygrad, and each applicable baseline from pinned revisions.
2. Preallocate and initialize all inputs and outputs before timed iterations.
3. Validate correctness before collecting performance samples.
4. Report compilation and autotuning separately from steady-state execution.
5. Warm up each candidate for at least 10 successful launches.
6. Collect at least 30 synchronized device-time samples per case.
7. Report median, 10th percentile, 90th percentile, and geometric-mean ratio.
8. Repeat a case when the interdecile range exceeds 5% of the median.
9. Record clocks, power mode, and other device settings that can affect results.
10. Preserve raw samples, generated source, launch geometry, and selected schedule.

Backend event timing should be used when both implementations expose comparable
events. Otherwise, use a synchronized host timer around only the launch and
execution. The same method must be used for TileGrad and its baseline.

A noisy case may be repeated at most twice after the initial run. At least one
complete run must have an interdecile range no greater than the claim's
permitted variance, which defaults to 5% of the median. If all three runs exceed
that limit, the case is invalid and the capability claim fails. A published
record must include the permitted and observed variance for every case.

## Correctness

Every timed case must compare TileGrad and baseline outputs against the same
reference. The capability record defines absolute and relative tolerances and
the inherited tinygrad numerical mode. Native Matrix cases must also run through
the portable path and inspect the generated program to prove that the native
path contains the selected typed matrix operation.

Failed correctness, failed native-operation inspection, unexpected fallback, or
an unsupported shape invalidates that benchmark sample. Such a case must not be
removed from the suite after a claim is published without a manifest revision.

## Initial Synchronous Copy Suite

The Vector and Cooperative profile must include:

- aligned contiguous copies at 4 MiB, 64 MiB, and 256 MiB
- an aligned two-dimensional copy
- a masked tail whose element count is not divisible by the selected vector width
- a two-dimensional transpose with irregular edge dimensions
- every dtype named by the claim, with at least one of `float16`, `bfloat16`, or `float32`

The record must report effective bandwidth, selected vector width, generated
load/store width, workgroup geometry, and whether scalar tail operations were
emitted. A competitive synchronous-copy claim requires the Competitive gate in
this manifest.

## Initial Native Matrix Suite

The first Native Matrix suite covers `float16` or `bfloat16` inputs, `float32`
accumulation, and the output dtype named by the capability record. A performance
claim names one capability tuple whose admitted shape set and layout rules cover
every required case below. Each case records its exact shape and resolved layout
under that tuple. An unsupported required case prevents a claim for the suite;
results from different capability tuples must not be combined to pass a gate.

The suite must include at least these logical GEMM shapes `(M, N, K)`:

| Family | Shapes |
| --- | --- |
| Square | `(1024, 1024, 1024)`, `(2048, 2048, 2048)`, `(4096, 4096, 4096)` |
| Rectangular | `(4096, 1024, 4096)`, `(1024, 4096, 4096)`, `(4096, 4096, 1024)` |
| Edge tiled | `(1000, 1000, 1000)`, `(4097, 4093, 1025)` |
| Fused epilogue | `(2048, 2048, 2048)` with one pointwise activation |

The suite must report latency, TFLOP/s, selected native matrix configuration,
native operation count, vector load/store width, shared-memory bytes, launch
geometry, and schedule parameters. At least one edge case must exercise masked
TileCopy and layout legalization.

A TileGrad 1.0 Native Matrix release claim must meet at least the Competitive
gate. A parity claim must meet the Parity gate.

## Regression Policy

Once published, a capability claim is part of the release contract for its
pinned environment. A new TileGrad revision must not retain the claim when:

- its geometric-mean ratio falls below the named level
- any required case falls below the per-case floor
- median TileGrad latency regresses by more than 5% under the same environment
- native operation, vectorization, layout, or launch inspection no longer matches the record

An intentional benchmark, baseline, or threshold change requires a new manifest
version and must retain the previous results for comparison.

## Claim Record

Each claim should provide a machine-readable record with at least these fields:

```yaml
manifest_version: 1
claim_level: competitive
profile: native-matrix
profile_version: 1
tilegrad_revision: <git-sha>
tinygrad_revision: <git-sha>
baseline:
  name: tilelang
  revision: <git-sha>
target: <tinygrad-device>
renderer: <renderer-name>
device: <model-and-architecture>
software: <driver-compiler-os>
dtypes: <input-accumulator-output>
layouts: <resolved-layout-identifiers>
schedule: <selected-configuration>
suite: native-matrix-v1
correctness: <reference-and-tolerances>
permitted_variance: <fraction>
observed_variance: <per-case-values>
geomean_ratio: <number>
minimum_case_ratio: <number>
artifacts: <path-or-url>
```

## Current Claims

No TileGrad 1.0 performance capability has been published yet. This section must
be updated only from reproducible benchmark artifacts; targets or mechanisms
must not be listed speculatively.
