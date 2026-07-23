# TileGrad Roadmap

TileGrad is the high-performance custom-kernel layer for tinygrad. It provides
explicit tile schedules, layouts, capability selection, tuning, and inspection
while tinygrad owns scalar semantics, native instruction configuration, code
generation, compilation, caching, and execution.

The core dataflow remains:

```text
global tile -> shared tile -> register tile/fragment -> mma -> global tile store
```

The portable scalar/SIMT lowering is the correctness oracle. Performance comes
from preserving `TileCopy` and `TileMMA` until target-aware capability selection,
not from maintaining a separate hand-written kernel path.

## Release Ladder

TileGrad has four milestones:

1. **Portable baseline:** builder, Tile IR, validation, scalar/SIMT fallback,
   inspection, and tinygrad runtime integration.
2. **Performance alpha:** builder-first vector/cooperative copy and native matrix
   lowering with a published capability record and manifest-compliant results.
3. **Core 1.0 conformance:** both frontends and portable correctness on the
   implementation's compatible tinygrad target matrix.
4. **Official TileGrad 1.0:** Core 1.0 plus at least one Native Matrix
   performance claim.

The performance alpha intentionally does not wait for the `@tg.kernel` frontend.
The builder and decorator must produce the same Tile IR before Core 1.0.

## Validation Environment Workflow

Use each environment only for the evidence it can provide:

| Environment | Required use | Must not claim |
| --- | --- | --- |
| CPU | Portable IR, validation, scalar semantics, and fast correctness tests | GPU lowering or performance |
| NVIDIA GTX 1050 (SM61) | Real CUDA scalar/SIMT execution, vector copy, transpose, barriers, and launch correctness | Native Matrix or modern tensor-core performance |
| AMD MockGPU | AMD capability selection, WMMA/MFMA lowering, native instruction inspection, launch geometry, and emulated numerical correctness | Real latency, bandwidth, occupancy, or throughput |
| Synthetic CUDA SM75/80/89 renderer tests | CUDA native-matrix eligibility, `Ops.WMMA`, generated `mma.sync`, and renderer/compiler acceptance | Runtime correctness or performance on that architecture |
| Physical modern GPU | Final native correctness, hardware resource behavior, tuning, and performance-manifest results | Results for a different device or capability tuple |

The required development loop is:

1. Prove portable semantics on CPU.
2. Prove real scalar/SIMT CUDA behavior on the GTX 1050.
3. Prove AMD native-matrix lowering and emulated correctness with MockGPU.
4. Prove CUDA native-matrix selection and source emission with synthetic modern
   renderer tests.
5. Run identical correctness and inspection cases on physical target hardware.
6. Tune and collect performance-manifest samples only on physical hardware.

MockGPU timing is emulator timing. It must never be recorded as device latency,
bandwidth, TFLOP/s, a performance regression, or evidence for Competitive or
Parity claims. MockGPU removes compiler-development and correctness blockers; it
does not remove the physical-hardware requirement for performance claims.

## Current Baseline

Implemented:

- `KernelBuilder`, TileGrad IR, validation, and tinygrad UOp lowering.
- Runtime execution through `Tensor.custom_kernel`.
- Global, local, serial, reduce, and unroll axes.
- Shared and register allocations, barriers, and effect ordering.
- Guarded loads and stores with edge and tail coverage.
- First-class `TileView`, `TileCopy`, and `TileMMA` nodes.
- Canonical `TileCopy` emission for every builder copy form.
- One portable scalar expansion for all `TileCopy` operations and scalar
  fallback lowering for `TileMMA`.
- 1D, 2D, and compact 3D copies with origins, strides, masks, bounds,
  zero-fill, and `coalesced_width` metadata.
- Cooperative-thread FP32 GEMM with shared-memory staging and register
  microtiles.
- M/N edge-tile and K-tail correctness coverage.
- Multi-element register lifetime handling across range boundaries.
- Inspection of tile IR, expansion stages, scalar IR, and optional tinygrad
  UOps.
- Exploratory benchmarks against tinygrad matmul.

The existing scalar path and correctness tests remain supported throughout the
optimization work.

## Architectural Gaps

- `TileCopy` and `TileMMA` are scalarized before target or capability selection.
- The canonical GEMM bypasses both tile primitives and emits manual scalar
  reductions.
- No capability records, profile selection, or native rejection diagnostics
  exist.
- No versioned tinygrad integration contract exists.
- Empty `KernelInfo.opts_to_apply` is used without a complete tested schedule
  preservation contract.
- Tile layouts are metadata placeholders and are rejected by validation.
- `coalesced_width` does not affect lowering.
- `TileMMA` accepts only FP32 operands and never emits a native matrix operation.
- `pipelined(...)` discards stage information and behaves like a serial loop.
- Existing benchmark scripts allocate outputs during timed calls and do not
  satisfy the performance manifest.

## Immediate Work Order

1. Publish the tinygrad integration contract and compatibility checks.
2. Replace exploratory benchmarks with the manifest runner.
3. Introduce scheduled/legalized IR and capability records.
4. Implement effects, alias, collective-legality, and numerical contracts.
5. Convert canonical GEMM to first-class `TileCopy` and `TileMMA`.
6. Implement core layouts.
7. Implement vector and cooperative synchronous copy.
8. Implement mixed-precision native matrix lowering.
9. Add reproducible schedule tuning and publish the first performance alpha.

## Phase 0: Stabilization And Integration Contract

Goal: make the portable path and tinygrad boundary reliable enough to optimize.

Work:

- Run portable tests on CPU and CUDA scalar/SIMT tests on available physical
  hardware.
- Add AMD MockGPU tests for native matrix lowering, instruction inspection,
  launch geometry, and emulated numerical correctness.
- Add synthetic CUDA SM75/80/89 renderer tests for native matrix selection and
  emitted `mma.sync` source.
- Keep performance suites separate from fast correctness tests.
- Add regressions for nested global/local/serial/reduce axes containing tile
  operations, larger register tiles, invalid fills/layouts, aliases, overlapping
  copies, barriers, and expression edge cases.
- Add a versioned integration contract describing admitted UOps,
  `KernelInfo` behavior, required legalization, schedule preservation, target
  discovery, native matrix selection, and known incompatibilities.
- Add compatibility tests for `Tensor.custom_kernel`, enclosing `TinyJit`, and
  `@function(precompile=True)`.
- Fail incompatible tinygrad revisions with a clear compatibility error.

Gate:

- Full correctness suite passes locally.
- No tile operation reaches UOp lowering without validation.
- Existing UOp and rendered-program inspection preserves grid/local geometry,
  memory scopes, and barriers through the current tinygrad integration.
- Portable outputs remain unchanged from the recorded baseline.

## Phase 1: Claim-Quality Benchmarking

Goal: measure the correct thing before optimizing it.

Work:

- Replace the existing GEMM scripts with one reusable benchmark runner.
- Support suite definitions, machine-readable claim records, raw samples,
  generated source, launch geometry, and selected schedule.
- Preallocate tensors and outputs before timed calls.
- Report compilation and autotuning separately from steady-state execution.
- Use comparable device-event timing when available and synchronized host timing
  otherwise.
- Detect MockGPU and refuse to emit performance-claim records from emulator
  timing.
- Pin TileLang for TileLang-like claims and retain tinygrad/vendor comparisons
  for diagnostics.
- Disable baseline mechanisms absent from the claimed TileGrad capability tuple.
- Define separate schemas for compile-time capability records, specialization
  keys, and published performance-claim records.
- Include TileGrad/tinygrad/baseline revisions, profile version, device and
  software environment, tuning-space version, correctness tolerances, observed
  variance, and artifact location in published claim records.
- Record device clocks, power mode, and every available setting that can affect
  benchmark results.

Gate:

- At least 10 warmup launches and 30 synchronized samples per case.
- Median, p10, p90, permitted variance, and raw samples are recorded.
- No more than three attempts are permitted to satisfy the variance gate.
- Portable benchmark results can be replayed from the emitted claim record.
- The runner can add capability and native inspection fields without changing
  its timing protocol.
- Performance records identify a physical device and reject MockGPU targets.

Performance thresholds are defined by `spec/performance-manifest.md`:

- **Competitive:** geometric-mean throughput ratio at least `0.80`, with every
  case at least `0.65` of the named baseline.
- **Parity:** geometric-mean throughput ratio at least `0.90`, with every case
  at least `0.80`.
- A retained claim fails on a same-environment median latency regression greater
  than 5% or on inspection drift.

## Phase 2: Scheduled IR And Capability Selection

Goal: establish the late-legalization framework without claiming optimized
eligibility before effects and layouts exist.

Compilation becomes:

```text
Tile IR
-> Scheduled Tile IR
-> capability eligibility and selection
-> Legalized Tile IR
-> tinygrad UOp Sink
```

Work:

- Preserve `TileCopy` and `TileMMA` through capability selection.
- Carry extents, masks, source/destination scopes, dtypes, accumulator dtype,
  alignment, contiguity, participant layouts, fragment layouts, pipeline stage,
  and requested mechanism. Missing facts make an optimized candidate ineligible.
- Implement `portable`, `auto`, and `native` selection semantics.
- Define versioned capability records containing compatible TileGrad and tinygrad
  revisions, target and renderer, operation and typed recognition path, dtypes,
  admitted shape set, layouts, alignment, storage scopes, participant geometry,
  required launch metadata, effects, synchronization/completion semantics,
  numerical mode, inspection signature, and correctness/performance gates.
- Record rejected native candidates and reasons in `auto` mode.
- Reject unsupported `native` requests instead of silently scalarizing.
- Make specialization and cache keys include the complete capability tuple and
  selected schedule.
- Convert canonical GEMM to use first-class `TileCopy` and `TileMMA`.
- Keep the current manual cooperative GEMM temporarily as a comparison baseline.

Gate:

- Tile operations disappear only after portable or profiled lowering is chosen.
- Portable selection and rejection of incomplete native candidates are
  deterministic from the capability tuple.
- Inspection reports the selected profile, layouts, and rejection reasons.
- Unsupported native requests and portable fallback are replayable from the
  capability record.
- Selection preserves current portable outputs and schedules on the current
  tested integration.

## Phase 3: Effects, Views, And Numerical Contract

Goal: make optimized collective lowering safe and inherited tinygrad numerics
testable.

Work:

- Add buffer access modes and base-allocation identity.
- Represent view offset, shape, strides, alignment, and writable-overlap facts.
- Prove copy disjointness or implement explicit snapshot semantics; reject
  unresolved overlap before parallel lowering.
- Track read/write effect regions and preserve RAW, WAR, WAW, and loop-carried
  dependencies.
- Define participant groups for collectives and validate convergence.
- Validate data races independently of layout coverage.
- Add a versioned numerical conformance manifest recording inherited tinygrad
  mode, rounding, reassociation, contraction, approximations, denormals, NaNs,
  infinities, signed zero, and operation/profile tolerances.
- Add portable/native differential tolerances before enabling mixed-precision
  native matrix lowering.

Gate:

- Capability selection has the alias, alignment, effect, and convergence facts
  required to legalize cooperative copy and MMA.
- Unknown overlap or collective convergence rejects with a source-level error.
- Effect tests cover aliases, masked tails, barriers, storage reuse, and nested
  ranges.
- The numerical manifest covers every dtype and operation admitted by the first
  Vector/Cooperative and Native Matrix capability records.

## Phase 4: Core Layouts

Goal: make ownership and data placement explicit enough for copy and matrix
lowering.

Implementation order:

1. Row-major and column-major layouts.
2. Blocked participant layouts.
3. Per-thread tile layouts.
4. Replicated layouts.
5. Reshape, repeat, and permutation composition.
6. Profile-owned fragment and shared-memory layouts.

Work:

- Give layouts a typed index-map representation.
- Validate coverage, replication, participant ownership, and race freedom.
- Make source-provided layouts exact.
- Make inferred layouts exact specialization data after selection.
- Use tinygrad `TensorCore` metadata for native atom mappings rather than
  duplicating renderer-internal lane maps.
- Add shared padding and swizzle representations only through declared profiles.

Gate:

- Incompatible explicit layouts reject instead of being reinterpreted.
- Layout tests cover tails, transpose, shared staging, fragments, and rendered
  participant mappings.
- Every performance capability names accepted layouts, alignment, and geometry.
- End-to-end capability selection consumes the effects and layout facts from
  Phases 3 and 4 before choosing an optimized lowering.
- Canonical GEMM retains selected participant, storage, and layout mappings
  through UOps and rendered code.

## Phase 5: Vector And Cooperative Copy

Goal: make `TileCopy` the optimized synchronous data-movement primitive.

Classifications:

- Scalar portable.
- Contiguous and coalesced.
- Vectorizable.
- Cooperative workgroup.
- Masked vector body with scalar tail.
- Layout transform and transpose.

Work:

- Select vector width from dtype, alignment, extent, target capability, and
  layout rather than treating `coalesced_width` as a promise.
- Implement global-to-global and global-to-shared paths first.
- Add shared-to-register/fragment paths when tinygrad exposes an appropriate
  typed representation.
- Preserve collective convergence and explicit barriers.
- Inspect both selected and rendered load/store widths.
- Request missing tinygrad primitives upstream instead of injecting renderer
  source.

Gate:

- Pass the synchronous-copy manifest suite for every dtype named by the claim.
- Cover aligned 4 MiB, 64 MiB, and 256 MiB copies, 2D copy, masked vector tail,
  and irregular-edge transpose.
- Report effective bandwidth, vector widths, workgroup geometry, and tail path.
- A published claim reaches at least the Competitive threshold.

The first claim is operation-scoped to synchronous copy. It does not imply
accelerated fill, map, or reduction; those require their own capability records
and benchmark suites after portable Core support exists.

## Phase 6: Native Mixed-Precision Matrix

Goal: publish one real Native Matrix capability tuple.

Work:

- Separate input, accumulator, and output dtype validation.
- Support FP16 or BF16 inputs with FP32 accumulation and an explicit output
  dtype.
- Select one renderer-advertised native matrix configuration supported by the
  current tested tinygrad integration.
- Initially use canonical multiply/reduce plus explicit `OptOps.TC` when that is
  the documented integration contract.
- Do not construct renderer-internal native matrix arguments in TileGrad.
- Prove the final typed native matrix operation in post-lowering UOps and
  rendered source.
- Use AMD MockGPU for the first end-to-end native matrix lowering and emulated
  correctness tests.
- Use synthetic SM75/80/89 CUDA renderer tests to validate CUDA native matrix
  selection and emitted source before physical hardware is available.
- Keep portable mixed-precision MMA for identical inputs, masks, and tails.
- Add fused activation epilogue support through normal TileGrad elementwise IR.

Gate:

- Run the complete Native Matrix suite from the performance manifest.
- Re-run MockGPU and renderer cases on the physical target before publishing a
  capability claim.
- Cover square, rectangular, edge-tiled, and fused-epilogue cases.
- At least one edge case exercises masked `TileCopy` and layout legalization.
- Report latency, TFLOP/s, native operation count/configuration, vector widths,
  shared-memory use, launch geometry, layouts, and schedule.
- The first release claim reaches at least the Competitive threshold against a
  mechanism-matched pinned TileLang baseline when TileLang supports the tuple.
  Otherwise it must name its recorded tinygrad revision or vendor baseline and
  must not use TileLang-like performance language.
- All latency, throughput, bandwidth, tuning, and resource results come from the
  physical device named by the capability tuple, never MockGPU.

## Phase 7: Reproducible Tuning

Goal: search schedules without violating exact source decisions.

Work:

- Define versioned bounded spaces over `BM`, `BN`, `BK`, participant layout,
  vector width, per-thread tile, native atom class, unroll, and shared staging.
- Keep pipeline depth fixed at one until Phase 8 provides real pipeline
  semantics.
- Search only parameters explicitly marked tunable.
- Key tuning results by the complete capability tuple and workload shape class.
- Record every attempted configuration, validation failure, compilation failure,
  and measured result.
- Separate tuning time from execution time.
- Support deterministic replay with tuning disabled.

Gate:

- Replaying a winner emits the same scheduled/legalized IR and inspection
  signature.
- No candidate bypasses correctness, capability, or resource validation.
- Published records name the tuning-space version and selected configuration.
- Tuned and untuned results are reported separately.

## Performance Alpha

The builder-first performance alpha requires:

- A versioned tinygrad integration contract.
- Manifest-compliant benchmark infrastructure.
- Scheduled/legalized IR and deterministic capability selection.
- Effect, alias, collective-legality, and numerical records for every claimed
  operation and dtype.
- Core layouts.
- Vector/cooperative synchronous copy.
- FP16/BF16 input with FP32 native accumulation.
- Reproducible tuning and inspection artifacts.
- At least one published Competitive Vector and Cooperative copy claim.
- At least one published Competitive Native Matrix capability claim.
- Physical-hardware benchmark artifacts for both performance claims; MockGPU
  artifacts are retained separately as correctness and compiler evidence.

The alpha does not require the decorator frontend, async copy, TMA, WGMMA,
clusters, warp specialization, or implicit autograd.

## Phase 8: Synchronous Software Pipeline

Goal: implement real pipeline structure before depending on asynchronous target
mechanisms.

Work:

- Preserve stage count and ordering in Tile IR.
- Add storage versioning, double buffering, dependency analysis, prologue,
  steady state, and epilogue.
- Keep copies synchronous initially and make barriers explicit.
- Make stage count tunable only when requested by the schedule.
- Extend the versioned tuning space with pipeline depth only after multi-stage
  correctness and inspection gates pass.
- Introduce async behavior later through typed tokens, commit, wait, and scoped
  synchronization supplied by tinygrad.

Gate:

- One-stage execution is equivalent to serial execution.
- Multi-stage execution passes tails and loop-carried dependency tests.
- Inspection proves alternating storage versions and required barriers.
- Any performance claim identifies the stage count and passes manifest timing
  and variance rules.

## Workload Milestones

### Batched Transpose

- Support dense and admitted strided 2D/3D inputs with shape-specialized launch
  geometry.
- Use vectorized global loads, register microtiles, shared-memory staging, and
  cooperative output stores.
- Add shared padding and profile-owned swizzles to avoid bank conflicts.
- Cover aligned tiles, irregular edges, non-vector-divisible tails, and batched
  strides.
- Use this as the first end-to-end proof of layouts plus Vector and Cooperative
  copy lowering.
- Add transpose shapes and bandwidth gates to the copy performance manifest.

### Quantization And Packed Formats

- Implement portable per-token, per-channel, and per-block abs-max scaling.
- Add FP8 casts and scale-factor outputs before FP4 and sub-byte packing.
- Represent scale-factor dtype, shape, stride, block geometry, and layout in
  capability records.
- Support precomputed scales, scale-only output, and cast-back references.
- Add packed load/store, unpacking, rounding, saturation, and zero behavior to
  the numerical conformance manifest.
- Add fused cast-and-transpose once standalone casting and transpose pass their
  correctness and performance gates.
- Cover aligned, strided, masked-tail, and irregular token/hidden dimensions.
- Publish per-token/per-channel/per-block suites through a performance-manifest
  revision before making throughput claims.

### Fused SwiGLU And Quantization

- Implement cast, select, clamp, exponential, reciprocal, and pointwise
  activation primitives through tinygrad scalar UOps.
- Fuse SwiGLU, optional routing weights, abs-max reduction, scale generation,
  and FP8 output in one kernel.
- Support optional token-to-top-k and token-to-expert mappings through ordinary
  indexed loads after alias and bounds validation.
- Add optional clamp counters only after typed atomics are available.
- Provide explicit forward and backward kernels with the same numerical record.
- Benchmark against an unfused tinygrad composition and a pinned matching fused
  baseline.

### Quantized GEMM

- Build portable unpack/dequantize references from the Quantization and Packed
  Formats milestone with FP32 accumulation.
- Add fused INT8 weight-only GEMM, then INT4 and admitted FP8/FP4 variants.
- Represent packed layouts, scales, and zero-points in capability records.
- Cover square, narrow-M decode, edge, and fused-epilogue workloads.
- Publish results only after a manifest revision defines the suite and baseline.

### Top-K Expert Selection

- Implement repeated maximum reduction with deterministic smallest-index tie
  breaking.
- Add replicated reducer state or an equivalent portable reduction mapping.
- Cover expert counts that do not align to subgroup or workgroup width.
- Support shape-specialized `k`, integer index outputs, masks, and negative
  infinity padding.
- Add subgroup acceleration only through a declared Subgroup capability.
- Differential-test every optimized path against the portable implementation.

### MoE Routing

- Implement token/expert counts, prefix sums, stable mappings, expansion, fused
  reduction, and routing-weight normalization.
- Add gather and scatter through validated indexed loads/stores.
- Use scans for offsets and typed atomics only where ownership cannot make writes
  disjoint.
- Cover empty experts, duplicated routes, uneven token counts, masked tensor
  parallel partitions, and deterministic ordering requirements.
- Add fused expansion/reduction only after standalone mapping operations pass.
- Define routing-specific correctness and performance suites before publishing
  claims.

### Normalization

- Implement correct sum/max reductions and required cast, exponential, and
  reciprocal-square-root operations.
- Add stable softmax with irregular tails.
- Add fused LayerNorm and RMSNorm with FP32 reduction state.
- Tune cooperative/vector loads, row ownership, and reduction layouts.
- Use the manifest protocol against a pinned matching baseline.

### Explicit Forward, Backward, And Multiple Outputs

- Support kernels returning multiple tensors and internal workspaces through
  normal `Tensor.custom_kernel` effects.
- Write explicit backward kernels for fused normalization, quantization,
  routing, and gating operations.
- Preserve saved intermediates without forcing host synchronization.
- Add partial-gradient outputs followed by explicit reduction kernels where a
  single launch cannot safely complete the reduction.
- Test lazy graph composition, `@function(precompile=True)`, and enclosing
  `TinyJit` replay for forward/backward pairs.
- Integrate tinygrad `grad_fxn` only as an optional wrapper around explicit
  kernels.

### Attention

- Build a synchronous tiled `QK^T -> online softmax -> PV` reference.
- Support causal and non-causal masks and irregular sequence/head dimensions.
- Use FP32 softmax state and native matrix lowering for both matrix operations.
- Tune block shapes, layouts, copy widths, and synchronous pipeline stages.
- Add decode and prefill suites through a manifest revision.
- Treat async copy, warp specialization, TMA, and WGMMA as later capabilities.

### Engram-Style Fused Kernels

- Start with a synchronous forward reference combining vector loads, dot and
  sum-of-squares reductions, reciprocal square root, gating activation, and
  fused output.
- Add a synchronous backward path with explicit partial weight-gradient output
  and a separate reduction kernel.
- Support admitted strided inputs, multiple outputs, saved intermediates, and
  shape-specialized hidden sizes.
- Add native matrix operations only where the dataflow contains eligible tile
  contractions.
- Tune block geometry, vector width, layouts, and synchronous pipeline stages.
- Add persistent scheduling only through the Scheduling profile.
- Add subgroup reductions, async copy, explicit waits, and double buffering only
  after tinygrad provides matching typed primitives and effects.
- Treat SM90/SM100-specific TMA, WGMMA, and warp-specialized variants as
  independent Target capability claims, not requirements for the portable
  workload.

## Core Language Completion

Goal: implement the full portable language required for Core 1.0 after the
performance architecture is proven.

Work:

- Add select, cast, bitcast, minimum, maximum, and the scalar operations needed
  by the Core workload gates.
- Add typed exponential, logarithm, reciprocal, square root, and reciprocal
  square root through tinygrad scalar UOps.
- Implement distributed `Fill` and `Clear` as first-class tile operations.
- Implement sum, product, minimum, and maximum reductions with explicit
  accumulator dtype and identity.
- Implement inclusive and exclusive sum/max scan on the portable path.
- Complete tile view, subview, transpose, reshape, and broadcast validation and
  lowering.
- Complete collective convergence, access-mode, alias, and race diagnostics for
  every Core operation.
- Extend the numerical conformance manifest and portable tests for each added
  operation and supported dtype.

Gate:

- All Core operations have positive, edge, rejection, and inspection tests.
- Unsupported target/dtype combinations reject instead of substituting types.
- Copy/transpose, portable mixed-precision GEMM, reductions/scans, softmax,
  LayerNorm, and fused GEMM epilogue workload gates pass.

## Core Frontend And Specialization

Goal: expose the completed language through both required frontends without
creating a second IR or runtime JIT.

Work:

- Implement the restricted Python `@tg.kernel` parser.
- Return a callable `TileKernel` that binds compile-time and shape-specialized
  values, validates repeated symbols, and allocates declared outputs.
- Make the decorator and `KernelBuilder` construct the same canonical Tile IR.
- Implement specialization keys containing dtype, rank, shape, admitted
  strides, target, layouts, numerical mode, profile/capability versions, tuned
  schedule and TileGrad revision.
- Cache scheduled/legalized Tile IR and UOp sinks while leaving binary and
  program caches to tinygrad.
- Construct normal `Tensor.custom_kernel` graphs without creating an internal
  `TinyJit`.
- Support execution inside `@function(precompile=True)` and an enclosing
  caller-owned `TinyJit`.

Gate:

- Equivalent decorator and builder programs produce identical canonical,
  scheduled, and legalized IR.
- Specializations differ whenever a value affecting generated UOps differs and
  reuse the cache otherwise.
- Output allocation, aliases, multiple kernel regions, and lazy return behavior
  pass correctness tests.
- Precompiled function and enclosing `TinyJit` capture/replay tests pass on the
  published target matrix.

## Core 1.0 Conformance

Core 1.0 requires:

- A restricted `@tg.kernel` frontend and the public builder frontend.
- Identical canonical Tile IR from both frontends.
- Complete specialization and cache identity for shapes, dtypes, layouts,
  profiles, target, revisions, and tuned schedule.
- Portable lowering on every target where the supported tinygrad integration can
  legally execute the required Core UOps.
- A published compatible and execution-tested target matrix.
- Composition inside `@function(precompile=True)` and enclosing `TinyJit`.
- Core workload gates for copy/transpose, mixed-precision portable GEMM,
  reductions/scans, softmax, LayerNorm, and fused GEMM epilogue.
- A complete versioned numerical conformance manifest for the published target
  and dtype matrix.

Core 1.0 conformance does not require Native Matrix support.

## Official TileGrad 1.0 Release

The official TileGrad 1.0 release requires:

- Core 1.0 conformance.
- At least one suitable Native Matrix capability tuple passing its correctness,
  native-emission, inspection, and performance-manifest gates.
- Published integration, numerical, capability, benchmark, and target-matrix
  artifacts for the release revisions.

## Parallel Tracks

### tinygrad Integration

- Stable target capability and resource-query API.
- Supported schedule-preserving `custom_kernel` contract.
- Native matrix request plus positive/negative inspection API.
- Typed vector load/store and shared-to-fragment operations.
- Composition coverage for `@function(precompile=True)` and `TinyJit`.
- Later typed packed formats, subgroup operations, atomics, and async tokens.

All integration remains behind `tinygrad_compat.py`. TileGrad must not depend on
renderer source strings or duplicate native instruction schemas.

### Correctness And Diagnostics

- Capability positive and rejection tests.
- Diagnostics naming operation, target, renderer, dtype, shape, layout, and
  missing feature.
- UOp and rendered-source inspection tests.
- Numerical mode and tolerance records.
- Portable/native differential tests.
- Compatible and execution-tested target matrix.

### Frontend And Specialization

The builder is sufficient for performance bring-up. The decorator frontend and
complete specialization cache are Core 1.0 blockers, not performance-alpha
blockers. Work may begin after canonical Tile IR stabilizes, but completion is
owned by the Core Frontend And Specialization phase and its equivalence gates.

## Deferred Profiles

Defer until synchronous Vector/Cooperative and Native Matrix claims are stable:

- Async copy, commit/wait groups, and transaction barriers.
- TMA-like transfers, WGMMA, TCGEN05, and tensor memory.
- Clusters and cooperative launch.
- Warp specialization and persistent kernels.
- Sparse native MMA.
- Broad target claims beyond recorded capability tuples.
- Automatic scheduling outside declared tuning spaces.
- Exact TileLang syntax compatibility.
- Implicit autograd beyond explicit backward kernels or tinygrad `grad_fxn`.
- Custom runtime/device backends.

These features require matching typed tinygrad operations, effects,
synchronization, renderer support, and launch metadata. They must not be claimed
through untyped source injection.

## Design Principles

- Keep TileGrad thin.
- Preserve exact schedules and tile intent until capability selection.
- Make every IR stage and fallback reason inspectable.
- Keep the portable path as the correctness oracle.
- Optimize through the same validation, capability, and inspection framework.
- Prefer one stable abstraction over speculative target APIs.
- Use tinygrad for scalar semantics, native atoms, codegen, compilation, caching,
  and runtime.
- Add missing backend mechanisms as small typed tinygrad primitives.
- Copy TileLang's useful programming model, not its backend complexity.
- Publish performance only from reproducible manifest artifacts.
