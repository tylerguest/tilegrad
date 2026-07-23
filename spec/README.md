# TileGrad specification

`tilegrad-spec.tex` is the normative north-star specification for TileGrad 1.0.
It defines a TileLang-like programming language that uses tinygrad UOps as its
sole normative backend IR while inheriting tinygrad scalar semantics, native
matrix configuration, codegen, and runtime.

TileGrad's `@tg.kernel` frontend parses and specializes tile programs, then
constructs `Tensor.custom_kernel` calls. tinygrad's `@function`, `TinyJit`,
compiled-program cache, and runtime own precompilation, capture, and replay.

The specification separates a stable Core language from Native Matrix,
Vector and Cooperative, Scheduling, Async Pipeline, Atomics, Subgroup, and
Target profiles. Core is portable to every target on which the pinned tinygrad
integration can legally lower and execute the required Core UOps. Optional
profiles are advertised through versioned capability tuples rather than broad
backend-family claims.

Performance claims are capability-specific. The
[`performance-manifest.md`](performance-manifest.md) file defines benchmark
protocols and the thresholds for competitive and parity claims. Core
conformance alone does not imply a performance claim, and unsupported native or
asynchronous mechanisms must fall back only in `auto` mode or reject when
requested exactly.

`tilegrad-spec.tex` is the normative source. `tilegrad-spec.pdf` is its generated
rendering for convenient reading.

Render the committed PDF after changing the source:

```bash
./render.sh
```

The script uses `tectonic` when available and otherwise falls back to
`pdflatex`.
