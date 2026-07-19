#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if command -v tectonic >/dev/null 2>&1; then
  tectonic tilegrad-spec.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error tilegrad-spec.tex
  pdflatex -interaction=nonstopmode -halt-on-error tilegrad-spec.tex
  rm -f tilegrad-spec.aux tilegrad-spec.log tilegrad-spec.out tilegrad-spec.toc
else
  echo "error: install tectonic or pdflatex to render tilegrad-spec.pdf" >&2
  exit 1
fi

echo "done: tilegrad-spec.pdf"
