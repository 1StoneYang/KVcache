#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

source_file="${1:-research_direction_archive.tex}"

if [[ ! -f "$source_file" ]]; then
  printf 'LaTeX source not found: %s\n' "$source_file" >&2
  exit 1
fi

if ! command -v xelatex >/dev/null 2>&1; then
  printf 'xelatex is required but was not found in PATH.\n' >&2
  exit 1
fi

# Three passes resolve the table of contents, references, and PDF bookmarks.
for pass in 1 2 3; do
  printf 'XeLaTeX pass %d/3: %s\n' "$pass" "$source_file"
  xelatex -interaction=nonstopmode -halt-on-error "$source_file"
done

printf 'Built %s\n' "${source_file%.tex}.pdf"
