#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for geometry in circle rectangles; do
  for n in 50 100 200; do
    python3 -m harandi.fem --geometry "$geometry" --n "$n" --output "results/fem_${geometry}_n${n}"
  done
done
python3 -m pytest -q tests/test_physics.py tests/test_fem.py tests/test_author_reference.py
