#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
    exec torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m plcdpo.train "$@"
fi
exec python -m plcdpo.train "$@"
