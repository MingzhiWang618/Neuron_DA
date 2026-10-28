#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"

cd "$ROOT"
exec conda run --no-capture-output -n pytc python "$HERE/train.py" \
  --config "$HERE/config_cremi_c.yaml" \
  --contrastive-config "$HERE/contrastive.yaml" \
  "$@"
