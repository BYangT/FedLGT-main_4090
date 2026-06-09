#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        PYTHON_BIN="python3"
    fi
fi

ENTRY_SCRIPT="${ENTRY_SCRIPT:-fed_main_sae.py}"
CLIENTS=(10 15 20)

for n_parties in "${CLIENTS[@]}"; do
    results_dir="./all_results/voc/client_${n_parties}"
    mkdir -p "$results_dir"

    echo "============================================================"
    echo "Running VOC with n_parties=${n_parties}"
    echo "results_new=${results_dir}"
    echo "============================================================"

    "$PYTHON_BIN" "$ENTRY_SCRIPT" \
        --dataset voc \
        --n_parties "$n_parties" \
        --results_new "$results_dir" \
        --batch_size 32 \
        --test_batch_size 32 \
        --learn_emb_type clip \
        --subspace_param_unlearn \
        --subspace_rank 256 \
        --forget_subspace_weight 2.0 \
        --subspace_align_weight 0.1 \
        "$@"
done
