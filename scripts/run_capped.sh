#!/bin/bash
# Run a LADA experiment with host RAM hard-capped at 20 GB (cgroup v2 via systemd-run).
# - GPU memory is capped to 16 GB inside main.py (RTX 5070 Ti has 16 GB).
# - tqdm progress bars print during training.
#
# Usage:
#   bash scripts/run_capped.sh                              # default: 16-shot order-I
#   bash scripts/run_capped.sh scripts/run_TAIL_fullshot.sh # any other script
set -e

SCRIPT="${1:-scripts/run_TAIL_16shot.sh}"
RAM_CAP="${RAM_CAP:-20G}"
REPO="/mnt/work/trn/works/research_paper/lada"

source /home/trn/miniconda3/etc/profile.d/conda.sh
conda activate lada
cd "$REPO"

echo "Running '$SCRIPT' with RAM capped at $RAM_CAP (GPU capped at 16 GB in main.py)"
exec systemd-run --user --scope -p MemoryMax="$RAM_CAP" -p MemorySwapMax=0 --quiet \
    bash "$SCRIPT"
