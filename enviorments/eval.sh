#!/bin/bash
#SBATCH -A naiss2026-4-1521-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -J eval_nanogpt
#SBATCH -o /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/New_Feature_Lazy/logs/%x-%j.out

# Score every snapshot of a run. Usage:
#   sbatch enviorments/eval.sh base
#   sbatch enviorments/eval.sh base --eval_batches=50
# Single GPU is enough -- this is a forward-only sweep.

set -euo pipefail

RUN=${1:?usage: sbatch enviorments/eval.sh <run-name> [--flag=value ...]}
shift

CACHE_DIR=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng
ROOT=$CACHE_DIR/New_Feature_Lazy        # this checkout: the alpha-aware eval.py / model.py
OLD_ROOT=$CACHE_DIR/GPT2_Training       # the previous checkout: holds nanogpt.sif

SIF=$OLD_ROOT/nanogpt.sif
[[ -f "$SIF" ]] || SIF=$ROOT/nanogpt.sif
[[ -f "$SIF" ]] || SIF=$ROOT/enviorments/nanogpt.sif
[[ -f "$SIF" ]] || { echo "ERROR: nanogpt.sif not found in $OLD_ROOT, $ROOT or $ROOT/enviorments" >&2; exit 1; }

# eval.py reads data/<dataset>/val.bin relative to the cwd; train_nano.sh links it in here
cd "$ROOT"
mkdir -p "$ROOT/logs"

RUN_DIR=$CACHE_DIR/runs/$RUN
[[ -d "$RUN_DIR" ]] || { echo "ERROR: $RUN_DIR does not exist" >&2; exit 1; }

echo "evaluating $RUN_DIR with $SIF"
apptainer exec --nv --bind /nobackup "$SIF" python eval.py --run_dir="$RUN_DIR" "$@"

echo "done:"
ls -lh "$RUN_DIR"/eval_*.csv