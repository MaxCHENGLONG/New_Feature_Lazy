#!/bin/bash
#SBATCH -A naiss2026-4-1521-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:20:00
#SBATCH -J sample_nanogpt
#SBATCH -o /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/New_Feature_Lazy/logs/%x-%j.out

# Generate text from one checkpoint of a run with inference/run.py. Usage:
#   sbatch enviorments/sample.sh a1                                   # runs/a1/ckpt.pt, prompt "\n"
#   sbatch enviorments/sample.sh a1 --prompt "The meaning of life is" --num_samples 3
#   sbatch enviorments/sample.sh a1 ckpt_0050000.pt --prompt "Once upon a time" --max_new_tokens 100
# the second positional arg is an optional checkpoint file inside runs/<run-name> (default
# ckpt.pt, the latest); anything starting with -- is passed straight through to run.py
# (--prompt, --max_new_tokens, --temperature, --top_k, --num_samples, --seed).
# The generated text lands in the Slurm log above.

set -euo pipefail

RUN=${1:?usage: sbatch enviorments/sample.sh <run-name> [ckpt.pt] [--flag value ...]}
shift
CKPT=ckpt.pt
if [[ $# -ge 1 && "$1" != --* ]]; then CKPT=$1; shift; fi

CACHE_DIR=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng
ROOT=$CACHE_DIR/New_Feature_Lazy        # this checkout: inference/run.py
OLD_ROOT=$CACHE_DIR/GPT2_Training       # the previous checkout: holds nanogpt.sif

SIF=$OLD_ROOT/nanogpt.sif
[[ -f "$SIF" ]] || SIF=$ROOT/nanogpt.sif
[[ -f "$SIF" ]] || SIF=$ROOT/enviorments/nanogpt.sif
[[ -f "$SIF" ]] || { echo "ERROR: nanogpt.sif not found in $OLD_ROOT, $ROOT or $ROOT/enviorments" >&2; exit 1; }

cd "$ROOT"
mkdir -p "$ROOT/logs"

CKPT_PATH=$CACHE_DIR/runs/$RUN/$CKPT
[[ -f "$CKPT_PATH" ]] || { echo "ERROR: $CKPT_PATH does not exist" >&2; exit 1; }

echo "sampling from $CKPT_PATH with $SIF"
apptainer exec --nv --bind /nobackup "$SIF" python inference/run.py --weights "$CKPT_PATH" "$@"
