#!/bin/bash
#SBATCH -A naiss2025-22-1730-gpu
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -J prepare_nanogpt
#SBATCH -o /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/GPT2_Training/logs/%x-%j.out

# Tokenize a dataset into train.bin/val.bin. Usage:
#   sbatch enviorments/prepare_data.sh                 # openwebtext (default)
#   sbatch enviorments/prepare_data.sh shakespeare_char
# No GPU is used here; the --gres line is only kept because this partition is gpu-only.

set -euo pipefail

DATASET=${1:-openwebtext}

CACHE_DIR=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng
ROOT=$CACHE_DIR/GPT2_Training

# the .sif may sit in the repo root or under enviorments/, depending on how it was built
SIF=$ROOT/nanogpt.sif
[[ -f "$SIF" ]] || SIF=$ROOT/enviorments/nanogpt.sif
[[ -f "$SIF" ]] || { echo "ERROR: nanogpt.sif not found in $ROOT or $ROOT/enviorments" >&2; exit 1; }
echo "using image: $SIF"

# keep the ~22.5GB HuggingFace cache on /nobackup, NOT in $HOME (quota)
export HF_HOME=$CACHE_DIR/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_DATASETS_CACHE"

echo "preparing dataset: $DATASET"
df -h "$CACHE_DIR" | tail -1

apptainer exec --bind /nobackup \
    --env HF_HOME="$HF_HOME" \
    --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    "$SIF" python "$ROOT/data/$DATASET/prepare.py"

echo "done, resulting files:"
ls -lh "$ROOT/data/$DATASET/"