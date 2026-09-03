#!/bin/bash
#SBATCH -A naiss2026-4-1521-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -J frustration
#SBATCH -o /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/New_Feature_Lazy/logs/%x-%j.out

# Frustration of every snapshot of a run, several snapshots in parallel. Usage:
#   sbatch enviorments/frustration.sh /nobackup/.../runs/lazy                  # every ckpt_*.pt
#   sbatch enviorments/frustration.sh /nobackup/.../runs/lazy/ckpt_0001000.pt  # one checkpoint
#   sbatch enviorments/frustration.sh /nobackup/.../runs/lazy --T=4 --is_embed # extra args go to the script
#   NPROC=16 sbatch enviorments/frustration.sh /nobackup/.../runs/lazy         # more parallel processes
#   FORCE=1 sbatch enviorments/frustration.sh /nobackup/.../runs/lazy          # redo snapshots already done
#
# The computation is numpy on the CPU (the greedy gauge flipping is sequential, a GPU would
# not help); the single GPU is only what gets us a node on this partition. Each snapshot runs
# as its own process and writes <run>/balance_<iter>.json, so snapshots already done are
# skipped on a resubmit; at the end they are merged into <run>/balance.json. One process
# peaks at ~3GB of RAM for the 124M model at T=1 without embeddings.

set -euo pipefail

RUN=${1:?usage: frustration.sh <run-dir | ckpt.pt> [script args...]}
shift
NPROC=${NPROC:-8}

CACHE_DIR=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng
ROOT=$CACHE_DIR/New_Feature_Lazy        # this checkout: transformer_frustration/
OLD_ROOT=$CACHE_DIR/GPT2_Training       # holds nanogpt.sif

SIF=$OLD_ROOT/nanogpt.sif
[[ -f "$SIF" ]] || SIF=$ROOT/nanogpt.sif
[[ -f "$SIF" ]] || SIF=$ROOT/enviorments/nanogpt.sif
[[ -f "$SIF" ]] || { echo "ERROR: nanogpt.sif not found in $OLD_ROOT, $ROOT or $ROOT/enviorments" >&2; exit 1; }

# the script resolves transformer_frustration/ relative to the cwd
cd "$ROOT"
mkdir -p "$ROOT/logs"

# split the cores between the processes so the BLAS threads do not fight each other
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-$NPROC} / NPROC ))
[[ $OMP_NUM_THREADS -ge 1 ]] || export OMP_NUM_THREADS=1

SCRIPT=transformer_frustration/transformer_frustration_and_distance.py
echo "run=$RUN  image=$SIF  nproc=$NPROC  omp=$OMP_NUM_THREADS  args=$*"

if [[ -f "$RUN" ]]; then
    apptainer exec --bind /nobackup "$SIF" python "$SCRIPT" --weights "$RUN" "$@"
elif [[ -d "$RUN" ]]; then
    # one line per checkpoint still to do. balance_<iter>.json is named from the checkpoint's
    # iter_num, which is the number in the filename for the ckpt_NNNNNNN.pt snapshots
    todo=$(for f in "$RUN"/ckpt_*.pt; do
        [[ -e "$f" ]] || continue
        it=${f##*/ckpt_}; it=${it%.pt}
        if [[ -z "${FORCE:-}" && -f "$RUN/balance_$it.json" ]]; then continue; fi
        echo "$f"
    done)
    if [[ -n "$todo" ]]; then
        echo "$(echo "$todo" | wc -l) checkpoint(s) to process"
        # "$@" is expanded by this shell before xargs runs, so the extra args reach every process
        echo "$todo" | xargs -P "$NPROC" -I{} \
            apptainer exec --bind /nobackup "$SIF" python "$SCRIPT" --weights {} "$@" \
            || echo "WARNING: some snapshots failed (xargs status $?), merging the ones that finished" >&2
    else
        echo "nothing to compute: every ckpt_*.pt in $RUN already has a balance_*.json (FORCE=1 to redo)"
    fi
    # stitch the per-snapshot files into one balance.json with the whole-run layout
    apptainer exec --bind /nobackup "$SIF" python "$SCRIPT" --merge --weights "$RUN"
else
    echo "ERROR: $RUN is neither a run directory nor a checkpoint file" >&2
    exit 1
fi

echo "done."
ls -lh "$(dirname "$RUN")"/balance_*.json "$RUN"/balance_*.json 2>/dev/null || true
