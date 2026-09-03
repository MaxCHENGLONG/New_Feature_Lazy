#!/bin/bash
#SBATCH -A naiss2026-4-1521-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -J train_nanogpt
#SBATCH -o /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/New_Feature_Lazy/logs/%x-%j.out

# Train GPT-2 124M on 1 node x 4 GH200 = 4 ranks. Usage:
#   sbatch enviorments/train_nano.sh smoke   # shakespeare_char, ~100 iters, validates the DDP path
#   sbatch enviorments/train_nano.sh         # openwebtext, ~10.1B tokens
#   sbatch enviorments/train_nano.sh full --batch_size=64 --gradient_accumulation_steps=8
#   sbatch enviorments/train_nano.sh full xavier --init_dist=xavier_uniform --seed=0
# the second positional arg is an optional run name (-> runs/<name>, its own checkpoints);
# anything starting with -- is passed straight through to train.py
#
# Single node on purpose: this cluster has no InfiniBand and no libfabric/aws-ofi-nccl
# plugin in the image, so NCCL falls back to TCP across nodes and the gradient allreduce
# dominates. Measured 8 ranks over 2 nodes at ~12% MFU vs single-node NVLink. Revisit
# -N 2 only once NCCL can drive the Slingshot (hsn*) NICs natively.
#
# train.py is driven by RANK/LOCAL_RANK/WORLD_SIZE, so we let srun place the ranks
# directly instead of nesting torchrun inside srun.

set -euo pipefail

MODE=${1:-full}
shift || true
# optional run name as the next positional arg -- anything starting with -- is a train.py
# override instead. Each run name gets its own out_dir, so parallel experiments never share
# a ckpt.pt (which would make one of them silently resume from another's weights).
RUN=""
if [[ $# -ge 1 && "$1" != --* ]]; then RUN=$1; shift; fi

CACHE_DIR=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng
ROOT=$CACHE_DIR/New_Feature_Lazy        # this checkout: train.py, model.py, configurator.py
OLD_ROOT=$CACHE_DIR/GPT2_Training       # the previous checkout: holds nanogpt.sif and the 17G dataset
# the SBATCH -o path above must exist before sbatch, or Slurm drops the log: mkdir -p $ROOT/logs once

SIF=$OLD_ROOT/nanogpt.sif
[[ -f "$SIF" ]] || SIF=$ROOT/nanogpt.sif
[[ -f "$SIF" ]] || SIF=$ROOT/enviorments/nanogpt.sif
[[ -f "$SIF" ]] || { echo "ERROR: nanogpt.sif not found in $OLD_ROOT, $ROOT or $ROOT/enviorments" >&2; exit 1; }

# train.py resolves configurator.py / data/ / out_dir relative to the cwd
cd "$ROOT"

# rendezvous for torch.distributed
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
# this image has no libfabric/aws-ofi-nccl plugin, so NCCL cannot drive the Slingshot NICs
# natively and falls back to TCP. Left alone it picks nsc-eth, the slow management network;
# restrict it to the hsn* interfaces so at least the TCP traffic rides the fast fabric.
export NCCL_SOCKET_IFNAME=hsn
export NCCL_CROSS_NIC=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# lets `pkill -ABRT python` dump every rank's Python stack into this log when it hangs
export PYTHONFAULTHANDLER=1

if [[ "$MODE" == "smoke" ]]; then
    DATASET=shakespeare_char
    OUT_DIR=$CACHE_DIR/runs/smoke-${RUN:-$SLURM_JOB_ID}
    ARGS=(--dataset=$DATASET --n_layer=4 --n_head=4 --n_embd=256
          --block_size=256 --batch_size=16 --gradient_accumulation_steps=8
          --max_iters=100 --lr_decay_iters=100 --warmup_iters=10
          --eval_interval=50 --eval_iters=20)
else
    DATASET=openwebtext
    # keyed by run name, NOT by job id: the run is longer than one walltime slot, so
    # successive jobs of the same experiment must find the previous checkpoint here
    OUT_DIR=$CACHE_DIR/runs/${RUN:-owt}
    ARGS=(--dataset=$DATASET)
    if [[ -f "$OUT_DIR/ckpt.pt" ]]; then
        echo "found $OUT_DIR/ckpt.pt -- resuming"
        ARGS+=(--init_from=resume)
    fi
fi
mkdir -p "$OUT_DIR" "$ROOT/logs"

# train.py reads data/<dataset>/ relative to the cwd. The tokenized .bin files were unpacked
# once under the old checkout; link them in file by file rather than copying 17G. The
# directory itself already exists here (prepare.py, readme.md), so it cannot be replaced by
# a link: `ln -s <dir> <existing dir>` would silently nest the link inside it
mkdir -p "$ROOT/data/$DATASET"
for f in train.bin val.bin meta.pkl; do
    if [[ ! -e "$ROOT/data/$DATASET/$f" && -f "$OLD_ROOT/data/$DATASET/$f" ]]; then
        ln -s "$OLD_ROOT/data/$DATASET/$f" "$ROOT/data/$DATASET/$f"
        echo "linked $ROOT/data/$DATASET/$f -> $OLD_ROOT/data/$DATASET/$f"
    fi
done
[[ -e "$ROOT/data/$DATASET/train.bin" ]] || { echo "ERROR: no train.bin in $ROOT/data/$DATASET or $OLD_ROOT/data/$DATASET" >&2; exit 1; }

echo "mode=$MODE  image=$SIF  nodes=$SLURM_NNODES  ranks=$SLURM_NTASKS"
echo "master=$MASTER_ADDR  out_dir=$OUT_DIR"

# pull the .bin files into every node's page cache first. get_batch does ~1000 small
# random reads per iteration across the ranks, and on a cold shared filesystem each one
# is a network round trip -- that alone costs more than the step it feeds.
echo "warming page cache with $(du -shL $ROOT/data/$DATASET/ | cut -f1) of $DATASET ..."
time srun --ntasks-per-node=1 bash -c "cat $ROOT/data/$DATASET/*.bin > /dev/null"

# sample GPU power/utilisation on this node for the duration of the run
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv -l 30 > "$ROOT/logs/gpu-$SLURM_JOB_ID.csv" 2>/dev/null &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

# the args are passed as positional parameters, NOT interpolated into the script text,
# so that newlines inside ARGS stay word separators instead of becoming command separators
srun --gpu-bind=none apptainer exec --nv --bind /nobackup "$SIF" bash -c '
    : "${SLURM_PROCID:?not visible inside the container -- cannot assign DDP ranks}"
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS
    # triton resolves libcuda.so through this image ldconfig cache, which points at a
    # compat path that apptainer --nv does not populate; point it at the real driver
    _libcuda=$(find /.singularity.d/libs /usr/lib64 /usr/lib/aarch64-linux-gnu \
                    /usr/local/cuda/compat/lib -name "libcuda.so.1" 2>/dev/null | head -1)
    if [ -n "$_libcuda" ]; then
        export TRITON_LIBCUDA_PATH=$(dirname "$_libcuda")
        [ "$RANK" = 0 ] && echo "TRITON_LIBCUDA_PATH=$TRITON_LIBCUDA_PATH"
    else
        echo "WARNING: no libcuda.so.1 found in the container -- torch.compile will fail" >&2
    fi
    exec python train.py "$@"
' _ "${ARGS[@]}" --out_dir="$OUT_DIR" "$@"

echo "done. checkpoints in $OUT_DIR"
ls -lh "$OUT_DIR"