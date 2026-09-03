#!/bin/bash
# Batch-run get_nanogpt_weights.py over every checkpoint in one or more run directories.
# Each ckpt lands in <run-dir>/extracted/<ckpt-stem>/, the same default the exporter uses.
#
#   bash enviorments/extract_all.sh trained_weights/base005
#   bash enviorments/extract_all.sh runs/owt runs/xavier          # several runs at once
#   PYTHON=~/anaconda3/envs/nanoGPT/bin/python bash enviorments/extract_all.sh runs/owt
#   FORCE=1 bash enviorments/extract_all.sh runs/owt              # redo already-extracted ones
#
# No SBATCH header: extraction is CPU-only and takes seconds per checkpoint, so this runs
# fine on a login node or locally. Inside the container use:
#   apptainer exec "$SIF" bash enviorments/extract_all.sh runs/owt
#
# Already-extracted checkpoints are skipped, so re-running after new snapshots appear only
# does the new work. A checkpoint that fails to export does not stop the rest.

set -euo pipefail

PYTHON=${PYTHON:-python}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXPORTER=$ROOT/get_nanogpt_weights.py

(( $# )) || { echo "usage: $0 <run-dir> [run-dir ...]" >&2; exit 1; }

ok=0
skipped=0
failed=0

for dir in "$@"; do
    [[ -d "$dir" ]] || { echo "ERROR: $dir is not a directory" >&2; exit 1; }

    # -maxdepth 1 keeps extracted/ and any other nested directory out of the scan
    while IFS= read -r ckpt; do
        stem=$(basename "$ckpt" .pt)
        out=$dir/extracted/$stem

        # metadata.json is written last, so its presence means that export finished
        if [[ -f "$out/metadata.json" && -z "${FORCE:-}" ]]; then
            echo "skip  $ckpt  (already in $out)"
            skipped=$((skipped + 1))
            continue
        fi

        echo "=== $ckpt -> $out"
        if "$PYTHON" "$EXPORTER" --ckpt "$ckpt" --savefile "$out"; then
            ok=$((ok + 1))
        else
            echo "FAILED: $ckpt" >&2
            failed=$((failed + 1))
        fi
    done < <(find "$dir" -maxdepth 1 -name "*.pt" | sort)
done

echo
echo "extracted $ok, skipped $skipped, failed $failed"
(( failed == 0 ))