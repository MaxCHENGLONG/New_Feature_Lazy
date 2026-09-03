"""
Frustration of the linearized transformer, seen as a signed weighted network, for every
snapshot of a run (or for one checkpoint). For each checkpoint the network is built by
transformer_utils.network_construction (residual stream, W_O W_V, W_1, W_2 and uniform causal
attention), the frustration index is computed by greedy gauge flipping, once on the real
weights and once on a weight-shuffled null model, and the L2 distance of the network
weights from the iter-0 snapshot is recorded.

  python transformer_frustration/transformer_frustration_and_distance.py \
      --weights /nobackup/.../runs/lazy                 # every ckpt_*.pt in the run dir
  python transformer_frustration/transformer_frustration_and_distance.py \
      --weights /nobackup/.../runs/lazy/ckpt_0001000.pt # a single checkpoint
  python transformer_frustration/transformer_frustration_and_distance.py \
      --weights <run-dir> --T 4 --is_embed --n_null 3   # longer context, embeddings, 3 nulls

Writes <run-dir>/balance.json (or --out). The distance column needs the iter-0 snapshot
(ckpt_0000000.pt, always written when snapshots are on), otherwise it is null.

enviorments/frustration.sh runs one process per snapshot, each writing balance_<iter>.json;
--merge stitches those back into one balance.json with the same layout as a whole-run call:

  python transformer_frustration/transformer_frustration_and_distance.py --merge --weights <run-dir>
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import glob
import numpy as np

from transformer_frustration.transformer_utils import load_model_information, network_construction
from transformer_frustration.compute_frustration import compute_frustration_edgelist


# the lists with one entry per snapshot; everything else in the json describes the run
PER_SNAPSHOT = ['epoch', 'loss', 'r_frust', 'r_spin', 'n_frust', 'n_spin', 'distance']


def merge(run_dir, out=None):
    """Merge <run_dir>/balance_<iter>.json (one per snapshot) into <run_dir>/balance.json."""
    assert os.path.isdir(run_dir), f"--merge needs a run directory, got {run_dir}"
    files = sorted(glob.glob(os.path.join(run_dir, 'balance_*.json')))
    assert files, f"no balance_*.json in {run_dir}"
    parts = []
    for f in files:
        with open(f) as fh:
            parts.append(json.load(fh))
    parts.sort(key=lambda r: r['epoch'][0])

    merged = {k: v for k, v in parts[0].items() if k not in PER_SNAPSHOT}
    merged['weights'] = os.path.abspath(run_dir)
    for r in parts:
        for k in ('T', 'is_embed'):
            assert r[k] == merged[k], f"{r['weights']} used {k}={r[k]}, the others {merged[k]}"
    for k in PER_SNAPSHOT:
        merged[k] = [x for r in parts for x in r[k]]

    out = out or os.path.join(run_dir, 'balance.json')
    with open(out, 'w') as fh:
        json.dump(merged, fh, indent=4)
    print(f"merged {len(parts)} snapshot file(s) into {out}: "
          f"iter {merged['epoch'][0]:,} .. {merged['epoch'][-1]:,}")


def run_real(rows, cols, vals):
    frustr, s = compute_frustration_edgelist(rows, cols, vals, verbose=False)
    return float(frustr), s.tolist()


def run_null(rows, cols, vals):
    vals_shuffled = vals.copy()
    np.random.shuffle(vals_shuffled)
    frustr, s = compute_frustration_edgelist(rows, cols, vals_shuffled, verbose=False)
    return float(frustr), s.tolist()


def main():
    p = argparse.ArgumentParser(description="frustration of the linearized transformer network")
    p.add_argument("--weights", required=True,
                   help="run directory holding ckpt_*.pt snapshots, or a single checkpoint .pt")
    p.add_argument("--T", type=int, default=1, help="context length of the unrolled network")
    p.add_argument("--is_embed", action="store_true", help="include the wte/wpe embedding block")
    p.add_argument("--n_null", type=int, default=1, help="shuffled-weight null models per snapshot (0 skips)")
    p.add_argument("--save_spin", action="store_true",
                   help="also store the +1/-1 gauge vectors (one entry per node, large)")
    p.add_argument("--seed", type=int, default=0, help="seed for the greedy flips and the null shuffles")
    p.add_argument("--out", help="output json. default <run-dir>/balance.json or <ckpt-dir>/balance_<iter>.json")
    p.add_argument("--merge", action="store_true",
                   help="compute nothing: merge <run-dir>/balance_*.json (one per snapshot) into <run-dir>/balance.json")
    args = p.parse_args()

    if args.merge:
        merge(args.weights, args.out)
        return

    if os.path.isdir(args.weights):
        files = sorted(glob.glob(os.path.join(args.weights, "ckpt_*.pt")))
        assert files, f"no ckpt_*.pt found in {args.weights}"
        out = args.out or os.path.join(args.weights, "balance.json")
    else:
        assert os.path.isfile(args.weights), f"{args.weights} is neither a directory nor a file"
        files = [args.weights]
        out = args.out  # resolved below, once the iter number is known
    print(f"{len(files)} checkpoint(s) from {args.weights}")

    np.random.seed(args.seed)

    save_data = {
        'weights': os.path.abspath(args.weights),
        'T': args.T,
        'is_embed': args.is_embed,
        'seed': args.seed,
        'std': None,
        'train_config': None,
        'epoch': [],
        'loss': [],
        'r_frust': [],
        'r_spin': [],
        'n_frust': [],   # one list of n_null values per snapshot
        'n_spin': [],
        'distance': [],
    }

    vals0 = None  # network weights at iter 0, for the distance column

    def init_vals(sibling_of):
        # single-checkpoint mode never sees the iter-0 snapshot itself; look for it next door
        init_path = os.path.join(os.path.dirname(os.path.abspath(sibling_of)), 'ckpt_0000000.pt')
        if not os.path.isfile(init_path):
            print(f'no {init_path}: distance column will be null')
            return None
        print(f'Loading the init from {init_path} for the distance column')
        info0 = load_model_information(init_path)
        assert info0['iter'] == 0, f"{init_path} is iter {info0['iter']}, not the init"
        return network_construction(info0, args.T, is_embed=args.is_embed)[2]

    for load_path in files:
        print(f'Loading model from {load_path}')
        model_info = load_model_information(load_path)
        epoch = model_info['iter']
        rows, cols, vals = network_construction(model_info, args.T, is_embed=args.is_embed)

        save_data['epoch'].append(epoch)
        save_data['loss'].append(model_info['loss'])
        if save_data['std'] is None:
            save_data['std'] = model_info['std']
            save_data['train_config'] = model_info['config']
        if epoch == 0:
            vals0 = vals
        elif vals0 is None:
            vals0 = init_vals(load_path)

        # Real model
        frust, s = run_real(rows, cols, vals)
        save_data['r_frust'].append(frust)
        save_data['r_spin'].append(s if args.save_spin else None)

        # Null models: same graph, weights shuffled over the edges
        n_frust, n_spin = [], []
        for _ in range(args.n_null):
            frust_n, s_n = run_null(rows, cols, vals)
            n_frust.append(frust_n)
            n_spin.append(s_n if args.save_spin else None)
        save_data['n_frust'].append(n_frust)
        save_data['n_spin'].append(n_spin)

        # Distance from init, over the network weights
        dist = float(np.sqrt(np.sum((vals - vals0) ** 2))) if vals0 is not None else None
        save_data['distance'].append(dist)

        null_str = f"{np.mean(n_frust):.4f}" if n_frust else "-"
        dist_str = f"{dist:.3e}" if dist is not None else "-"
        print(f"iter {epoch:>7,}  loss {model_info['loss']:.4f}  frustration {frust:.4f}  "
              f"null {null_str}  distance {dist_str}  edges {len(vals):,}")

        if out is None:
            out = os.path.join(os.path.dirname(os.path.abspath(load_path)), f"balance_{epoch:07d}.json")
        # write after every checkpoint, so a long run that gets killed still leaves its results
        with open(out, 'w') as file:
            json.dump(save_data, file, indent=4)

    print(f"\nwrote {out} ({len(save_data['epoch'])} snapshot(s))")


if __name__ == "__main__":
    main()
