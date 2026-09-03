"""
Evaluate every snapshot of a run offline: val loss / perplexity, plus per-tensor weight
statistics and per-block activation statistics. Writes two CSVs into the run directory.

$ python eval.py --run_dir=/nobackup/.../runs/base
$ python eval.py --run_dir=... --eval_batches=50   # quicker, fewer val batches

Every checkpoint is scored on the SAME deterministic, non-overlapping slices of val.bin,
so numbers are comparable across snapshots and across runs.
"""
import os
import csv
import glob
import math
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
run_dir = '/nobackup/proj/disk/naiss2025-22-1730/personal/licheng/runs/base'
dataset = '' # '' = read it from each checkpoint's stored config
eval_batches = 0 # how many val batches per checkpoint. 0 = the whole val set
batch_size = 32
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# -----------------------------------------------------------------------------
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
    torch.backends.cuda.enable_cudnn_sdp(True)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpts = sorted(glob.glob(os.path.join(run_dir, 'ckpt_*.pt')))
assert ckpts, f"no ckpt_*.pt found in {run_dir}"
print(f"found {len(ckpts)} snapshots in {run_dir}")

# resolve the dataset from the first checkpoint if it was not given explicitly
first = torch.load(ckpts[0], map_location='cpu', weights_only=False)
dataset = dataset or first['config']['dataset']
block_size = first['model_args']['block_size']
# alpha-centered runs train logits - logits_at_init, so score them the same way: the frozen
# init is the iter-0 snapshot, always written when snapshots are on
alpha_center = first['config'].get('alpha_center', False)
model0 = None
if alpha_center:
    ck0 = torch.load(os.path.join(run_dir, 'ckpt_0000000.pt'), map_location=device, weights_only=False)
    assert ck0['iter_num'] == 0, "alpha_center needs the untrained init in ckpt_0000000.pt"
    model0 = GPT(GPTConfig(**ck0['model_args']))
    model0.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ck0['model'].items()})
    model0.eval().to(device)
    print("alpha_center=True: subtracting the iter-0 logits from every snapshot")
val_path = os.path.join('data', dataset, 'val.bin')
data = np.memmap(val_path, dtype=np.uint16, mode='r')
print(f"dataset={dataset}  val.bin={len(data):,} tokens  block_size={block_size}")

# fixed, non-overlapping slices so every snapshot sees exactly the same tokens
n_blocks = (len(data) - 1) // block_size
n_blocks -= n_blocks % batch_size
if eval_batches:
    n_blocks = min(n_blocks, eval_batches * batch_size)
starts = np.arange(n_blocks) * block_size
print(f"evaluating on {n_blocks // batch_size} batches x {batch_size} x {block_size} "
      f"= {n_blocks * block_size:,} tokens")

def val_batches():
    for i in range(0, n_blocks, batch_size):
        sel = starts[i:i + batch_size]
        x = np.stack([data[s:s + block_size] for s in sel]).astype(np.int64)
        y = np.stack([data[s + 1:s + 1 + block_size] for s in sel]).astype(np.int64)
        yield (torch.from_numpy(x).to(device, non_blocking=True),
               torch.from_numpy(y).to(device, non_blocking=True))

def tensor_stats(t):
    t = t.detach().float()
    return (t.mean().item(), t.std().item(), t.abs().max().item(), t.norm().item())

loss_rows, stat_rows = [], []
for path in ckpts:
    ck = torch.load(path, map_location=device, weights_only=False)
    it = ck['iter_num']

    model = GPT(GPTConfig(**ck['model_args']))
    sd = ck['model']
    for k in list(sd):                      # torch.compile prefixes the keys
        if k.startswith('_orig_mod.'):
            sd[k[len('_orig_mod.'):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(device)

    # per-tensor weight statistics
    for name, p in model.named_parameters():
        mean, std, absmax, l2 = tensor_stats(p)
        stat_rows.append((it, 'weight', name, p.numel(), mean, std, absmax, l2))

    # per-block activation statistics, captured on the first batch only
    acts = {}
    def make_hook(name):
        def hook(mod, inp, out):
            acts[name] = tensor_stats(out)
        return hook
    handles = [b.register_forward_hook(make_hook(f'block_{i}'))
               for i, b in enumerate(model.transformer.h)]
    handles.append(model.transformer.ln_f.register_forward_hook(make_hook('ln_f')))

    total_loss, n = 0.0, 0
    with torch.inference_mode():
        with ctx:
            for x, y in val_batches():
                ref = model0(x, y)[0] if model0 is not None else None
                _, loss = model(x, y, ref_logits=ref)
                total_loss += loss.item()
                n += 1
                if n == 1:                  # activations only need one batch
                    for h in handles:
                        h.remove()
                    handles = []
    for h in handles:
        h.remove()

    val_loss = total_loss / n
    loss_rows.append((it, val_loss, math.exp(val_loss)))
    for name, (mean, std, absmax, l2) in acts.items():
        stat_rows.append((it, 'act', name, 0, mean, std, absmax, l2))
    print(f"iter {it:>7,}  val_loss {val_loss:.4f}  ppl {math.exp(val_loss):9.2f}")

    del model, ck
    torch.cuda.empty_cache()

loss_csv = os.path.join(run_dir, 'eval_loss.csv')
with open(loss_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['iter', 'val_loss', 'perplexity'])
    w.writerows(sorted(loss_rows))

stat_csv = os.path.join(run_dir, 'eval_stats.csv')
with open(stat_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['iter', 'kind', 'name', 'numel', 'mean', 'std', 'absmax', 'l2'])
    w.writerows(sorted(stat_rows, key=lambda r: (r[0], r[1], r[2])))

print(f"\nwrote {loss_csv} ({len(loss_rows)} rows)")
print(f"wrote {stat_csv} ({len(stat_rows)} rows)")