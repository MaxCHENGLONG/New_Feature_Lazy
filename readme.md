# GPT2_Training

nanoGPT-style GPT-2 (124M) training, plus a from-scratch forward pass that runs on the
exported weights.

```
data/<dataset>/         train.bin / val.bin (tokenized, uint16)
train.py, model.py      training and the model definition
get_nanogpt_weights.py  checkpoint -> per-tensor weight export
inference/              hand-written forward pass (gpt_forward.py) + CLI (run.py)
enviorments/            Slurm / apptainer wrappers for the cluster
```

---

## 1. Unpack the dataset

The tokenized dataset ships as a tar archive on the server; do **not** re-run
`data/<dataset>/prepare.py`. Unpack it so that `train.bin` and `val.bin` end up directly
in `data/<dataset>/` — that is where `train.py` looks (`data_dir = data/<dataset>`).

```bash
# from the repository root
tar -xzvf <archive>.tar.gz -C data/<dataset>/

# e.g. openwebtext
mkdir -p data/openwebtext
tar -xzvf openwebtext.tar.gz -C data/openwebtext/
```

Use `-xvf` instead of `-xzvf` if the archive is not gzip-compressed. Check the layout
before extracting (`tar -tzvf <archive>.tar.gz | head`): if the archive already contains a
top-level directory, extract into `data/` instead and let it create `data/<dataset>/`.

Verify:

```bash
ls -lh data/openwebtext/          # expect train.bin (~17GB) and val.bin (~8.5MB)
```

For a character-level dataset such as `shakespeare_char`, `meta.pkl` must be unpacked
alongside the two `.bin` files — `train.py` reads it to recover `vocab_size`.

---

## 2. Export the weights — `get_nanogpt_weights.py`

Turns one training checkpoint into a directory of individual `.pt` tensors (fused and
per-head), plus a `metadata.json` describing every shape. The export uses the
`[in_features, out_features]` convention, so nanoGPT's `nn.Linear` matrices are transposed
on the way out.

```bash
python get_nanogpt_weights.py --ckpt <run-dir>/<ckpt>.pt [--savefile <out-dir>]
```

| argument | meaning |
| --- | --- |
| `--ckpt` | path to a nanoGPT checkpoint `.pt` (required) |
| `--savefile` | output directory; default `<ckpt-dir>/extracted/<ckpt-stem>/` |

```bash
# single checkpoint, default output location
python get_nanogpt_weights.py --ckpt trained_weights/base005/ckpt_0192000.pt
#   -> trained_weights/base005/extracted/ckpt_0192000/

# explicit output directory
python get_nanogpt_weights.py --ckpt trained_weights/base005/ckpt_0000000.pt \
    --savefile /tmp/init_weights/
```

Output layout:

```
<out-dir>/
  fused/      wte, wpe, lm_head, W_Q/K/V/O, W_fc, W_proj, ln1/ln2/ln_f weights (+ bias files)
  per_head/   W_Q/K/V/O_heads  [n_layer, n_head, ...]
  metadata.json
```

Notes:

- Runs trained with `bias=False` have no biases anywhere; the `b_*.pt` / `*_bias.pt` files
  are written as **all zeros** so loaders that read them by name keep working (adding zero
  is a no-op). A checkpoint trained with `bias=True` is rejected with an assertion.
- `lm_head.pt` is the output projection. With `tie_weights=True` (the default) it is the same
  tensor as `wte.pt`; with `tie_weights=False` it is an independent matrix — always read
  `lm_head.pt`, never `wte.pt`, as the head. `metadata.json` records `tie_weights`.
  `vocab_size` is the padded 50304.
- The `--alpha` runs multiply the logits by `alpha` at forward time; that factor is **not**
  baked into `lm_head.pt`. `metadata.json` records it as a top-level `alpha` (1.0 for ordinary
  runs) and `inference/gpt_forward.py` applies it after the head, so sampling and `verify.py`
  match `model.py`. Exports made before this key existed are read as `alpha = 1.0`.
- The `init_*` settings of the run (`init_std`, `init_wte_scale`, `init_wpe_scale`,
  `init_attn_scale`, `init_mlp_scale`, `init_head_scale`, ...) **are** baked into the saved
  tensors, since `model.py` applies them in place at init. `metadata.json` records them as a
  top-level `init` (defaults filled in for older checkpoints), whose `tensor_scale` gives the
  multiplier each exported tensor started with relative to a plain `init_std` draw
  (`wte`, `wpe`, `W_Q/W_K/W_V`, `W_O`, `W_fc`, `W_proj`, `lm_head`).

Batch export — every checkpoint in one or more run directories, skipping the ones already
done:

```bash
bash enviorments/extract_all.sh trained_weights/base005
bash enviorments/extract_all.sh runs/owt runs/xavier     # several runs
FORCE=1 bash enviorments/extract_all.sh runs/owt         # redo existing exports
PYTHON=/path/to/python bash enviorments/extract_all.sh runs/owt
```

---

## 3. Forward pass / sampling — `inference/run.py`

`inference/gpt_forward.py` re-implements the GPT-2 forward pass with explicit matmuls and
softmaxes (no `model.py` import). `run.py` is the CLI on top of it.

```bash
python inference/run.py --weights <export-dir-or-ckpt.pt> [options]
```

`--weights` accepts **either** form:

- an export directory from step 2 (`.../extracted/ckpt_0192000`) — read from `fused/`
  + `metadata.json`;
- a raw checkpoint `.pt` — the same de-prefixing and transposing is applied on the fly.

| option | default | meaning |
| --- | --- | --- |
| `--weights` | — | export directory or checkpoint `.pt` (required) |
| `--prompt` | `"\n"` | prompt text, GPT-2 BPE encoded |
| `--max_new_tokens` | `200` | tokens to generate |
| `--temperature` | `0.8` | `<1` sharper, `>1` more random |
| `--top_k` | `200` | keep only the k most likely tokens per step |
| `--num_samples` | `1` | how many completions to draw |
| `--seed` | `1337` | RNG seed |
| `--device` | auto | `cuda` → `mps` → `cpu`, in that order |

```bash
# from an export directory
python inference/run.py --weights trained_weights/base005/extracted/ckpt_0192000 \
    --prompt "The meaning of life is"

# straight from a checkpoint, three samples
python inference/run.py --weights trained_weights/base005/ckpt_0192000.pt --num_samples 3

# greedy-ish, short, on CPU
python inference/run.py --weights <export-dir> --temperature 0.1 --top_k 1 \
    --max_new_tokens 50 --device cpu
```

Sampling is restricted to the first 50257 rows of the logits (the real GPT-2 BPE vocab),
so the 47 padding rows can never be emitted.

To check this forward pass against `model.py`, give `verify.py` both the checkpoint and its
export directory:

```bash
python inference/verify.py --ckpt <run-dir>/<ckpt>.pt --weights <export-dir>
```

---

## 4. Weight initialization

Initialization only takes effect for a **from-scratch** run (`--init_from=scratch`, the
default); `resume` and `gpt2*` load existing weights instead.

```bash
# local / single GPU
python train.py --init_dist=trunc_normal --init_std=0.015 --seed=0
python train.py --init_dist=xavier_uniform --init_gain=1.0 --seed=1

# on the cluster: <mode> <run-name> <train.py overrides ...>
sbatch enviorments/train_nano.sh full xavier --init_dist=xavier_uniform --seed=0
sbatch enviorments/train_nano.sh smoke --init_dist=uniform --init_std=0.02
```

Each run name gets its own `out_dir`, so parallel init sweeps never share a checkpoint.

### Options

| flag | default | effect |
| --- | --- | --- |
| `--init_from` | `scratch` | `scratch` \| `resume` \| `gpt2*` (OpenAI weights). The knobs below apply to `scratch` only |
| `--init_dist` | `normal` | `normal` \| `trunc_normal` \| `uniform` \| `xavier_normal` \| `xavier_uniform` |
| `--init_std` | `0.02` | std of `normal` / `trunc_normal` / `uniform`; ignored by the xavier variants |
| `--init_gain` | `1.0` | gain of `xavier_normal` / `xavier_uniform`; ignored by the others |
| `--init_scale_residual` | `True` | multiply every `c_proj.weight` by `1/sqrt(2*n_layer)` (GPT-2 paper) |
| `--init_proj_scale` | `1.0` | extra multiplier on every `c_proj.weight`, composed with the above. `1e-3` or `0.0` gives the saddle-to-saddle init |
| `--init_block_scale` | `1.0` | multiplier on every Linear inside the blocks (`c_attn` = Q/K/V, `attn.c_proj` = O, `c_fc`, `mlp.c_proj`), i.e. those start at std `init_std * init_block_scale`; `wte` / `wpe` / LayerNorm stay at `init_std`. Applied before `init_proj_scale`, composes with it. Same as setting `--init_attn_scale` and `--init_mlp_scale` together |
| `--init_wte_scale` | `1.0` | multiplier on the token embedding `wte`: it starts at std `init_std * init_wte_scale`. With `tie_weights=True` (default) `wte` **is** `lm_head`, so this and `--init_head_scale` multiply the same tensor; pass `--tie_weights=False` to set embedding and head apart |
| `--init_wpe_scale` | `1.0` | multiplier on the position embedding `wpe`: it starts at std `init_std * init_wpe_scale` |
| `--init_attn_scale` | `1.0` | multiplier on `c_attn` (Q/K/V) and `attn.c_proj` (O) in every block: they start at std `init_std * init_attn_scale` |
| `--init_mlp_scale` | `1.0` | multiplier on `c_fc` and `mlp.c_proj` in every block: they start at std `init_std * init_mlp_scale` |
| `--init_head_scale` | `1.0` | multiplier on `lm_head.weight` (and on `wpe` when tied, to keep the token/position balance). `>> 1`: lazy regime set by the readout magnitude, pass `learning_rate / scale` yourself; `<< 1`: output starts near 0 |
| `--alpha` | `1.0` | logits are multiplied by `alpha` at forward time; `learning_rate` and `min_lr` are divided by `alpha` (not `alpha^2`: AdamW's step is set by `lr`, not the gradient). No weight is scaled |
| `--seed` | `1337` | changes the draw (and the data order) |

What each distribution does to `nn.Linear` and `nn.Embedding` weights:

| `init_dist` | draw |
| --- | --- |
| `normal` | `N(0, init_std)` — the GPT-2 default |
| `trunc_normal` | `N(0, init_std)` truncated to `±2*init_std` |
| `uniform` | `U(-a, a)` with `a = init_std*sqrt(3)`, variance-matched to `normal` |
| `xavier_normal` | `xavier_normal_(gain=init_gain)` — fan-in/fan-out scaled |
| `xavier_uniform` | `xavier_uniform_(gain=init_gain)` |

Not configurable: all `nn.Linear` biases start at zero, LayerNorm is `weight=1` /
`bias=0`, and the residual rescale is applied **after** the draw (in place), so it composes
with any `init_dist`.

The run prints the settings it used, and every checkpoint stores them in its `config`
dict — `metadata.json` carries them through to the export as `source.train_config`:

```
weight init: dist=normal, std=0.02, gain=1.0, scale_residual=True, proj_scale=1.0, block_scale=1.0, wte_scale=1.0, wpe_scale=1.0, attn_scale=1.0, mlp_scale=1.0, head_scale=1.0, alpha=1.0
```

The per-group multipliers (`init_wte_scale`, `init_wpe_scale`, `init_attn_scale`,
`init_mlp_scale`, `init_head_scale`) are independent: set one alone or any combination, and
each group starts at std `init_std * <its scale>`. Everything is a scalar multiplier applied in
place after the draw, so `init_block_scale`, `init_proj_scale` and the residual rescale simply
multiply on top (e.g. `attn.c_proj` starts at
`init_std * init_block_scale * init_attn_scale * init_proj_scale / sqrt(2*n_layer)`).
`GPTConfig.init_scales()` returns the multiplier every tensor ends up with (`wte`, `wpe`,
`c_attn`, `attn.c_proj`, `c_fc`, `mlp.c_proj`, `lm_head`); the export writes it to
`metadata.json` as `init.tensor_scale` and the frustration script to `balance.json` as
`init_scales`, so the per-group init std is recoverable from either file.

A from-scratch run also writes `<out_dir>/init_info.json` (and prints the same table) stating
what the command did to the weights: the `command`, the `overrides` it made to the defaults,
one `effect` line per overridden init flag naming the weight groups it touched, the full
`knobs` -> groups map, and per group the multiplier, the std measured right after init, the
shape and the tensor names:

```
"effect": ["init_scale_residual=False: attn.c_proj, mlp.c_proj", "init_attn_scale=5.0: c_attn, attn.c_proj"],
"groups": {
  "c_attn": {"scale": 5.0, "from": "init_block_scale=1.0 * init_attn_scale=5.0", "std": 0.0999,
             "shape": [2304, 768], "tensors": ["transformer.h.0.attn.c_attn.weight", ...]},
  ...
```

### Feature learning vs lazy learning

All three share `std=0.02` for every matrix (so the GPT-2 `1/sqrt(2*n_layer)` rescale is
switched off), LayerNorm `weight=1` / `bias=0`, and the default AdamW settings.

```bash
# baseline
python train.py --init_scale_residual=False
# feature learning: residual-branch output projections start at std = 0.02 * 1e-3 (or 0.0)
python train.py --init_scale_residual=False --init_proj_scale=1e-3
# lazy learning: logits * 32, learning_rate and min_lr / 32
python train.py --init_scale_residual=False --alpha=32.0
# init-scale sweep on the block matrices only: Q/K/V/O/fc/proj start at std = 0.02 * scale,
# wte / wpe stay at 0.02 so the residual stream and the output logits keep their scale
python train.py --init_scale_residual=False --init_block_scale=0.1    # towards feature
python train.py --init_scale_residual=False --init_block_scale=5.0    # towards lazy
# per-group init scale: each of the four knobs alone, or any combination (std = 0.02 * scale)
python train.py --init_scale_residual=False --init_attn_scale=5.0     # Q/K/V/O only
python train.py --init_scale_residual=False --init_mlp_scale=0.1      # fc / mlp.c_proj only
python train.py --init_scale_residual=False --init_wpe_scale=0.1      # position embedding only
python train.py --init_scale_residual=False --init_wte_scale=3.0      # token embedding (and lm_head, since tied)
python train.py --init_scale_residual=False --init_head_scale=8.0     # lm_head (and wte / wpe, since tied)
# token embedding and head set apart, plus wpe, attention and MLP, all at once
python train.py --init_scale_residual=False --tie_weights=False --init_wte_scale=0.5 \
    --init_wpe_scale=2.0 --init_attn_scale=5.0 --init_mlp_scale=0.2 --init_head_scale=8.0
```

On the cluster the same overrides go after the run name. Every run name is its own
`runs/<name>`, and a `full` job resumes from `runs/<name>/ckpt.pt` when it exists, so give
each init its own name and resubmit a continuation with the identical flags:

```bash
sbatch enviorments/train_nano.sh smoke attn5 --init_scale_residual=False --init_attn_scale=5.0   # ~100 iters, check first
sbatch enviorments/train_nano.sh full attn5 --init_scale_residual=False --init_attn_scale=5.0
sbatch enviorments/train_nano.sh full mlp0.1 --init_scale_residual=False --init_mlp_scale=0.1
sbatch enviorments/train_nano.sh full wpe0.1 --init_scale_residual=False --init_wpe_scale=0.1
# lazy via the readout: lm_head (and the tied wte / wpe) x8, learning_rate and min_lr / 8
sbatch enviorments/train_nano.sh full head8 --init_scale_residual=False --init_head_scale=8.0 \
    --learning_rate=7.5e-5 --min_lr=7.5e-6
# untied: token embedding and head set apart
sbatch enviorments/train_nano.sh full wte0.5_head8 --init_scale_residual=False --tie_weights=False \
    --init_wte_scale=0.5 --init_head_scale=8.0 --learning_rate=7.5e-5 --min_lr=7.5e-6
# a sweep: one job per value
for s in 0.1 0.5 2.0 5.0; do
    sbatch enviorments/train_nano.sh full attn$s --init_scale_residual=False --init_attn_scale=$s
done
```

Snapshot `ckpt_0000000.pt` is iteration 0, i.e. the untrained initialization itself — export
it with step 2 to inspect the initial weights directly.

---

## 5. Frustration — `transformer_frustration/`

Treats the linearized transformer (residual stream, `W_O W_V`, `W_1`, `W_2`, uniform causal
attention; LayerNorm gains, Q/K and the head are ignored) as a signed weighted network and
computes its frustration index by greedy gauge flipping, once on the real weights and once
on a weight-shuffled null model, plus the L2 distance of the network weights from iter 0.
CPU/numpy only, no GPU needed, but the 124M model gives ~64M edges per snapshot at `T=1`,
so run it on a compute node, not the login node.

```bash
# every ckpt_*.pt of a run -> <run-dir>/balance.json
python transformer_frustration/transformer_frustration_and_distance.py --weights <run-dir>
# a single checkpoint -> <ckpt-dir>/balance_<iter>.json
python transformer_frustration/transformer_frustration_and_distance.py --weights <run-dir>/ckpt_0001000.pt

# on the cluster: one process per snapshot, NPROC (default 8) in parallel, one
# balance_<iter>.json each, merged into <run-dir>/balance.json at the end; snapshots that
# already have their file are skipped (FORCE=1 redoes them)
sbatch enviorments/frustration.sh <run-dir>
sbatch enviorments/frustration.sh <run-dir>/ckpt_0000000.pt        # time one snapshot first
NPROC=16 sbatch enviorments/frustration.sh <run-dir> --T=4 --is_embed  # extra args reach the script

# merge by hand (e.g. after adding snapshots), no computation
python transformer_frustration/transformer_frustration_and_distance.py --merge --weights <run-dir>
```

| option | default | meaning |
| --- | --- | --- |
| `--weights` | — | run directory with `ckpt_*.pt`, or one checkpoint `.pt` (required) |
| `--T` | `1` | context length the network is unrolled over |
| `--is_embed` | off | include the `wte` / `wpe` embedding block (adds ~39M edges) |
| `--n_null` | `1` | shuffled-weight null models per snapshot, `0` skips them |
| `--save_spin` | off | also store the ±1 gauge vector per node (large json) |
| `--seed` | `0` | seed for the greedy flips and the shuffles |
| `--out` | see above | output json path |

The json holds, per snapshot: `epoch`, `loss`, `r_frust` (real), `n_frust` (list of nulls),
`distance`, plus the run's `std` (`init_std`), `init_scales` (the multiplier each weight group
got on top of it at init: `wte`, `wpe`, `c_attn`, `attn.c_proj`, `c_fc`, `mlp.c_proj`,
`lm_head`) and `train_config`, so the file records which regime it came from.

### Plot

`learning_analysis/plot_balance.py` draws the figure of `analysis_frustration.ipynb` (rows:
real vs null frustration, loss at the evaluated iterations, relative distance
`||theta_t - theta_0|| / ||theta_0||`) with one column per `balance.json`. It is headless, so
it runs on the login node straight from the run directories, no job needed:

```bash
CACHE_DIR=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng
apptainer exec --bind /nobackup $CACHE_DIR/GPT2_Training/nanogpt.sif \
    python learning_analysis/plot_balance.py --out learning_analysis/attn_wd0 \
    --run $CACHE_DIR/runs/rich_bs1wd0/balance.json    'Rich (init scale $= 1$, wd $= 0$)' rich \
    --run $CACHE_DIR/runs/lazy_attn5_wd0/balance.json  'QKVO $\times 5$, wd $= 0$' \
    --run $CACHE_DIR/runs/lazy_attn10_wd0/balance.json 'QKVO $\times 10$, wd $= 0$' \
    --run $CACHE_DIR/runs/lazy_attn20_wd0/balance.json 'QKVO $\times 20$, wd $= 0$'
```

`--run` takes `FILE TITLE [COLOR]` (`lazy` = orange, the default; `rich` = blue; or any
matplotlib colour), one per column, and `--out` is written as `.pdf` and `.png`. The distance
row is normalised by `||theta_0||` computed from the json's `init_scales`, so runs whose init
scaled only Q/K/V/O or only the MLP are comparable; files from before that key existed fall
back to `init_block_scale`.
