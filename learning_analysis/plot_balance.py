"""
Plot the frustration / loss / distance-from-init curves of one or more runs from their
balance.json (transformer_frustration_and_distance.py), one column per run, in the layout of
analysis_frustration.ipynb. Headless, so it runs on the cluster login node:

  python learning_analysis/plot_balance.py --out learning_analysis/attn_wd0 \
      --run runs/rich_bs1wd0/balance.json   'Rich (init scale $= 1$, wd $= 0$)' rich \
      --run runs/lazy_attn5_wd0/balance.json  'QKVO $\\times 5$, wd $= 0$' \
      --run runs/lazy_attn10_wd0/balance.json 'QKVO $\\times 10$, wd $= 0$'

--run takes FILE TITLE [COLOR]; COLOR is lazy (orange, the default), rich (blue) or any
matplotlib colour. Writes <out>.pdf and <out>.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# NeurIPS: 5.5 in text width, Times body font, embedded TrueType (no Type 3) fonts in the PDF.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {"lazy": "#eb6834", "rich": "#2a78d6"}
NULL_COLOR = "#898781"


def init_edge_norm(d):
    """||theta_0||: L2 norm of the network edge weights at init (T=1, no embeddings).

    The edges are W1, W2 and the product WO@WV of every block (transformer_utils.network_construction);
    the identity / attention edges are constants and cancel in the distance. Every weight is
    N(0, std^2) times the multiplier its group got at init, so E||W||^2 = numel * var. balance.json
    carries those multipliers as init_scales (c_attn = WV, attn.c_proj = WO, c_fc = W1,
    mlp.c_proj = W2). Files written before that key existed only had init_block_scale (missing
    from the alpha runs: 1.0) and the residual rescale, so for them it is rebuilt from train_config.
    """
    tc = d["train_config"]
    dim, L, std = tc["n_embd"], tc["n_layer"], tc["init_std"]
    s = d.get("init_scales")
    if s is None:
        blk = tc.get("init_block_scale", 1.0)
        proj = tc["init_proj_scale"] * (1 / np.sqrt(2 * L) if tc["init_scale_residual"] else 1.0)
        s = {"c_attn": blk, "attn.c_proj": blk * proj, "c_fc": blk, "mlp.c_proj": blk * proj}
    v = {k: (std * s[k]) ** 2 for k in ("c_attn", "attn.c_proj", "c_fc", "mlp.c_proj")}
    per_layer = (4 * dim * dim * v["c_fc"] + 4 * dim * dim * v["mlp.c_proj"]   # W1 + W2
                 + dim * dim * (dim * v["c_attn"] * v["attn.c_proj"]))         # WO@WV: d products per entry
    return np.sqrt(L * per_layer)


def plot_runs(runs, out):
    """runs: [(file, panel title, colour), ...] -> 3 rows x len(runs) columns, saved as out.pdf / out.png"""
    fig, axes = plt.subplots(3, len(runs), figsize=(5.5, 4.8), sharex=True, constrained_layout=True, squeeze=False)
    for col, (fname, title, color) in enumerate(runs):
        d = json.load(open(fname))
        tc = d["train_config"]
        if d["T"] != 1 or d["is_embed"]:
            print(f"WARNING: {fname} was computed with T={d['T']} is_embed={d['is_embed']}; "
                  f"the ||theta_0|| normalisation assumes T=1 without embeddings")
        it = np.array(d["epoch"])  # training iterations
        ax_frust, ax_loss, ax_dist = axes[:, col]

        ax_frust.plot(it, d["r_frust"], color=color, label="real")
        if all(d["n_frust"]):  # --n_null=0 leaves the null lists empty
            ax_frust.plot(it, [v[0] for v in d["n_frust"]], color=NULL_COLOR, ls="--", label="null")
        ax_frust.legend(loc="lower right", frameon=False)
        ax_frust.set_title(f"({'abcdefgh'[col]}) {title}")

        # loss is only re-evaluated every eval_interval iterations; snapshots in between repeat the stale value
        evaluated = it % tc["eval_interval"] == 0
        ax_loss.plot(it[evaluated], np.array(d["loss"])[evaluated], color=color)

        # the json stores the absolute L2 distance ||theta_t - theta_0|| over the edge weights, which
        # scales with the init magnitude; divide by ||theta_0|| so the runs are comparable
        norm0 = init_edge_norm(d)
        rel = np.array(d["distance"], dtype=float) / norm0
        print(f"{title}: ||theta_0|| = {norm0:.1f}, final relative distance = {rel[-1]:.3f}")
        ax_dist.plot(it, rel, color=color)
        ax_dist.set_xlabel("Training iteration")
        ax_dist.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))

    # same y scale across the runs in every row (each column joins the first one's group)
    for row in axes:
        for ax in row[1:]:
            ax.sharey(row[0])
    axes[0, 0].set_ylabel("Frustration")
    axes[1, 0].set_ylabel("Loss")
    axes[2, 0].set_ylabel(r"$\|\theta_t - \theta_0\| \, / \, \|\theta_0\|$")
    for ax in axes.flat:
        ax.grid(axis="y", color="#e1e0d9", lw=0.5)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", action="append", required=True, metavar="ARG",
                   help="FILE TITLE [COLOR]: a balance.json, its panel title, and lazy | rich | any "
                        "matplotlib colour (default lazy). Repeat for every column")
    p.add_argument("--out", required=True, help="output path without extension; .pdf and .png are written")
    args = p.parse_args()
    runs = []
    for r in args.run:
        assert 2 <= len(r) <= 3, f"--run takes FILE TITLE [COLOR], got {r}"
        color = r[2] if len(r) == 3 else "lazy"
        runs.append((r[0], r[1], COLORS.get(color, color)))
    plot_runs(runs, args.out)


if __name__ == "__main__":
    main()
