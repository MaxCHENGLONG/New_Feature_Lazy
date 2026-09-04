"""Plot real vs null frustration over epochs for the feature and lazy runs on one figure.

Left panel: frustration itself. Right panel: change of the real frustration from its
value at iter 0, on a symlog axis so the three runs (which differ by 1000x) are all visible.

Usage:
    python learning_analysis/plot_frustration.py

Writes frustration.png next to this file.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent

TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"

# (file, label, color) -- categorical slots 1..3
RUNS = [
    ("feature_balance.json", "feature", "#2a78d6"),
    ("lazy_alpha8balance.json", "lazy α=8", "#1baf7a"),
    ("lazy_balance.json", "lazy α=32", "#eb6834"),
]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

for json_name, run, color in RUNS:
    d = json.load(open(HERE / json_name))
    epoch = d["epoch"]
    real = d["r_frust"]
    null = [x[0] for x in d["n_frust"]]

    ax.plot(epoch, real, color=color, lw=2, ls="-", label=f"{run} real")
    ax.plot(epoch, null, color=color, lw=2, ls="--", label=f"{run} null")
    # no direct labels on the left panel: the two lazy runs sit within 1e-3 of each other

    delta = [x - real[0] for x in real]
    ax2.plot(epoch, delta, color=color, lw=2, label=run)
    ax2.annotate(run, (epoch[-1], delta[-1]), xytext=(6, 0),
                 textcoords="offset points", va="center", color=TEXT, fontsize=8)

ax.set_title("Frustration vs epoch", color=TEXT, fontsize=12, loc="left")
ax.set_ylabel("Frustration", color=MUTED)
ax.legend(frameon=False, loc="lower right", fontsize=8)

ax2.set_title("Change of real frustration from init", color=TEXT, fontsize=12, loc="left")
ax2.set_ylabel("Frustration(t) − Frustration(0)", color=MUTED)
ax2.set_yscale("symlog", linthresh=1e-4)   # 1e-4 is about the noise floor of the greedy flips
ax2.axhline(0, color=GRID, lw=0.8)
ax2.legend(frameon=False, loc="upper center", fontsize=8)

for a in (ax, ax2):
    a.set_xlabel("Epoch", color=MUTED)
    a.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        a.spines[side].set_color(GRID)
    a.tick_params(colors=MUTED)

fig.tight_layout()
out = HERE / "frustration.png"
fig.savefig(out)
print("saved", out)
