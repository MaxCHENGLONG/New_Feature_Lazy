"""
Check the hand-written forward pass against the training model.

Two comparisons on the same random token batch:
  1. loading from the export directory vs. loading from the raw ckpt -> must be identical
  2. gpt_forward.forward vs. model.GPT.forward -> must match to float32 round-off

  python inference/verify.py --ckpt trained_weights/base005/ckpt_0192000.pt \
      --weights trained_weights/base005/extracted/ckpt_0192000
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # repo root, for model.py
from gpt_forward import forward, load_weights
from model import GPT, GPTConfig

GPT2_VOCAB = 50257


def reference_logits(ckpt_path, idx):
    """Full-sequence logits from model.py. Passing targets makes it return all positions
    instead of just the last one."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
    model.eval()
    with torch.no_grad():
        logits, _ = model(idx, targets=idx)
    return logits


def report(name, a, b):
    diff = (a - b).abs().max().item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"{name:38s} max|diff| = {diff:.3e}   argmax agreement = {agree:.4%}")
    return diff, agree


def main():
    p = argparse.ArgumentParser(description="verify the hand-written forward pass")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--weights", required=True, help="export directory")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    idx = torch.randint(0, GPT2_VOCAB, (args.batch, args.seq))

    w_dir, cfg_dir = load_weights(args.weights)
    w_ckpt, cfg_ckpt = load_weights(args.ckpt)
    assert cfg_dir == cfg_ckpt, f"config mismatch: {cfg_dir} vs {cfg_ckpt}"

    ours_dir = forward(idx, w_dir, cfg_dir)
    ours_ckpt = forward(idx, w_ckpt, cfg_ckpt)
    ref = reference_logits(args.ckpt, idx)

    print()
    d1, _ = report("export dir vs raw ckpt", ours_dir, ours_ckpt)
    d2, a2 = report("gpt_forward vs model.py", ours_dir, ref)

    assert d1 == 0.0, "the two loading paths disagree"
    # logits are alpha x larger in the alpha runs, so float32 round-off scales with them
    tol = 1e-3 * max(1.0, cfg_dir["alpha"])
    assert d2 < tol and a2 == 1.0, "forward pass does not match model.py"
    print("\nOK")


if __name__ == "__main__":
    main()
