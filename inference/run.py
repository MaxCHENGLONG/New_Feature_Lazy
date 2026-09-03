"""
Generate text with the hand-written forward pass in gpt_forward.py.

  python inference/run.py --weights trained_weights/base005/extracted/ckpt_0192000 \
      --prompt "The meaning of life is"
  python inference/run.py --weights trained_weights/base005/ckpt_0192000.pt --num_samples 3
"""
import argparse
import os
import sys

import tiktoken
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt_forward import generate, load_weights

GPT2_VOCAB = 50257  # real BPE vocab; the model's 50304 is that padded up for speed


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser(description="sample from extracted nanoGPT weights")
    p.add_argument("--weights", required=True, help="export directory or ckpt .pt file")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=default_device())
    args = p.parse_args()

    torch.manual_seed(args.seed)
    w, cfg = load_weights(args.weights, device=args.device)
    print(f"loaded {args.weights} on {args.device}: {cfg}")

    enc = tiktoken.get_encoding("gpt2")
    start_ids = enc.encode(args.prompt, allowed_special={"<|endoftext|>"})
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

    for _ in range(args.num_samples):
        y = generate(x, w, cfg, args.max_new_tokens, temperature=args.temperature,
                     top_k=args.top_k, valid_vocab=GPT2_VOCAB)
        print(enc.decode(y[0].tolist()))
        print("---------------")


if __name__ == "__main__":
    main()
