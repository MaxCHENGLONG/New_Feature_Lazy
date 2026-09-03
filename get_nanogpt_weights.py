# nanoGPT Checkpoint -> Complete Weight Extraction
# ------------------------------------------
# Extract ALL weights from a nanoGPT training checkpoint into the same layout
# get_GPT2_weights.py produces for HuggingFace GPT-2, so the same downstream
# *_Inference.py / analysis code can read either one.
#
# Usage:
#   python get_nanogpt_weights.py --ckpt trained_weights/base005/ckpt_0000000.pt
#   python get_nanogpt_weights.py --ckpt .../ckpt_0192000.pt --savefile my_dir/
# Default savefile is <ckpt_dir>/extracted/<ckpt_stem>/
#
# Output structure: identical to get_GPT2_weights.py (per_head/, fused/, metadata.json)
#
# ================== Differences from HuggingFace GPT-2 ==================
#
# 1. WEIGHT ORIENTATION (this is the big one).
#    HF GPT-2 uses Conv1D, whose weight is [in_features, out_features].
#    nanoGPT uses nn.Linear, whose weight is [out_features, in_features].
#    So every attention/MLP matrix here IS transposed relative to HF and we
#    transpose it back, to keep the saved [in, out] convention identical.
#      c_attn.weight  (3*embed, embed) -> .T -> (embed, 3*embed)
#      c_proj.weight  (embed, embed)   -> .T -> (embed, embed)
#      mlp.c_fc       (4*embed, embed) -> .T -> (embed, 4*embed)
#      mlp.c_proj     (embed, 4*embed) -> .T -> (4*embed, embed)
#
# 2. NO BIASES. These runs train with bias=False: attention, MLP and all three
#    LayerNorms have weight only. We still write the b_*.pt / *_bias.pt files as
#    all-zero tensors so downstream code that loads them by name keeps working
#    (adding zero is a no-op). metadata.json records bias=false.
#
# 3. PADDED VOCAB. vocab_size is 50304 (50257 padded up to a multiple of 64 for
#    speed), so wte is [50304, embed_dim]. Saved as-is, not truncated.
#
# 4. LM HEAD. lm_head.pt is always written. With tie_weights=True (GPT-2 default) it is
#    the same tensor as wte.pt; with tie_weights=False it is an independent matrix.
#    metadata.json records tie_weights.
#
# 5. ALPHA. The alpha runs multiply the logits by alpha at forward time (model.py); that
#    factor is NOT baked into lm_head.pt. metadata.json records it as a top-level "alpha"
#    (1.0 for ordinary runs) and inference/gpt_forward.py applies it after the head.
#
# Unchanged from HF and safe to reuse:
#   - QKV layout: c_attn output is viewed as (3, n_head, head_dim) with 3 as the
#     slowest dim, i.e. flat [Q | K | V] blocks, each split into contiguous heads.
#     Same as HF's split(n_embd) + contiguous head slicing.
#   - Activation: nn.GELU(approximate='tanh') == HF 'gelu_new'.
#   - Pre-norm: ln_1 before attention, ln_2 before MLP, ln_f at the end.
#   - LayerNorm eps: 1e-5.

import argparse
import json
import os
import torch

LN_EPS = 1e-5              # hardcoded in model.py's LayerNorm.forward
HIDDEN_ACT = "gelu_new"    # nn.GELU(approximate='tanh')
GPT2_VOCAB = 50257         # real GPT-2 BPE vocab, which nanoGPT pads up for speed


def load_state_dict(ckpt_path):
    """Load the checkpoint and return (clean state_dict, model_args, checkpoint)."""
    # weights_only=False because our checkpoints carry the model_args/config dicts too
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # torch.compile wraps the model, so every key is prefixed with _orig_mod.
    sd = {k.removeprefix("_orig_mod."): v.float() for k, v in ckpt["model"].items()}
    return sd, ckpt["model_args"], ckpt


def extract(sd, model_args):
    """Split the state_dict into the per-head / fused tensors of the target format."""
    num_layers = model_args["n_layer"]
    num_heads = model_args["n_head"]
    embed_dim = model_args["n_embd"]
    head_dim = embed_dim // num_heads
    intermediate_size = sd["transformer.h.0.mlp.c_fc.weight"].shape[0]  # nn.Linear: [out, in]
    # every b_*.pt / *_bias.pt below is written as zeros, which is only correct when the run
    # had no biases to begin with. Fail loudly rather than silently export wrong weights.
    assert not model_args["bias"], (
        "this checkpoint was trained with bias=True, but the exporter writes all biases as "
        "zeros. Read the real biases out of the state_dict before exporting."
    )

    print(f"\nnanoGPT config:")
    print(f"  num_layers = {num_layers}")
    print(f"  embed_dim = {embed_dim}")
    print(f"  num_heads = {num_heads}")
    print(f"  head_dim = {head_dim}")
    print(f"  intermediate_size = {intermediate_size}")
    print(f"  bias = {model_args['bias']}")
    print()

    z = torch.zeros
    w = {
        # embeddings
        "wte": sd["transformer.wte.weight"],
        "wpe": sd["transformer.wpe.weight"],
        # LM head: identical to wte when tied, an independent matrix when not (see note 4)
        "lm_head": sd.get("lm_head.weight", sd["transformer.wte.weight"]),
        # attention, full
        "W_Q": z(num_layers, embed_dim, embed_dim),
        "W_K": z(num_layers, embed_dim, embed_dim),
        "W_V": z(num_layers, embed_dim, embed_dim),
        "W_O": z(num_layers, embed_dim, embed_dim),
        # attention, per-head
        "W_Q_heads": z(num_layers, num_heads, embed_dim, head_dim),
        "W_K_heads": z(num_layers, num_heads, embed_dim, head_dim),
        "W_V_heads": z(num_layers, num_heads, embed_dim, head_dim),
        "W_O_heads": z(num_layers, num_heads, head_dim, embed_dim),
        # mlp
        "W_fc": z(num_layers, embed_dim, intermediate_size),
        "W_proj": z(num_layers, intermediate_size, embed_dim),
        # layernorm weights
        "ln1_weight": z(num_layers, embed_dim),
        "ln2_weight": z(num_layers, embed_dim),
        "ln_f_weight": sd["transformer.ln_f.weight"],
        # biases: bias=False, so these stay zero (see note 2 in the header)
        "b_Q": z(num_layers, embed_dim),
        "b_K": z(num_layers, embed_dim),
        "b_V": z(num_layers, embed_dim),
        "b_O": z(num_layers, embed_dim),
        "b_Q_heads": z(num_layers, num_heads, head_dim),
        "b_K_heads": z(num_layers, num_heads, head_dim),
        "b_V_heads": z(num_layers, num_heads, head_dim),
        "b_fc": z(num_layers, intermediate_size),
        "b_proj": z(num_layers, embed_dim),
        "ln1_bias": z(num_layers, embed_dim),
        "ln2_bias": z(num_layers, embed_dim),
        "ln_f_bias": z(embed_dim),
    }

    print(f"  wte (word token embeddings): {list(w['wte'].shape)}")
    print(f"  wpe (positional embeddings): {list(w['wpe'].shape)}")
    print("\nExtracting per-layer weights...")
    for i in range(num_layers):
        p = f"transformer.h.{i}."
        # .T turns nn.Linear's [out, in] into the [in, out] convention of this format
        c_attn = sd[p + "attn.c_attn.weight"].T          # [embed, 3*embed]
        W_Q = c_attn[:, :embed_dim]
        W_K = c_attn[:, embed_dim:2 * embed_dim]
        W_V = c_attn[:, 2 * embed_dim:]
        W_O = sd[p + "attn.c_proj.weight"].T             # [embed, embed]

        w["W_Q"][i], w["W_K"][i], w["W_V"][i], w["W_O"][i] = W_Q, W_K, W_V, W_O
        w["W_fc"][i] = sd[p + "mlp.c_fc.weight"].T       # [embed, intermediate]
        w["W_proj"][i] = sd[p + "mlp.c_proj.weight"].T   # [intermediate, embed]
        w["ln1_weight"][i] = sd[p + "ln_1.weight"]
        w["ln2_weight"][i] = sd[p + "ln_2.weight"]

        for h in range(num_heads):
            s, e = h * head_dim, (h + 1) * head_dim
            w["W_Q_heads"][i, h] = W_Q[:, s:e]
            w["W_K_heads"][i, h] = W_K[:, s:e]
            w["W_V_heads"][i, h] = W_V[:, s:e]
            w["W_O_heads"][i, h] = W_O[s:e, :]

        print(f"  Layer {i:2d}: W_Q={list(W_Q.shape)}, W_Q^h={list(w['W_Q_heads'][i, 0].shape)}, "
              f"W_fc={list(w['W_fc'][i].shape)}")

    dims = dict(num_layers=num_layers, num_heads=num_heads, head_dim=head_dim,
                embed_dim=embed_dim, intermediate_size=intermediate_size,
                vocab_size=w["wte"].shape[0], max_position_embeddings=w["wpe"].shape[0],
                tie_weights=model_args.get("tie_weights", True),  # older ckpts predate the flags
                alpha=model_args.get("alpha", 1.0))
    return w, dims


# which tensors go in per_head/ vs fused/, and under which filename
PER_HEAD_FILES = ["W_Q_heads", "W_K_heads", "W_V_heads", "W_O_heads",
                  "b_Q_heads", "b_K_heads", "b_V_heads"]
FUSED_GROUPS = [
    ("Embedding weights", ["wte", "wpe"]),
    ("Full (fused) attention matrices + biases",
     ["W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V", "b_O"]),
    ("Pre-attention LayerNorm (ln_1)", ["ln1_weight", "ln1_bias"]),
    ("MLP matrices", ["W_fc", "b_fc", "W_proj", "b_proj"]),
    ("Pre-MLP LayerNorm (ln_2)", ["ln2_weight", "ln2_bias"]),
    ("Final LayerNorm (ln_f)", ["ln_f_weight", "ln_f_bias"]),
    ("LM head", ["lm_head"]),
]


def save_weights(w, dims, ckpt, ckpt_path, save_prefix):
    per_head_dir = os.path.join(save_prefix, "per_head")
    fused_dir = os.path.join(save_prefix, "fused")
    os.makedirs(per_head_dir, exist_ok=True)
    os.makedirs(fused_dir, exist_ok=True)

    print(f"\nSaving weights to {save_prefix}/ ...")

    print(f"\n[per_head/] Per-head attention matrices and biases:")
    # b_O is shared across heads, so it lands in both directories unsplit
    for name in PER_HEAD_FILES + ["b_O"]:
        torch.save(w[name], os.path.join(per_head_dir, f"{name}.pt"))
        print(f"  {name}.pt - shape {list(w[name].shape)}")

    for title, names in FUSED_GROUPS:
        print(f"\n[fused/] {title}:")
        for name in names:
            torch.save(w[name], os.path.join(fused_dir, f"{name}.pt"))
            print(f"  {name}.pt - shape {list(w[name].shape)}")

    metadata = build_metadata(dims, ckpt, ckpt_path)
    metadata_path = os.path.join(save_prefix, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  {metadata_path}")

    unique = sum(w[n].numel() for _, names in FUSED_GROUPS for n in names)
    if dims["tie_weights"]:
        unique -= w["lm_head"].numel()  # same tensor as wte, not a second parameter
    print(f"\n  Total unique parameters saved: {unique:,}")
    print("\nDone!")


def build_metadata(dims, ckpt, ckpt_path):
    nl, nh, hd = dims["num_layers"], dims["num_heads"], dims["head_dim"]
    ed, isz = dims["embed_dim"], dims["intermediate_size"]
    vs, mpe = dims["vocab_size"], dims["max_position_embeddings"]
    tied = dims["tie_weights"]

    return {
        "model": os.path.basename(ckpt_path),
        "model_type": f"nanoGPT ({nl}L, {ed}d, {nh}H)",
        # provenance: which run and which step produced these weights
        "source": {
            "checkpoint": os.path.abspath(ckpt_path),
            "iter_num": ckpt["iter_num"],
            "best_val_loss": float(ckpt["best_val_loss"]),
            "train_config": ckpt["config"],
        },
        "num_layers": nl,
        "num_heads": nh,
        "head_dim": hd,
        "embed_dim": ed,
        "hidden_size": ed,
        "intermediate_size": isz,
        "vocab_size": vs,
        "max_position_embeddings": mpe,
        "hidden_act": HIDDEN_ACT,
        "layer_norm_eps": LN_EPS,
        "bias": False,
        "tie_weights": tied,
        "alpha": dims["alpha"],
        "architecture_notes": {
            "norm_type": "pre-norm (LayerNorm before attention/MLP, not after)",
            "layer_type": "nn.Linear (weight is [out, in]) - TRANSPOSED on export to the "
                          "[in, out] convention used by the HuggingFace GPT-2 export",
            "bias": "trained with bias=False: no biases anywhere, not in attention, MLP, or "
                    "any LayerNorm. All b_*.pt / *_bias.pt files are all-zero placeholders so "
                    "downstream loaders keep working (adding zero is a no-op)",
            "vocab_padding": (
                f"vocab_size is {vs}, i.e. the real GPT-2 vocab of {GPT2_VOCAB} padded up by "
                f"{vs - GPT2_VOCAB} for speed. Rows {GPT2_VOCAB}: do not correspond to real tokens"
                if vs > GPT2_VOCAB else
                f"vocab_size is {vs}, not padded: every row corresponds to a real token"
            ),
            "lm_head": (
                "tie_weights=True: lm_head.weight is transformer.wte.weight (same storage), so "
                "lm_head.pt and wte.pt hold the same tensor"
                if tied else
                "tie_weights=False: lm_head.pt is an independent matrix, NOT wte.pt. Use it, "
                "not wte, as the output projection"
            ),
            "alpha": (
                f"logits = alpha * ln_f(x) @ lm_head.T with alpha = {dims['alpha']}. The factor "
                "is applied at forward time and is NOT baked into lm_head.pt"
            ),
        },
        "directory_structure": {
            "per_head/": "Per-head attention matrices and biases",
            "fused/": "Full attention, MLP, LayerNorm, and embeddings weights",
        },
        "per_head_files": {
            "W_Q_heads.pt": f"[{nl}, {nh}, {ed}, {hd}]",
            "W_K_heads.pt": f"[{nl}, {nh}, {ed}, {hd}]",
            "W_V_heads.pt": f"[{nl}, {nh}, {ed}, {hd}]",
            "W_O_heads.pt": f"[{nl}, {nh}, {hd}, {ed}]",
            "b_Q_heads.pt": f"[{nl}, {nh}, {hd}] (all zero)",
            "b_K_heads.pt": f"[{nl}, {nh}, {hd}] (all zero)",
            "b_V_heads.pt": f"[{nl}, {nh}, {hd}] (all zero)",
            "b_O.pt": f"[{nl}, {ed}] (all zero; output bias is shared, not per-head)",
        },
        "fused_files": {
            "embeddings": {
                "wte.pt": f"[{vs}, {ed}] - word token embeddings",
                "wpe.pt": f"[{mpe}, {ed}] - positional embeddings",
            },
            "lm_head": {
                "lm_head.pt": f"[{vs}, {ed}] - output projection, logits = ln_f(x) @ lm_head.T"
                              + (" (same tensor as wte.pt)" if tied else " (independent of wte.pt)"),
            },
            "attention": {
                "W_Q.pt": f"[{nl}, {ed}, {ed}]",
                "W_K.pt": f"[{nl}, {ed}, {ed}]",
                "W_V.pt": f"[{nl}, {ed}, {ed}]",
                "W_O.pt": f"[{nl}, {ed}, {ed}]",
                "b_Q.pt": f"[{nl}, {ed}] (all zero)",
                "b_K.pt": f"[{nl}, {ed}] (all zero)",
                "b_V.pt": f"[{nl}, {ed}] (all zero)",
                "b_O.pt": f"[{nl}, {ed}] (all zero)",
                "ln1_weight.pt": f"[{nl}, {ed}] - pre-attention LayerNorm",
                "ln1_bias.pt": f"[{nl}, {ed}] (all zero)",
            },
            "mlp": {
                "W_fc.pt": f"[{nl}, {ed}, {isz}] - up projection (activation: {HIDDEN_ACT})",
                "b_fc.pt": f"[{nl}, {isz}] (all zero)",
                "W_proj.pt": f"[{nl}, {isz}, {ed}] - down projection",
                "b_proj.pt": f"[{nl}, {ed}] (all zero)",
                "ln2_weight.pt": f"[{nl}, {ed}] - pre-MLP LayerNorm",
                "ln2_bias.pt": f"[{nl}, {ed}] (all zero)",
            },
            "final": {
                "ln_f_weight.pt": f"[{ed}] - final LayerNorm",
                "ln_f_bias.pt": f"[{ed}] (all zero)",
            },
        },
        "notes": {
            "weight_convention": "saved as [in_features, out_features], matching the HuggingFace "
                                 "GPT-2 export. nanoGPT stores [out, in], so every matrix here "
                                 "was transposed on the way out",
            "W_Q_heads": f"Each head: [{ed}, {hd}], input -> query projection",
            "W_K_heads": f"Each head: [{ed}, {hd}], input -> key projection",
            "W_V_heads": f"Each head: [{ed}, {hd}], input -> value projection",
            "W_O_heads": f"Each head: [{hd}, {ed}], attention output -> hidden",
            "W_fc": f"Up projection: [{ed}, {isz}]",
            "W_proj": f"Down projection: [{isz}, {ed}]",
            "qkv_layout": "c_attn output is [Q | K | V] contiguous blocks, each split into "
                          "contiguous per-head chunks - identical to HuggingFace GPT-2",
            "ln1": "Pre-attention LayerNorm",
            "ln2": "Pre-MLP LayerNorm",
            "ln_f": "Final LayerNorm after last transformer layer",
            "layernorm": f"weight (gamma) only, bias (beta) is all zero. eps = {LN_EPS}",
            "activation": f"MLP activation is nn.GELU(approximate='tanh'), equivalent to "
                          f"HuggingFace's '{HIDDEN_ACT}'",
            "hidden_size": "Alias of embed_dim, kept so the same metadata key works across "
                           "model families",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Extract ALL weights from a nanoGPT checkpoint")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="path to a nanoGPT checkpoint .pt file")
    parser.add_argument("--savefile", type=str, default=None,
                        help="output directory (default: <ckpt_dir>/extracted/<ckpt_stem>/)")
    args = parser.parse_args()

    if args.savefile is None:
        stem = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.savefile = os.path.join(os.path.dirname(args.ckpt), "extracted", stem)

    print(f"Loading checkpoint: {args.ckpt}")
    sd, model_args, ckpt = load_state_dict(args.ckpt)
    print(f"Loaded iter {ckpt['iter_num']:,}, best val loss {float(ckpt['best_val_loss']):.4f}\n")

    with torch.no_grad():
        w, dims = extract(sd, model_args)
        save_weights(w, dims, ckpt, args.ckpt, args.savefile)


if __name__ == "__main__":
    main()
