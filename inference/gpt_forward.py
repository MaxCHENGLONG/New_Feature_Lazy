"""
GPT-2 (nanoGPT) forward pass, written out from the trained weights.

Independent of model.py: nothing is imported from it. Where model.py uses nn.Linear,
F.layer_norm and F.scaled_dot_product_attention, this file spells everything out as
explicit matrix multiplies and softmaxes, so the code lines up with the equations.

Weight convention: every matrix here is [in_features, out_features], so the forward pass
is always `x @ W` with no transposes. That is the convention get_nanogpt_weights.py
exports (nanoGPT's nn.Linear stores [out, in] and the exporter transposes on the way out).
Loading straight from a checkpoint does the same transpose, see _from_ckpt.

One Block (pre-norm residual):
    x = x + Attn(LN1(x))
    x = x + MLP(LN2(x))
and finally logits = alpha * LN_f(x) @ lm_head.T   (lm_head is wte itself when the run tied
them, an independent matrix when it did not; alpha is the lazy-regime output multiplier from
training, 1 for ordinary runs, read from model_args / metadata.json)
"""

import json
import math
import os

import torch

LN_EPS = 1e-5  # hardcoded in model.py's LayerNorm.forward

# everything in fused/, which is also everything the forward pass below needs
FUSED_FILES = [
    "wte", "wpe",
    "W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V", "b_O",
    "ln1_weight", "ln1_bias", "ln2_weight", "ln2_bias",
    "W_fc", "b_fc", "W_proj", "b_proj",
    "ln_f_weight", "ln_f_bias",
]


# ============================== loading ==============================

def load_weights(path, device="cpu"):
    """path is either a get_nanogpt_weights.py export directory or a raw ckpt .pt file.

    Returns (w, cfg): w maps name -> tensor, cfg holds n_layer / n_head / n_embd /
    vocab_size / block_size / alpha.
    """
    w, cfg = _from_extracted(path) if os.path.isdir(path) else _from_ckpt(path)
    w = {k: v.to(device=device, dtype=torch.float32).contiguous() for k, v in w.items()}
    return w, cfg


def _from_extracted(dirpath):
    """Export directory: names and layouts already match what the forward pass wants."""
    with open(os.path.join(dirpath, "metadata.json")) as f:
        meta = json.load(f)
    w = {n: torch.load(os.path.join(dirpath, "fused", f"{n}.pt"), map_location="cpu")
         for n in FUSED_FILES}
    # exports made before lm_head.pt existed were all tied runs, so wte is the head there
    head = os.path.join(dirpath, "fused", "lm_head.pt")
    w["lm_head"] = torch.load(head, map_location="cpu") if os.path.exists(head) else w["wte"]
    # exports made before the top-level alpha key existed predate the logit multiplier
    cfg = dict(n_layer=meta["num_layers"], n_head=meta["num_heads"], n_embd=meta["embed_dim"],
               vocab_size=meta["vocab_size"], block_size=meta["max_position_embeddings"],
               alpha=meta.get("alpha", 1.0))
    return w, cfg


def _from_ckpt(ckpt_path):
    """Raw checkpoint: redo the two things get_nanogpt_weights.py does — strip
    torch.compile's _orig_mod. prefix, and transpose nn.Linear's [out, in] to [in, out]."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("_orig_mod."): v.float() for k, v in ckpt["model"].items()}
    args = ckpt["model_args"]
    L, D = args["n_layer"], args["n_embd"]
    I = sd["transformer.h.0.mlp.c_fc.weight"].shape[0]  # nn.Linear is [out, in]

    def pick(key, shape):
        # with bias=False these keys do not exist; zeros are a no-op when added
        return sd[key] if key in sd else torch.zeros(shape)

    names = ("W_Q W_K W_V b_Q b_K b_V W_O b_O W_fc b_fc W_proj b_proj "
             "ln1_weight ln1_bias ln2_weight ln2_bias").split()
    layers = {n: [] for n in names}

    for i in range(L):
        p = f"transformer.h.{i}."
        # c_attn computes Q|K|V in one shot; transpose, then slice the three column blocks
        c_attn_w = sd[p + "attn.c_attn.weight"].T              # [D, 3D]
        c_attn_b = pick(p + "attn.c_attn.bias", (3 * D,))      # [3D]
        for j, n in enumerate("QKV"):
            layers[f"W_{n}"].append(c_attn_w[:, j * D:(j + 1) * D])
            layers[f"b_{n}"].append(c_attn_b[j * D:(j + 1) * D])
        layers["W_O"].append(sd[p + "attn.c_proj.weight"].T)   # [D, D]
        layers["b_O"].append(pick(p + "attn.c_proj.bias", (D,)))
        layers["W_fc"].append(sd[p + "mlp.c_fc.weight"].T)     # [D, I]
        layers["b_fc"].append(pick(p + "mlp.c_fc.bias", (I,)))
        layers["W_proj"].append(sd[p + "mlp.c_proj.weight"].T)  # [I, D]
        layers["b_proj"].append(pick(p + "mlp.c_proj.bias", (D,)))
        layers["ln1_weight"].append(sd[p + "ln_1.weight"])
        layers["ln1_bias"].append(pick(p + "ln_1.bias", (D,)))
        layers["ln2_weight"].append(sd[p + "ln_2.weight"])
        layers["ln2_bias"].append(pick(p + "ln_2.bias", (D,)))

    w = {n: torch.stack(v) for n, v in layers.items()}         # each becomes [L, ...]
    w["wte"] = sd["transformer.wte.weight"]
    w["lm_head"] = sd.get("lm_head.weight", w["wte"])  # untied runs carry their own head
    w["wpe"] = sd["transformer.wpe.weight"]
    w["ln_f_weight"] = sd["transformer.ln_f.weight"]
    w["ln_f_bias"] = pick("transformer.ln_f.bias", (D,))

    cfg = dict(n_layer=L, n_head=args["n_head"], n_embd=D,
               vocab_size=w["wte"].shape[0], block_size=w["wpe"].shape[0],
               alpha=args.get("alpha", 1.0))  # older ckpts predate the flag
    return w, cfg


# ============================== primitives ==============================

def layer_norm(x, gamma, beta):
    """Normalize over the last (feature) dim, then scale and shift per channel."""
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)  # biased variance: divide by D, not D-1
    return (x - mu) / torch.sqrt(var + LN_EPS) * gamma + beta


def gelu(x):
    """nn.GELU(approximate='tanh'), i.e. HuggingFace's 'gelu_new'."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


# ============================== one Block ==============================

def causal_self_attention(x, w, l, n_head):
    """x: [B, T, D] -> [B, T, D]. Causal self-attention, all heads at once."""
    B, T, D = x.shape
    hd = D // n_head  # head_dim

    # project, then reshape to [B, n_head, T, hd] so the head dim batches the matmuls
    q = (x @ w["W_Q"][l] + w["b_Q"][l]).view(B, T, n_head, hd).transpose(1, 2)
    k = (x @ w["W_K"][l] + w["b_K"][l]).view(B, T, n_head, hd).transpose(1, 2)
    v = (x @ w["W_V"][l] + w["b_V"][l]).view(B, T, n_head, hd).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)          # [B, n_head, T, T]
    # causal mask: position i may only attend to j <= i, so the strict upper triangle
    # goes to -inf and softmax turns it into exactly 0
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
    att = torch.softmax(att.masked_fill(mask, float("-inf")), dim=-1)

    y = att @ v                                              # [B, n_head, T, hd]
    y = y.transpose(1, 2).reshape(B, T, D)                   # heads back side by side
    return y @ w["W_O"][l] + w["b_O"][l]                     # output projection


def mlp(x, w, l):
    """x: [B, T, D] -> [B, T, D]. Up to 4D, GELU, back down to D."""
    h = gelu(x @ w["W_fc"][l] + w["b_fc"][l])                # [B, T, 4D]
    return h @ w["W_proj"][l] + w["b_proj"][l]               # [B, T, D]


# ============================== full forward ==============================

@torch.no_grad()
def forward(idx, w, cfg):
    """idx: [B, T] token ids -> logits: [B, T, vocab_size]"""
    B, T = idx.shape
    assert T <= cfg["block_size"], f"sequence length {T} exceeds block_size {cfg['block_size']}"

    # token + position embeddings; wpe[:T] is [T, D] and broadcasts over the batch
    x = w["wte"][idx] + w["wpe"][:T]

    for l in range(cfg["n_layer"]):
        x = x + causal_self_attention(layer_norm(x, w["ln1_weight"][l], w["ln1_bias"][l]),
                                      w, l, cfg["n_head"])
        x = x + mlp(layer_norm(x, w["ln2_weight"][l], w["ln2_bias"][l]), w, l)

    x = layer_norm(x, w["ln_f_weight"], w["ln_f_bias"])
    logits = x @ w["lm_head"].T  # output projection (== wte.T when the run tied them)
    # lazy-regime output multiplier, exactly as model.py applies it in training. It is not
    # baked into lm_head, so it has to be applied here too or the logits come out 1/alpha
    alpha = cfg.get("alpha", 1.0)
    return logits * alpha if alpha != 1.0 else logits


@torch.no_grad()
def generate(idx, w, cfg, max_new_tokens, temperature=1.0, top_k=None, valid_vocab=None):
    """Autoregressive sampling. Recomputes the whole sequence each step (no KV cache),
    which keeps the code identical to the equations above.

    valid_vocab: sample only from the first valid_vocab tokens. vocab_size is padded to
    50304 and the trailing rows are not real tokens, so sampling one breaks decoding.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg["block_size"]:]                # crop to the context window
        logits = forward(idx_cond, w, cfg)[:, -1, :] / temperature  # last position only
        if valid_vocab is not None:
            logits[:, valid_vocab:] = -float("inf")
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
    return idx
