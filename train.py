"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

On a GH200 node (1 Hopper GPU, 96/144GB HBM3e) a single process is usually enough. The
141GB of HBM wants a much larger micro-batch than the 8xA100 defaults below, e.g.:
$ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py --batch_size=48 --gradient_accumulation_steps=10 --compile_mode=max-autotune

To sweep weight initializations, see the init_* knobs below, e.g.:
$ python train.py --init_dist=trunc_normal --init_std=0.015 --seed=0
$ python train.py --init_dist=xavier_uniform --init_gain=1.0 --seed=1
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier, broadcast

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 1000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
# permanent snapshots, kept alongside the rolling ckpt.pt. iter 0 (the untrained init) is
# always included, then log-spaced points over the first snapshot_every iters where the
# loss moves fastest, then a fixed interval to the end. Weights only, no optimizer state.
snapshot_every = 200 # fixed snapshot interval after the log-spaced phase. 0 disables snapshots
snapshot_log_points = 12 # log-spaced snapshots between iter 1 and snapshot_every
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 8 # used to simulate larger batch sizes. must be divisible by the DDP world size
batch_size = 64 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# weight init (only takes effect when init_from='scratch')
init_dist = 'normal' # 'normal' | 'trunc_normal' | 'uniform' | 'xavier_normal' | 'xavier_uniform'
init_std = 0.02 # std of the normal/trunc_normal/uniform init. ignored by the xavier ones
init_gain = 1.0 # gain of the xavier inits. ignored by the others
init_scale_residual = True # rescale residual projections by 1/sqrt(2*n_layer), per GPT-2
init_proj_scale = 1.0 # extra multiplier on every c_proj.weight. 1e-3 or 0.0: saddle-to-saddle init
seed = 1337 # rng seed. change it to draw a different init (and a different data order)
# lazy regime, per Chizat & Bach (2019): logits = alpha * f(theta) with every weight drawn
# identically (no weight is scaled, so no softmax saturates). learning_rate and min_lr are
# divided by alpha^2, which is what keeps the alpha-amplified function from diverging.
# alpha=1 is the usual model; alpha >> 1 is the lazy regime
alpha = 1.0
alpha_center = False # subtract the logits of a frozen copy of the init, so the model starts at 0
tie_weights = True # tie lm_head to wte (GPT-2 default)
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 19200 # total number of training iterations (~10.1B tokens at 524,288 tokens/iter)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 19200 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
compile_mode = 'default' # 'default' | 'max-autotune'. the latter compiles slower but tunes kernels for the GPU
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
if alpha != 1.0:
    learning_rate /= alpha ** 2
    min_lr /= alpha ** 2
    print(f"alpha={alpha}: learning_rate -> {learning_rate:.3e}, min_lr -> {min_lr:.3e}")

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"total tokens over {max_iters:,} iters: {tokens_per_iter * max_iters / 1e9:.1f}B")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.set_float32_matmul_precision('high') # tf32 for the float32 matmuls torch.compile emits
# on Hopper (H100/H200/GH200) the cuDNN attention backend beats the default flash backend,
# but PyTorch does not always pick it on its own
if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
    torch.backends.cuda.enable_cudnn_sdp(True)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  init_dist=init_dist, init_std=init_std, init_gain=init_gain,
                  init_scale_residual=init_scale_residual, init_proj_scale=init_proj_scale,
                  alpha=alpha, tie_weights=tie_weights) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    # weights_only=False because our checkpoints carry the model_args/config dicts too
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # alpha/tie_weights cannot be silently taken from the checkpoint: learning_rate was already
    # divided by the command-line alpha^2 above, and loading an untied checkpoint into a tied
    # model does not error -- lm_head.weight just overwrites wte. Demand the same flags instead.
    for k, default in [('alpha', 1.0), ('tie_weights', True)]:
        saved = checkpoint_model_args.get(k, default) # older checkpoints predate the flags
        assert model_args[k] == saved, \
            f"checkpoint was trained with {k}={saved}, got {k}={model_args[k]}: pass --{k}={saved}"
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# every rank drew its own init (seed + ddp_rank). DDP broadcasts rank 0's weights when it wraps
# the model further down, but the copies below are taken before that, so sync here first
if ddp:
    for p in model.parameters():
        broadcast(p.data, src=0)

# frozen copy of the init for alpha-centering, and a copy of theta_0 to measure how far the
# weights travel. From scratch that is the model itself; on resume it is the iter-0 snapshot
# (ckpt_0000000.pt, always written when snapshots are on). Pretrained inits have no theta_0.
model0 = None
params0 = None
if init_from in ('scratch', 'resume'):
    import copy
    model0 = copy.deepcopy(model).eval().requires_grad_(False)
    if init_from == 'resume':
        init_sd = torch.load(os.path.join(out_dir, 'ckpt_0000000.pt'), map_location=device, weights_only=False)['model']
        init_sd = {k.removeprefix('_orig_mod.'): v for k, v in init_sd.items()}
        model0.load_state_dict(init_sd)
    # (name, live parameter, frozen copy) -- holds the plain module's Parameters directly, so it
    # keeps working after torch.compile/DDP rename them
    theta0 = dict(model0.named_parameters())
    params0 = [(n, p, theta0[n]) for n, p in model.named_parameters()]
    if not alpha_center:
        model0 = None # only needed for its parameters, which params0 now holds
elif alpha_center:
    raise ValueError("alpha_center needs the untrained init, not a pretrained model")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print(f"compiling the model with mode={compile_mode}... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model, mode=compile_mode) # requires PyTorch 2.0
    if model0 is not None:
        # same kernels as the live model, so the centering subtraction cancels to the same
        # rounding (matters at large alpha), and the extra forward is not left running eager
        model0 = torch.compile(model0, mode=compile_mode)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

def forward_loss(X, Y):
    # alpha-centering: subtract the logits of the frozen init so the trained function starts at 0
    ref = None
    if model0 is not None:
        with torch.no_grad():
            ref, _ = model0(X, Y)
    return model(X, Y, ref_logits=ref)

@torch.no_grad()
def weight_movement():
    # relative distance from the init, ||theta_t - theta_0|| / ||theta_0||, over all weights and
    # the per-tensor extremes. lazy training keeps this small; feature learning is O(1)
    # 'hidden' excludes lm_head: it is the hidden weights whose movement tells the regime apart
    num = den = num_h = den_h = 0.0
    per = {}
    for n, p, p0 in params0:
        d = (p.detach().float() - p0.float()).norm().item() ** 2
        w = p0.float().norm().item() ** 2
        num += d
        den += w
        if not n.startswith('lm_head'):
            num_h += d
            den_h += w
        per[n] = math.sqrt(d / w) if w > 0 else 0.0
    lo, hi = min(per, key=per.get), max(per, key=per.get)
    return math.sqrt(num / den), math.sqrt(num_h / den_h), (lo, per[lo]), (hi, per[hi])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = forward_loss(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# iterations at which to keep a permanent snapshot: iter 0, then log-spaced over the first
# snapshot_every iters, then every snapshot_every to the end
snapshot_iters = set()
if snapshot_every > 0:
    snapshot_iters.add(0)
    for i in range(snapshot_log_points):
        e = i / (snapshot_log_points - 1) if snapshot_log_points > 1 else 1.0
        snapshot_iters.add(round(snapshot_every ** e))
    snapshot_iters.update(range(snapshot_every, max_iters + 1, snapshot_every))
    snapshot_iters = {i for i in snapshot_iters if i <= max_iters}
if master_process and snapshot_iters:
    s = sorted(snapshot_iters)
    print(f"{len(s)} snapshots: {s[:14]} ... every {snapshot_every} to {s[-1]:,}")

def save_checkpoint(path, with_optimizer=True):
    # the config dict carries init_dist/init_std/init_gain/seed, so each snapshot records
    # which run produced it without having to encode that in the filename
    ckpt = {
        'model': raw_model.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    if with_optimizer:
        ckpt['optimizer'] = optimizer.state_dict()
    torch.save(ckpt, path)
    print(f"saved {path}")
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints. every rank runs the eval
    # so that any collective inside the forward stays symmetric across ranks under DDP;
    # only the master's numbers are used for logging and checkpointing.
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
    if iter_num % eval_interval == 0 and master_process:
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        rel_move = None
        if params0 is not None:
            rel_move, rel_move_h, (lo_n, lo), (hi_n, hi) = weight_movement()
            print(f"  weight movement |dw|/|w0|: all {rel_move:.3e}, hidden {rel_move_h:.3e}, "
                  f"min {lo:.3e} ({lo_n}), max {hi:.3e} ({hi_n})")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "weight_movement": rel_move,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                save_checkpoint(os.path.join(out_dir, 'ckpt.pt'))

    # permanent snapshot, kept separately from the rolling ckpt.pt so runs with different
    # weight inits stay comparable. iter 0 is included on purpose: it is the untrained init.
    # No optimizer state -- these are for analysis, resume always goes through ckpt.pt.
    if iter_num in snapshot_iters and master_process:
        save_checkpoint(os.path.join(out_dir, f'ckpt_{iter_num:07d}.pt'), with_optimizer=False)

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = forward_loss(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    # let every rank drain its in-flight NCCL work and arrive here before any of them
    # tears the communicators down, otherwise the ranks still in a collective see
    # "Connection closed by remote peer" and spin forever
    torch.cuda.synchronize()
    barrier()
    destroy_process_group()
