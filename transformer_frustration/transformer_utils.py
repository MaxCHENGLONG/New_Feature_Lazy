import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

def load_model_information(load_path):
    '''Load model information to dict from a train.py checkpoint (ckpt_*.pt or ckpt.pt).'''
    # weights_only=False because our checkpoints carry the model_args/config dicts too
    data = torch.load(load_path, map_location='cpu', weights_only=False)

    L   = data['model_args']['n_layer']     # Number of layers
    H   = data['model_args']['n_head']      # Number of heads
    d   = data['model_args']['n_embd']      # Token dimension
    dt  = data['model_args']['vocab_size']  # Embedding token dimension
    dp  = data['model_args']['block_size']  # Embedding position dimension

    # use_bias = data['model_args']['bias']

    print('Shaping the matrix')
    # torch.compile prefixes every key with _orig_mod.; runs with compile=False have no prefix
    model = {k.removeprefix('_orig_mod.'): v.float().cpu().numpy() for k, v in data['model'].items()}

    # Embedding layer
    Et = model['transformer.wte.weight'].T
    Ep = model['transformer.wpe.weight'].T

    WQ_list = []; WK_list = []; WV_list = []
    WO_list = []; W1_list = []; W2_list = []

    for c in range(L):
        # Attention. nn.Linear stores [out, in], so c_attn is [3d, d] with Q|K|V stacked on rows
        QKV = model[f'transformer.h.{c}.attn.c_attn.weight']     # key, query, value
        WQ, WK, WV = np.split(QKV, 3, axis=0)
        WQ_list.append(WQ)
        WK_list.append(WK)
        WV_list.append(WV)

        WO = model[f'transformer.h.{c}.attn.c_proj.weight']   # output projection
        WO_list.append(WO)

        # MLP
        W1 = model[f'transformer.h.{c}.mlp.c_fc.weight']
        W2 = model[f'transformer.h.{c}.mlp.c_proj.weight']
        W1_list.append(W1)
        W2_list.append(W2)

    model_info = {
        'L': L,
        'H': H,
        'd': d,
        'dt': dt,
        'dp': dp,
        'Et': Et,
        'Ep': Ep,
        'WQ_list': WQ_list,
        'WK_list': WK_list,
        'WV_list': WV_list,
        'WO_list': WO_list,
        'W1_list': W1_list,
        'W2_list': W2_list,
        'std': float(data['model_args']['init_std']),
        'loss': float(data['best_val_loss']),
        'iter': int(data['iter_num']),
        'config': data.get('config', {}),   # the train.py settings of the run (alpha, seed, ...)
    }

    return model_info

def network_construction(model_info, T=1, is_embed=False):
    print(f'Building the Transformer network')
    L = model_info['L'] # Number of layers
    H = model_info['H']
    d = model_info['d'] # Token dimension
    df = 4 * d
    # dh = d//H

    if is_embed:
        dp = model_info['dp']
        dt = model_info['dt']
        de = T * (dp + dt)      # dimension of the embedding layer
        Et = model_info['Et']
        Ep = model_info['Ep']
    else:
        de = 0

    dm = 3 * T * d + T * df     # dimension of one layer
    dim = de + d * T + dm * L   # dimension of the transformer

    # Build MHA FO * FV
    WV_list = model_info['WV_list']
    WO_list = model_info['WO_list']
    W1_list = model_info['W1_list']
    W2_list = model_info['W2_list']
    all_rows = []
    all_cols = []
    all_vals = []

    def add_identity(r0, c0, size):
        '''Add identity block of given size.'''
        idx = np.arange(size)
        rows = r0 + idx
        cols = c0 + idx
        vals = np.ones(size)
        return rows, cols, vals

    def add_kron_eye(r0, c0, A, i, dr, dc):
        '''Add kron(I_n, A).'''
        # ar, ac = np.nonzero(A)
        ar, ac = np.indices(A.shape)
        ar, ac = ar.ravel(), ac.ravel()
        avals = A[ar, ac]

        rows = r0 + i*dr + ar
        cols = c0 + i*dc + ac
        vals = avals
        return rows, cols, vals

    def add_causal_attention(r0, c0, n, d):
        '''Causal attention (GPT-style).'''
        idx = np.arange(d)

        rows = []; cols = []; vals = []
        for i in range(n):
            scale = 1.0 / (i + 1)
            for j in range(i + 1):
                rows.extend(r0 + i*d + idx)
                cols.extend(c0 + j*d + idx)
                vals.extend(np.full(d, scale))
        return rows, cols, vals

    # Embedding block
    if is_embed:
        print(f'Embedding layer')
        # ---- position embedding ----
        r0 = de; c0 = 0
        for i in range(T):
            print(Ep.shape)
            rows, cols, vals = add_kron_eye(r0, c0, Ep, i, d, dp)
            all_rows.append(np.array(rows, dtype=np.int32))
            all_cols.append(np.array(cols, dtype=np.int32))
            all_vals.append(np.array(vals, dtype=np.float32))

        # ---- token embedding ----
        r0 = de; c0 = T * dp
        for i in range(T):
            rows, cols, vals = add_kron_eye(r0, c0, Et, i, d, dt)
            all_rows.append(np.array(rows, dtype=np.int32))
            all_cols.append(np.array(cols, dtype=np.int32))
            all_vals.append(np.array(vals, dtype=np.float32))
        
        # del Et, Ep

    # Transformer block
    for c in range(L):
        print(f'Layer {c+1} of {L}')
        row_offset = de + c * dm + d * T
        col_offset = de + c * dm

        # ---- causal attention ----
        r0 = row_offset
        c0 = col_offset
        rows, cols, vals = add_causal_attention(r0, c0, T, d)
        all_rows.append(np.array(rows, dtype=np.int32))
        all_cols.append(np.array(cols, dtype=np.int32))
        all_vals.append(np.array(vals, dtype=np.float32))

        # ---- residual: identity (n*d × n*d) ----
        r0 = row_offset + T * d
        c0 = col_offset
        rows, cols, vals = add_identity(r0, c0, T*d)
        all_rows.append(np.array(rows, dtype=np.int32))
        all_cols.append(np.array(cols, dtype=np.int32))
        all_vals.append(np.array(vals, dtype=np.float32))

        # ---- Fo * Fv block ---- 
        r0 = row_offset + T * d
        c0 = col_offset + T * d
        for i in range(T):
            rows, cols, vals = add_kron_eye(r0, c0, WO_list[c]@WV_list[c], i, d, d)
            all_rows.append(np.array(rows, dtype=np.int32))
            all_cols.append(np.array(cols, dtype=np.int32))
            all_vals.append(np.array(vals, dtype=np.float32))

        # ---- MLP first layer ----
        r0 = row_offset + 2 * T * d
        c0 = col_offset + 2 * T * d
        for i in range(T):
            rows, cols, vals = add_kron_eye(r0, c0, W1_list[c], i, df, d)
            all_rows.append(np.array(rows, dtype=np.int32))
            all_cols.append(np.array(cols, dtype=np.int32))
            all_vals.append(np.array(vals, dtype=np.float32))

        # ---- MLP second layer ----
        r0 = row_offset + 2 * T * d + T * df
        c0 = col_offset + 3 * T * d
        for i in range(T):
            rows, cols, vals = add_kron_eye(r0, c0, W2_list[c], i, d, df)
            all_rows.append(np.array(rows, dtype=np.int32))
            all_cols.append(np.array(cols, dtype=np.int32))
            all_vals.append(np.array(vals, dtype=np.float32))

        # ---- residual MLP ----
        r0 = row_offset + 2 * T * d + T * df
        c0 = col_offset + 2 * T * d
        rows, cols, vals = add_identity(r0, c0, T*d)
        all_rows.append(np.array(rows, dtype=np.int32))
        all_cols.append(np.array(cols, dtype=np.int32))
        all_vals.append(np.array(vals, dtype=np.float32))
    # del WV_list, WO_list, W1_list, W2_list

    # Convert to arrays
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.concatenate(all_vals)
    return rows, cols, vals

if __name__ == "__main__":
    name = 'GPT2'
    idx = 1
    fld = f'pretrained/{name}/config{idx}'
    epoch = 152000
    # epoch = 0
    is_plot = False
    load_path = f'{fld}/models/ckpt_{epoch:07d}.pt'
    # data = torch.load(load_path, map_location='cpu', weights_only=True)
    # print(data.keys())
    # print(data['model_args'])

    model_info = load_model_information(load_path)
    rows, cols, vals = network_construction(model_info)
    dim = max(rows.max(), cols.max()) + 1
    print(vals.shape)

    # Plot adjacency matrix
    if is_plot:
        import matplotlib.pyplot as plt
        import scipy.sparse as sp

        A = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim))
        plt.spy(A, markersize=1)
        plt.show()