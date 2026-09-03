import numpy as np

def change_frustration_edgelist(idx_row, idx_col, weights, out_ptr, in_ptr, in_edges, row_col, I):
    """Change row col sum vector after gauge transformation at index I

    Args:
        idx_row (np.array): index row vector
        idx_col (np.array): index col vector
        weights (np.array): weight vector of the network
        out_ptr (np.array): cumulative col vector
        in_ptr (np.array): cummulative row vector
        in_edges (np.array): mapping
        row_col (np.array): row col vector
        I (int): Flipping index

    Returns:
        row_col (np.array): Updated row col vector
    """
    # outgoing edges
    eo, eo_end = out_ptr[I], out_ptr[I + 1]
    # w_out = weights[eo:eo_end]

    if eo < eo_end:
        w_out = weights[eo:eo_end]
        delta = -2.0 * w_out
        weights[eo:eo_end] *= -1

        row_col[I] += delta.sum()
        np.add.at(row_col, idx_col[eo:eo_end], delta)

    # incoming edges
    ei, ei_end = in_ptr[I], in_ptr[I + 1]

    if ei < ei_end:
        e_idx = in_edges[ei:ei_end]
        w_in = weights[e_idx]
        delta = -2.0 * w_in
        weights[e_idx] *= -1

        row_col[I] += delta.sum()
        np.add.at(row_col, idx_row[e_idx], delta)
    # return row_col, weights

def compute_frustration_edgelist(idx_row, idx_col, weights, max_iter=1_000_000, verbose=False):
    """Compute gauge vector s minimizing frustration for the edgelist format.

    Args:
        idx_row (np.array): Source vector.
        idx_col (np.array): Target vector.
        weights (np.array): Weight vector.
        max_iter (int, optional): Maximal iterations. Defaults to 1_000_000.
        verbose (bool, optional): Print information. Defaults to False.

    Returns:
        frustration (float): Frustration index of the matrix [0, 1]
        s (np.array): Gauge transformation vector (+1 / -1)
    """

    n = max(np.max(idx_row), np.max(idx_col)) + 1

    # Sort by source
    order = np.argsort(idx_row, kind="stable")
    idx_row = idx_row[order]
    idx_col = idx_col[order]
    weights = weights[order]
    del order

    # compute row_col via bincount
    row_col = np.bincount(idx_row, weights, minlength=n)
    row_col += np.bincount(idx_col, weights, minlength=n)

    # outgoing CSR pointer
    count_row = np.bincount(idx_row, minlength=n).astype(np.int32)
    out_ptr = np.empty(n + 1, dtype=np.int32)
    out_ptr[0] = 0
    np.cumsum(count_row, out=out_ptr[1:])
    del count_row

    # incoming CSR pointer
    in_edges = np.argsort(idx_col, kind="stable").astype(np.int32)

    count_col = np.bincount(idx_col, minlength=n)
    in_ptr = np.empty(n + 1, dtype=np.int32)
    in_ptr[0] = 0
    np.cumsum(count_col, out=in_ptr[1:])
    del count_col

    prev_pick = -1
    randpicks = 0

    # Gauge transformation vector
    s = np.ones(n, dtype=np.int8)

    # Apply initial random spins
    # neg = np.flatnonzero(row_col < 0)
    neg = np.where(row_col < 0)[0]
    if neg.size != 0:
        size = min(1000, neg.size)
        rand_idx = neg[np.random.randint(0, neg.size, size=size)]
        for I in rand_idx:
            s[I] *= -1
            change_frustration_edgelist(idx_row, idx_col, weights, out_ptr, in_ptr, in_edges, row_col, I)

    # Main loop
    for iter in range(max_iter):
        Imin = np.argmin(row_col)
        if row_col[Imin] >= 0 or randpicks > 10:
            if verbose:
                print(row_col[Imin], randpicks)
                print(f"Finish after {iter} iterations.")
            break
        
        if prev_pick == Imin:
            # neg = np.flatnonzero(row_col < 0)
            neg = np.where(row_col < 0)[0]
            if len(neg) == 0:
                return Imin
            else:
                I = neg[np.random.randint(neg.size)]
            randpicks += 1
        else:
            I = Imin
            randpicks = 0
            prev_pick = I

        s[I] *= -1
        change_frustration_edgelist(idx_row, idx_col, weights, out_ptr, in_ptr, in_edges, row_col, I)

        if verbose and iter % 1000 == 0:
            print(f"Iteration {iter}\ttotal sum: {np.sum(row_col):.4f}\tnumber negative sum: {np.sum(row_col < 0)}")

    # Frustration
    frustration = 0.5 * (1 - weights.sum() / np.abs(weights).sum())

    return frustration, s

def compute_frustration_edgelist_dir_weight(A_mat, max_iter=1_000_000, verbose=False):
    """Compute gauge vector s minimizing frustration for a directed, weighted, signed network.

    Args:
        A_mat (sp.csr_matrix): An adjacency matrix
        max_iter (int, optional): Maximal iterations. Defaults to 1_000_000.
        verbose (bool, optional): Print information. Defaults to False.

    Returns:
        frustration (float): Frustration index of the matrix [0, 1]
        s (np.array): Gauge transformation vector (+1 / -1)
    """
    A = A_mat.tocoo()

    idx_row = A.row
    idx_col = A.col
    weights = A.data
    return compute_frustration_edgelist(idx_row, idx_col, weights, max_iter, verbose)