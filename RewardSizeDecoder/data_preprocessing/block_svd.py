import numpy as np
import os


def stream_mean_std(stream_segments, P, compute_std=True, dtype=np.float32, eps=1e-12):
    mean = np.zeros(P, dtype=np.float64)
    M2   = np.zeros(P, dtype=np.float64) if compute_std else None
    n = 0
    for X_seg in stream_segments():          # X: (T_i, P)
        X_seg = X_seg.astype(np.float64, copy=False)
        for row in X_seg:
            n += 1
            delta = row - mean
            mean += delta / n
            if compute_std:
                M2 += delta * (row - mean)
    mu = mean.astype(dtype)
    if not compute_std:
        return mu, None
    var = (M2 / max(n - 1, 1)).astype(dtype)
    std = np.sqrt(var, dtype=dtype)
    std = np.maximum(std, eps)
    return mu, std


def per_chunk_block(X_i, mu, std, k_seg):
    # X_i: (T_i, P), mu: (P,), k_seg <= min(T_i, P)
    print('start an increment')

    Xc = X_i.astype(np.float32, copy=False)
    Xc = (Xc - mu) / std
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)  # U:(T_i,r), S:(r,), VT:(r,P)
    S_k  = S[:k_seg]
    VT_k = VT[:k_seg, :]                 # (k, P)
    B_i  = (VT_k.T * S_k)                # (P, k)  == V_i Σ_i
    return U[:, :k_seg], S_k, VT_k, B_i


# -----------------------------
# full pipeline (RAM only)
# -----------------------------
def svd_in_blocks(
    X,
    n_segments,
    P,
    k_seg=300,          # per-chunk components to keep
    k_global=500,       # number of temporal components to output
    dtype=np.float32,
):
    """
    Returns:
      W  : (T_total, k_global) temporal components
      Ug : (P, k_global) global spatial basis
      Sg : (k_global,) singular values (from SVD(B))
      mu : (P,) global mean used for centering
    """
    frames_per_seg = int(np.ceil(X.shape[0] / n_segments))
    stream_segments = lambda: make_stream_segments_by_frames(X, frames_per_seg)

    # Normalize in-place, columnwise, in small blocks to cap memory
    std = X.std(axis=0)
    mu = X.mean(axis=0)

    # ---- pass 0: count total frames & segments for sizing ----
    n_seg, T_total = 0, 0
    for X_seg in stream_segments():
        n_seg += 1
        T_total += X_seg.shape[0]

    # ---- pass 1: global mean (mean-only; no scaling for minimal steps) ----
    #mu, std = stream_mean_std(stream_segments, P, compute_std=True, dtype=dtype)

    # ---- pass 2: build B in RAM by concatenating per-chunk B_i ----
    L = n_seg * k_seg                      # total columns in B
    B = np.zeros((P, L), dtype=dtype)      # RAM allocation
    c0 = 0
    for X_seg in stream_segments():
        print(X_seg.shape)
        _, _, _, B_i = per_chunk_block(X_seg, mu, std, k_seg)  # B_i: (P, k_seg)
        print('finish an increment')
        c1 = c0 + k_seg
        B[:, c0:c1] = B_i
        c0 = c1

    # ---- one-liner economy SVD on B (P x L) ----
    # U_b:(P, r), S_b:(r,), VT_b:(r, L); r = min(P, L)
    U_b, S_b, VT_b = np.linalg.svd(B.astype(np.float64, copy=False), full_matrices=False)
    # take first k_global
    Ug = U_b[:, :k_global]                # global spatial basis (P x k)
    Sg = S_b[:k_global]

    # ---- pass 3: project all frames to get temporal components W ----
    W = np.zeros((T_total, k_global), dtype=dtype)
    t0 = 0
    for X_seg in stream_segments():
        Xc = (X_seg.astype(dtype, copy=False) - mu) / std
        Wi = Xc @ Ug                      # (T_i, k_global)
        t1 = t0 + X_seg.shape[0]
        W[t0:t1, :] = Wi
        t0 = t1

    return W, Ug


def make_stream_segments_by_frames(X: np.ndarray, frames_per_seg: int):
    """
    Yield contiguous chunks with up to frames_per_seg rows each.
    """
    T, _ = X.shape
    if frames_per_seg <= 0:
        raise ValueError("frames_per_seg must be >= 1")
    for t0 in range(0, T, frames_per_seg):
        t1 = min(t0 + frames_per_seg, T)
        yield X[t0:t1, :]






# W is (T_total x k_global): the first k temporal components.