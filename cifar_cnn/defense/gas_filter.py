"""
GAS — Gradient Anomaly Splitting. Pure statistical filter, NO machine learning.

Added as a THIRD, PARALLEL filtering layer alongside L1 (DBSCAN) and L2
(distance+direction). It does not read or modify any L1/L2 state — callers
combine results by union:

    final_rejected = L1L2_rejected  ∪  gas_flagged

Method (spec-locked, one-variable-at-a-time GAS ablation project):
  1. Split the d-dimensional gradient into p=10 chunks along a FIXED random
     permutation of coordinates (seed=1234 by default) — identical partition
     across all rounds/seeds so results are comparable.
  2. Per chunk, the reference is the coordinate-wise median of that chunk,
     computed over clients that already passed the same 70%-closest-to-median
     prefilter GDS uses (noniid_handler.py L178) — for methodological
     consistency, not by sharing runtime state with GDS.
  3. gas_score[client] = sum over chunks of || sub_vector - reference ||.
  4. Flag if gas_score > median(gas_score) + K_GAS * MAD(gas_score). This is
     an adaptive threshold — NOT top-k, since the attacker count is forbidden
     information at run time.
"""

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

GAS_NUM_CHUNKS = 10
GAS_SEED = 1234
GAS_K_DEFAULT = 2.5
GAS_PREFILTER_KEEP_RATIO = 0.7
GAS_K_SWEEP_DEFAULT = (1.5, 2.0, 2.5, 3.0, 3.5)


def _prefilter_indices(grad_matrix: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Replicate NonIIDHandler's keep_ratio pre-filter (closest-to-median clients)."""
    n = len(grad_matrix)
    g_median = np.median(grad_matrix, axis=0)
    dists = np.linalg.norm(grad_matrix - g_median, axis=1)
    num_keep = max(2, int(n * keep_ratio))
    return np.argsort(dists)[:num_keep]


def _chunk_indices(d: int, p: int, seed: int) -> List[np.ndarray]:
    """Fixed-seed random partition of d coordinates into p (near-)equal chunks."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(d)
    return np.array_split(perm, p)


def _mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def compute_gas_scores(
    full_gradients: Sequence[np.ndarray],
    seq_cids: Sequence[int],
    p: int = GAS_NUM_CHUNKS,
    seed: int = GAS_SEED,
    prefilter_keep_ratio: float = GAS_PREFILTER_KEEP_RATIO,
) -> Dict[int, float]:
    """gas_score per client = sum over p chunks of ||sub_vector - chunk_reference||."""
    n = len(full_gradients)
    if n == 0:
        return {}
    grad_matrix = np.vstack([g.flatten() for g in full_gradients])
    d = grad_matrix.shape[1]
    if n < 3:
        return {seq_cids[i]: 0.0 for i in range(n)}

    safe_idx = _prefilter_indices(grad_matrix, prefilter_keep_ratio)
    safe_matrix = grad_matrix[safe_idx]

    scores = np.zeros(n)
    for chunk in _chunk_indices(d, p, seed):
        if len(chunk) == 0:
            continue
        ref = np.median(safe_matrix[:, chunk], axis=0)
        scores += np.linalg.norm(grad_matrix[:, chunk] - ref, axis=1)

    return {seq_cids[i]: float(scores[i]) for i in range(n)}


def gas_flag_from_scores(
    gas_scores: Dict[int, float],
    k_gas: float = GAS_K_DEFAULT,
) -> Tuple[Set[int], float, float]:
    """Adaptive MAD-threshold flagging. Returns (flagged_set, median, mad)."""
    if not gas_scores:
        return set(), 0.0, 0.0
    ids = list(gas_scores.keys())
    vals = np.array([gas_scores[i] for i in ids])
    median = float(np.median(vals))
    mad = _mad(vals)
    # GUARD: điểm GAS suy biến (mọi client gần bằng nhau) → mad≈0 → thr=median
    # sẽ gắn cờ ~nửa số client (FPR giả). Không có độ tản thì KHÔNG gắn cờ ai.
    if mad < 1e-9:
        return set(), median, mad
    thr = median + k_gas * mad
    flagged = {cid for cid, s in gas_scores.items() if s > thr}
    return flagged, median, mad


def gas_filter(
    full_gradients: Sequence[np.ndarray],
    seq_cids: Sequence[int],
    p: int = GAS_NUM_CHUNKS,
    seed: int = GAS_SEED,
    k_gas: float = GAS_K_DEFAULT,
    prefilter_keep_ratio: float = GAS_PREFILTER_KEEP_RATIO,
) -> Tuple[Set[int], Dict]:
    """Run the full GAS layer. Returns (gas_flagged_set, info) where info holds
    the raw per-client scores plus median/mad/threshold (no ground truth used)."""
    gas_scores = compute_gas_scores(
        full_gradients, seq_cids, p=p, seed=seed, prefilter_keep_ratio=prefilter_keep_ratio
    )
    flagged, median, mad = gas_flag_from_scores(gas_scores, k_gas=k_gas)
    info = {
        "gas_scores": gas_scores,
        "gas_score_median": median,
        "gas_score_mad": mad,
        "k_gas": k_gas,
        "threshold": median + k_gas * mad,
    }
    return flagged, info


def compute_gas_diagnostics(
    gas_scores: Dict[int, float],
    gt_malicious: Dict[int, bool],
    k_gas: float = GAS_K_DEFAULT,
    k_sweep: Sequence[float] = GAS_K_SWEEP_DEFAULT,
) -> Dict:
    """Diagnostic-only numbers (uses ground truth) — NEVER used to decide flagging,
    only to calibrate/understand the threshold offline (per attack/alpha)."""
    ids = list(gas_scores.keys())
    scores = np.array([gas_scores[i] for i in ids])
    is_mal = np.array([bool(gt_malicious.get(i, False)) for i in ids])

    median = float(np.median(scores)) if len(scores) else 0.0
    mad = _mad(scores)

    benign_scores = scores[~is_mal]
    attacker_scores = scores[is_mal]

    median_benign = float(np.median(benign_scores)) if len(benign_scores) else None
    median_attacker = float(np.median(attacker_scores)) if len(attacker_scores) else None
    mad_benign = _mad(benign_scores) if len(benign_scores) else None

    if median_benign is not None and median_attacker is not None and mad_benign and mad_benign > 1e-12:
        separation_mad = (median_attacker - median_benign) / mad_benign
    else:
        separation_mad = None

    if len(benign_scores) and len(attacker_scores):
        min_attacker = float(np.min(attacker_scores))
        overlap_rate = float(np.mean(benign_scores > min_attacker))
    else:
        overlap_rate = None

    thr = median + k_gas * mad
    benign_pct_above_thr = float(np.mean(benign_scores > thr)) if len(benign_scores) else None
    thr_percentile_in_benign = float(np.mean(benign_scores <= thr)) if len(benign_scores) else None
    attacker_pct_below_thr = float(np.mean(attacker_scores < thr)) if len(attacker_scores) else None

    k_sweep_out = {}
    for k in k_sweep:
        t = median + k * mad
        nb = int(np.sum(benign_scores > t)) if len(benign_scores) else 0
        na = int(np.sum(attacker_scores > t)) if len(attacker_scores) else 0
        k_sweep_out[str(k)] = {"n_benign_flag": nb, "n_attacker_flag": na}

    return {
        "gas_score_median": median,
        "gas_score_mad": mad,
        "gas_median_benign": median_benign,
        "gas_median_attacker": median_attacker,
        "gas_mad_benign": mad_benign,
        "separation_mad": separation_mad,
        "overlap_rate": overlap_rate,
        "thr": thr,
        "benign_pct_above_thr": benign_pct_above_thr,
        "attacker_pct_below_thr": attacker_pct_below_thr,
        "thr_percentile_in_benign": thr_percentile_in_benign,
        "k_sweep": k_sweep_out,
    }
