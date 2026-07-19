"""
4-Analysis Instrumentation (A/B/C/D) — logging-only hooks for offline analysis.
NO defense logic is modified. All _compute_raw_metrics calls go through the SAME
noniid_handler instance used by the defense (identical formula, different row subsets).

Files per run (instrument/{attack}_a{alpha}/):
  version_snapshot.txt     — config freeze at run start
  C_warmup.csv             — (C) raw GDS components per warmup round (BENIGN_GT)
  C_warmup_summary.csv     — (C) anchor + std computed after warmup finalization
  D_signals.csv            — (D) set-reflection components, post-warmup rounds 11-50
  [DUMP_ATTACKS = {minmax, alie, backdoor} only]:
  A_grads_r{r}.npy         — (A) GAS probe: full gradient matrix at rounds 15,25,35,45
  A_cids_r{r}.npy          — (A) partition_ids in row order
  A_isatt_r{r}.npy         — (A) bool: 1 if partition_id < 12 (attacker)
  A_status_r{r}.npy        — (A) int: 0=ACCEPTED 1=SUSPICIOUS 2=REJECTED
  B_momentum_r{r}.npy      — (B) historical_momentum snapshot BEFORE blend, all post-warmup
  B_median_r{r}.npy        — (B) current_median over ALL clients this round
  B_cleanagg_r{r}.npy      — (B) mean of BENIGN_GT (partition 12-35) gradients

GDSProbeAB — NEW lightweight probe (no .npy) for run_instrumentation_AB.py:
Files per run (instrument_AB_l{n}/{attack}_a{alpha}/):
  A_gas_raw.csv    — (A) GAS probe + DR sanity at rounds 15,25,35,45
  B_mom_raw.csv    — (B) momentum vs median cosine, all post-warmup rounds
  probe_sanity.csv — MISMATCH rows (empty if baseline correct)
"""

import csv
import os
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ── Constants (CONFIG from spec) ─────────────────────────────────────────────
BENIGN_GT_MIN = 12    # partition_id inclusive lower bound for BENIGN_GT
BENIGN_GT_MAX = 35    # partition_id inclusive upper bound for BENIGN_GT

SUBN_SEED    = 1234
SUBN_REPS    = 5
LAMBDA_SLOW  = 0.05   # EMA decay for dynamic shadow baseline

GAS_DUMP_ROUNDS = {15, 25, 35, 45}
DUMP_ATTACKS    = {'backdoor', 'minmax', 'alie'}

PREFILTER_KEEP_RATIO = 0.7   # must match NonIIDHandler.compute_heterogeneity_score


# ── Helper ───────────────────────────────────────────────────────────────────

def _prefilter_indices(grad_matrix: np.ndarray) -> np.ndarray:
    """Replicate the keep_ratio=0.7 pre-filter from NonIIDHandler."""
    n = len(grad_matrix)
    g_median = np.median(grad_matrix, axis=0)
    dists = np.linalg.norm(grad_matrix - g_median, axis=1)
    num_keep = max(2, int(n * PREFILTER_KEEP_RATIO))
    return np.argsort(dists)[:num_keep]


# ── version_snapshot ─────────────────────────────────────────────────────────

def write_version_snapshot(
    output_dir: str,
    attack: str,
    alpha: float,
    dataset: str,
    cosine_threshold: float,
    k_reject: float,
    distance_multiplier: float,
    weight_baseline: float,
    defense_params: dict,
) -> None:
    """Write frozen config snapshot at run start. ABORTS if cosine≠0.8 or dataset≠cifar."""
    os.makedirs(output_dir, exist_ok=True)
    noniid = defense_params.get('noniid', {})
    lines = [
        "# version_snapshot — run start",
        f"dataset: {dataset}",
        f"cosine_threshold: {cosine_threshold}",
        f"mad_k_reject: {k_reject}",
        f"distance_multiplier: {distance_multiplier}",
        f"weight_baseline (confidence): {weight_baseline}",
        f"GDS_weights: cv={noniid.get('weight_cv', 0.4)}, "
        f"sim={noniid.get('weight_sim', 0.4)}, "
        f"cluster={noniid.get('weight_cluster', 0.2)}",
        "normalization: on (GDS_NORM_RANGES_CIFAR10 applied per component before weighted sum)",
        "variant: B (theta_signal = GDS, NOT H)",
        f"pre_filter_keep_ratio: {PREFILTER_KEEP_RATIO}",
        f"benign_gt_range: [{BENIGN_GT_MIN}, {BENIGN_GT_MAX}]",
        f"attack: {attack}",
        f"alpha_dirichlet: {alpha}",
        f"subn_seed: {SUBN_SEED}  subn_reps: {SUBN_REPS}",
        f"lambda_slow: {LAMBDA_SLOW}",
        f"gas_dump_rounds: {sorted(GAS_DUMP_ROUNDS)}",
        f"dump_npy_for_attacks: {sorted(DUMP_ATTACKS)}",
        "# C_warmup.csv    — raw components per warmup round (BENIGN_GT, normalization OFF)",
        "# C_warmup_summary.csv — anchor=mean, std over warmup rounds (normalization OFF)",
        "# D_signals.csv   — ALL/PREFILTER/BENIGN_GT/ACCEPTED/SUBN per round (normalization OFF)",
        "# A_*_r{r}.npy    — GAS probe at rounds 15,25,35,45 (DUMP_ATTACKS only)",
        "# B_*_r{r}.npy    — momentum verify, all post-warmup rounds (DUMP_ATTACKS only)",
    ]
    path = os.path.join(output_dir, "version_snapshot.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"   [PROBE] version_snapshot → {path}")

    # ABORT guards
    if abs(cosine_threshold - 0.8) > 1e-6:
        raise RuntimeError(
            f"ABORT: cosine_threshold={cosine_threshold} ≠ 0.8 (required by spec). "
            "Fix defense.layer2.cosine-threshold in pyproject.toml."
        )
    if "cifar" not in dataset.lower():
        raise RuntimeError(
            f"ABORT: dataset='{dataset}' ≠ cifar-10 (required by spec). "
            "Fix dataset in pyproject.toml."
        )


# ── GDSProbe ─────────────────────────────────────────────────────────────────

class GDSProbe:
    """
    Logging-only instrumentation that reuses the defense's noniid_handler
    to compute raw GDS components on different client subsets (no side effects).

    Analysis A: GAS probe — gradient dump at 4 probe rounds
    Analysis B: Momentum verify — historical_momentum vs median vs cleanagg
    Analysis C: Warmup baseline — raw components on BENIGN_GT during warmup
    Analysis D: Set reflection — compare ALL/PREFILTER/BENIGN_GT/ACCEPTED/SUBN post-warmup
    """

    def __init__(
        self,
        attack: str,
        alpha: float,
        output_dir: str,
        noniid_handler,          # NonIIDHandler instance — only _compute_raw_metrics used
    ):
        self.attack      = attack
        self.alpha       = alpha
        self.output_dir  = output_dir
        self.nh          = noniid_handler
        self.do_npy      = attack in DUMP_ATTACKS

        os.makedirs(output_dir, exist_ok=True)

        # ── (C) Warmup accumulation ──────────────────────────────────────────
        self._wu_dcv:      List[float] = []
        self._wu_dsim:     List[float] = []
        self._wu_dcluster: List[float] = []

        # ── Anchors (mean of warmup) ─────────────────────────────────────────
        self.anchor_DCV:      Optional[float] = None
        self.anchor_Dsim:     Optional[float] = None
        self.anchor_Dcluster: Optional[float] = None

        # ── Dynamic shadow baseline (EMA, LAMBDA_SLOW; NEVER fed into defense) ─
        self._dyn_DCV:      Optional[float] = None
        self._dyn_Dsim:     Optional[float] = None
        self._dyn_Dcluster: Optional[float] = None

        # ── (C) C_warmup.csv ────────────────────────────────────────────────
        self._wu_f = open(os.path.join(output_dir, "C_warmup.csv"), "w", newline="")
        self._wu_w = csv.writer(self._wu_f)
        self._wu_w.writerow([
            "# (C) WARMUP BASELINE | normalization=off | raw metrics on BENIGN_GT clients | "
            "GDS_weights cv=0.4 sim=0.4 cluster=0.2"
        ])
        self._wu_w.writerow(["round", "DCV", "Dsim", "Dcluster"])

        # ── (D) D_signals.csv ────────────────────────────────────────────────
        self._d_f = open(os.path.join(output_dir, "D_signals.csv"), "w", newline="")
        self._d_w = csv.writer(self._d_f)
        self._d_w.writerow([
            "# (D) SET REFLECTION | normalization=off (raw, no clip) | "
            "GDS_weights cv=0.4 sim=0.4 cluster=0.2 | "
            "SHADOW baselines (static/dynamic) NOT fed into defense | "
            "SUBN: sample BENIGN_GT to |ALL| size (without replacement, seed=1234, reps=5)"
        ])
        self._d_w.writerow([
            "round", "n_total", "n_benign", "n_accepted", "n_attacker",
            # DCV columns
            "DCV_all", "DCV_prefilter", "DCV_benignGT", "DCV_accepted",
            "DCV_subN_mean", "DCV_subN_std", "DCV_static", "DCV_dynamic",
            # Dsim columns
            "Dsim_all", "Dsim_prefilter", "Dsim_benignGT", "Dsim_accepted",
            "Dsim_subN_mean", "Dsim_subN_std", "Dsim_static", "Dsim_dynamic",
            # Dcluster columns
            "Dcluster_all", "Dcluster_prefilter", "Dcluster_benignGT", "Dcluster_accepted",
            "Dcluster_subN_mean", "Dcluster_subN_std", "Dcluster_static", "Dcluster_dynamic",
        ])

    # =========================================================================
    # (C) WARMUP BASELINE
    # =========================================================================

    def record_warmup(self, server_round: int, trusted_grads: list) -> None:
        """
        Record one warmup round.
        trusted_grads = BENIGN_GT gradients (partition 12-35; warmup only uses trusted clients).
        """
        if not trusted_grads:
            return
        grad_matrix = np.vstack([g.flatten() for g in trusted_grads])
        dcv, dsim, dcluster = self.nh._compute_raw_metrics(grad_matrix)

        self._wu_dcv.append(dcv)
        self._wu_dsim.append(dsim)
        self._wu_dcluster.append(dcluster)

        self._wu_w.writerow([server_round, dcv, dsim, dcluster])
        self._wu_f.flush()

    def finalize_warmup(self) -> None:
        """
        Compute anchor = mean and std of warmup components.
        Write C_warmup_summary.csv. Initialize dynamic shadow to anchor.
        Call after the last warmup round.
        """
        if not self._wu_dcv:
            print("   [PROBE] WARNING: No warmup data — anchor not set.")
            return

        self.anchor_DCV      = float(np.mean(self._wu_dcv))
        self.anchor_Dsim     = float(np.mean(self._wu_dsim))
        self.anchor_Dcluster = float(np.mean(self._wu_dcluster))

        std_DCV      = float(np.std(self._wu_dcv))
        std_Dsim     = float(np.std(self._wu_dsim))
        std_Dcluster = float(np.std(self._wu_dcluster))

        # Initialize dynamic shadow to anchor
        self._dyn_DCV      = self.anchor_DCV
        self._dyn_Dsim     = self.anchor_Dsim
        self._dyn_Dcluster = self.anchor_Dcluster

        # Write C_warmup_summary.csv
        summary_path = os.path.join(self.output_dir, "C_warmup_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "# (C) WARMUP SUMMARY | anchor=mean over warmup rounds | "
                "std=env oscillation baseline | normalization=off"
            ])
            w.writerow([
                "anchor_DCV", "anchor_Dsim", "anchor_Dcluster",
                "std_DCV", "std_Dsim", "std_Dcluster",
            ])
            w.writerow([
                self.anchor_DCV, self.anchor_Dsim, self.anchor_Dcluster,
                std_DCV, std_Dsim, std_Dcluster,
            ])

        print(
            f"   [PROBE] Warmup anchor ({len(self._wu_dcv)} rounds): "
            f"DCV={self.anchor_DCV:.4f}±{std_DCV:.4f}  "
            f"Dsim={self.anchor_Dsim:.4f}±{std_Dsim:.4f}  "
            f"Dcluster={self.anchor_Dcluster:.4f}±{std_Dcluster:.4f}"
        )

    # =========================================================================
    # (D) SET REFLECTION
    # =========================================================================

    def record_round(
        self,
        server_round: int,
        full_gradients: list,
        seq_cids: list,
        all_filtered: set,
        GDS_prefilter_norm: float,   # kept for caller compatibility (not written to D_signals)
        alpha_blend_used: float,     # kept for caller compatibility (not written to D_signals)
        precision: float,            # kept for caller compatibility (not written to D_signals)
        recall: float,               # kept for caller compatibility (not written to D_signals)
        fpr: float,                  # kept for caller compatibility (not written to D_signals)
    ) -> None:
        """
        Compute and log raw GDS components for ALL/PREFILTER/BENIGN_GT/ACCEPTED/SUBN sets.
        Calls self.nh._compute_raw_metrics — same formula as defense, different row subsets.
        D_signals.csv does NOT include detection metrics (those are auxiliary).
        """
        if self.anchor_DCV is None:
            return
        n = len(full_gradients)
        if n < 2:
            return

        grad_matrix = np.vstack([g.flatten() for g in full_gradients])

        # ── Define index sets ────────────────────────────────────────────────
        prefilter_idx = _prefilter_indices(grad_matrix)

        benign_gt_idx = np.array(
            [i for i, cid in enumerate(seq_cids) if BENIGN_GT_MIN <= cid <= BENIGN_GT_MAX],
            dtype=int,
        )
        accepted_idx = np.array(
            [i for i, cid in enumerate(seq_cids) if cid not in all_filtered],
            dtype=int,
        )

        n_total    = n
        n_benign   = int(len(benign_gt_idx))
        n_accepted = int(len(accepted_idx))
        n_attacker = int(sum(1 for cid in seq_cids if cid < BENIGN_GT_MIN))

        # ── Raw metrics per set (ALL via _compute_raw_metrics) ───────────────
        C_all = self.nh._compute_raw_metrics(grad_matrix)
        C_pf  = self.nh._compute_raw_metrics(grad_matrix[prefilter_idx])
        C_bg  = (
            self.nh._compute_raw_metrics(grad_matrix[benign_gt_idx])
            if n_benign >= 2 else (0.0, 0.0, 0.0)
        )
        C_ac  = (
            self.nh._compute_raw_metrics(grad_matrix[accepted_idx])
            if n_accepted >= 2 else (0.0, 0.0, 0.0)
        )

        # ── SUBN: sample BENIGN_GT down to |ALL| size, SUBN_REPS repeats ────
        # Size target = n_total (|ALL|); if n_benign < n_total, use n_benign (max available).
        subn_dcv:      List[float] = []
        subn_dsim:     List[float] = []
        subn_dcluster: List[float] = []

        if n_benign >= 2:
            rng       = random.Random(SUBN_SEED)
            sample_n  = min(n_total, n_benign)   # "xuống ĐÚNG |ALL|"; cap at n_benign
            bg_list   = list(benign_gt_idx)
            for _ in range(SUBN_REPS):
                samp = rng.sample(bg_list, sample_n)
                r = self.nh._compute_raw_metrics(grad_matrix[np.array(samp)])
                subn_dcv.append(r[0])
                subn_dsim.append(r[1])
                subn_dcluster.append(r[2])

        def _ms(vals):
            if vals:
                return float(np.mean(vals)), float(np.std(vals))
            return 0.0, 0.0

        sn_dcv_m,  sn_dcv_s  = _ms(subn_dcv)
        sn_dsim_m, sn_dsim_s = _ms(subn_dsim)
        sn_dcl_m,  sn_dcl_s  = _ms(subn_dcluster)

        # ── Dynamic shadow EMA update (SHADOW, NOT fed into defense) ─────────
        # Use ACCEPTED as proxy for clean; fall back to BENIGN_GT if empty.
        dyn_in_dcv = C_ac[0] if n_accepted >= 2 else C_bg[0]
        dyn_in_dsim = C_ac[1] if n_accepted >= 2 else C_bg[1]
        dyn_in_dcl  = C_ac[2] if n_accepted >= 2 else C_bg[2]

        self._dyn_DCV      = (1 - LAMBDA_SLOW) * self._dyn_DCV      + LAMBDA_SLOW * dyn_in_dcv
        self._dyn_Dsim     = (1 - LAMBDA_SLOW) * self._dyn_Dsim     + LAMBDA_SLOW * dyn_in_dsim
        self._dyn_Dcluster = (1 - LAMBDA_SLOW) * self._dyn_Dcluster + LAMBDA_SLOW * dyn_in_dcl

        # ── Write D_signals row ──────────────────────────────────────────────
        self._d_w.writerow([
            server_round, n_total, n_benign, n_accepted, n_attacker,
            # DCV
            C_all[0], C_pf[0], C_bg[0], C_ac[0],
            sn_dcv_m, sn_dcv_s, self.anchor_DCV, self._dyn_DCV,
            # Dsim
            C_all[1], C_pf[1], C_bg[1], C_ac[1],
            sn_dsim_m, sn_dsim_s, self.anchor_Dsim, self._dyn_Dsim,
            # Dcluster
            C_all[2], C_pf[2], C_bg[2], C_ac[2],
            sn_dcl_m, sn_dcl_s, self.anchor_Dcluster, self._dyn_Dcluster,
        ])
        self._d_f.flush()

    # =========================================================================
    # (A) GAS PROBE — dump at probe rounds
    # =========================================================================

    def dump_gas_npy(
        self,
        server_round: int,
        full_gradients: list,
        seq_cids: list,
        all_filtered: set,
        client_statuses: Optional[Dict] = None,  # dict[seq_id → ClientStatus]
    ) -> None:
        """
        Dump full gradient matrix + metadata at GAS_DUMP_ROUNDS.
        A_status encoding: 0=ACCEPTED 1=SUSPICIOUS 2=REJECTED/FILTERED
        """
        if not self.do_npy or server_round not in GAS_DUMP_ROUNDS:
            return

        from cifar_cnn.defense.reputation import ClientStatus

        grad_matrix = np.vstack([g.flatten() for g in full_gradients]).astype(np.float32)
        cids_arr = np.array(seq_cids, dtype=np.int32)
        is_att   = np.array([cid < BENIGN_GT_MIN for cid in seq_cids], dtype=bool)

        # Status: 2=REJECTED/FILTERED, 1=SUSPICIOUS (accepted but flagged), 0=ACCEPTED
        status_list = []
        for cid in seq_cids:
            if cid in all_filtered:
                status_list.append(2)
            elif (client_statuses is not None and
                  client_statuses.get(cid) == ClientStatus.SUSPICIOUS):
                status_list.append(1)
            else:
                status_list.append(0)
        status_arr = np.array(status_list, dtype=np.int32)

        r = server_round
        np.save(os.path.join(self.output_dir, f"A_grads_r{r}.npy"),    grad_matrix)
        np.save(os.path.join(self.output_dir, f"A_cids_r{r}.npy"),     cids_arr)
        np.save(os.path.join(self.output_dir, f"A_isatt_r{r}.npy"),    is_att)
        np.save(os.path.join(self.output_dir, f"A_status_r{r}.npy"),   status_arr)
        print(
            f"   [PROBE] (A) GAS r{r}: shape={grad_matrix.shape}  "
            f"n_att={is_att.sum()}  "
            f"n_acc={(status_arr == 0).sum()}  n_susp={(status_arr == 1).sum()}  "
            f"n_rej={(status_arr == 2).sum()}"
        )

    # =========================================================================
    # (B) MOMENTUM VERIFY — all post-warmup rounds
    # =========================================================================

    def dump_momentum_npy(
        self,
        server_round: int,
        full_gradients: list,
        seq_cids: list,
        historical_momentum_snapshot,   # captured BEFORE compute_adaptive_reference
    ) -> None:
        """
        Dump historical_momentum / current_median / cleanagg for momentum analysis.
        B_momentum_r{r}.npy — historical_momentum BEFORE blend
        B_median_r{r}.npy   — current_median over ALL clients
        B_cleanagg_r{r}.npy — mean of BENIGN_GT clients only
        """
        if not self.do_npy:
            return

        grad_matrix = np.vstack([g.flatten() for g in full_gradients])
        r = server_round

        if historical_momentum_snapshot is not None:
            np.save(
                os.path.join(self.output_dir, f"B_momentum_r{r}.npy"),
                historical_momentum_snapshot.astype(np.float32),
            )

        current_median = np.median(grad_matrix, axis=0)
        np.save(
            os.path.join(self.output_dir, f"B_median_r{r}.npy"),
            current_median.astype(np.float32),
        )

        bg_idx = [i for i, cid in enumerate(seq_cids) if BENIGN_GT_MIN <= cid <= BENIGN_GT_MAX]
        if bg_idx:
            cleanagg = np.mean(grad_matrix[bg_idx], axis=0)
            np.save(
                os.path.join(self.output_dir, f"B_cleanagg_r{r}.npy"),
                cleanagg.astype(np.float32),
            )

        if historical_momentum_snapshot is not None and bg_idx:
            mom_norm    = float(np.linalg.norm(historical_momentum_snapshot))
            median_norm = float(np.linalg.norm(current_median))
            clean_norm  = float(np.linalg.norm(cleanagg))
            print(
                f"   [PROBE] (B) momentum r{r}: "
                f"||momentum||={mom_norm:.2f}  ||median||={median_norm:.2f}  "
                f"||cleanagg||={clean_norm:.2f}  n_benign={len(bg_idx)}"
            )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def close(self) -> None:
        for f in (self._wu_f, self._d_f):
            try:
                f.close()
            except Exception:
                pass


# ── GDSProbeAB ───────────────────────────────────────────────────────────────

class GDSProbeAB:
    """
    Lightweight probe for analyses A (GAS, corrected baseline) and B (Momentum verify).
    CSV-only — no .npy dumps, no defense logic changes.
    Used by run_instrumentation_AB.py for 7×4×3 = 84 runs.

    Caller contract (called from server_app.aggregate_fit AFTER all_filtered computed):
      probe_ab.record_ab_round(server_round, full_gradients, seq_cids,
                               gt_malicious_batch, all_filtered,
                               DR_log, historical_momentum_snapshot)

    Outputs per run dir:
      A_gas_raw.csv    — per GAS round (15,25,35,45): AUC, gain, DR sanity
      B_mom_raw.csv    — per post-warmup round: cosine comparisons
      probe_sanity.csv — MISMATCH rows (must be empty if baseline is correct)
    """

    GAS_ROUNDS     = {15, 25, 35, 45}
    GAS_SPLITS     = 10
    GAS_SEED       = 1234
    BENIGN_GT_MIN  = 12
    BENIGN_GT_MAX  = 35
    WARMUP_ROUNDS  = 10

    def __init__(self, attack: str, alpha: float, output_dir: str) -> None:
        self.attack = attack
        self.alpha  = alpha
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # A_gas_raw.csv
        self._a_f = open(os.path.join(output_dir, "A_gas_raw.csv"), "w", newline="")
        self._a_w = csv.writer(self._a_f)
        self._a_w.writerow([
            "attack", "alpha", "round",
            "gas_AUC", "n_attacker", "L1L2_caught", "gas_caught_topk",
            "gas_gain", "DR_probe", "DR_log", "valid",
        ])

        # B_mom_raw.csv
        self._b_f = open(os.path.join(output_dir, "B_mom_raw.csv"), "w", newline="")
        self._b_w = csv.writer(self._b_f)
        self._b_w.writerow([
            "attack", "alpha", "round", "cos_mom", "cos_med", "mom_cleaner",
        ])

        # probe_sanity.csv (MISMATCH rows → must remain empty)
        self._s_f = open(os.path.join(output_dir, "probe_sanity.csv"), "w", newline="")
        self._s_w = csv.writer(self._s_f)
        self._s_w.writerow(["attack", "alpha", "round", "DR_probe", "DR_log", "status"])

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.flatten(), b.flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ── main entry point ─────────────────────────────────────────────────────

    def record_ab_round(
        self,
        server_round: int,
        full_gradients: List[np.ndarray],
        seq_cids: List[int],
        gt_malicious_batch: List[bool],
        all_filtered: set,                         # final rejected set used for DR_log
        DR_log: float,                             # recall from _calculate_metrics same round
        historical_momentum: Optional[np.ndarray], # snapshot BEFORE compute_adaptive_reference
    ) -> None:
        """
        Record A (GAS at probe rounds) and B (momentum verify every round 11-50).
        No .npy files; all computation is inline and immediately written to CSV.
        """
        if server_round <= self.WARMUP_ROUNDS:
            return

        try:
            grad_matrix = np.vstack([g.flatten() for g in full_gradients])  # (n, d)
        except Exception as exc:
            print(f"   [PROBE-AB] record_ab_round: grad_matrix build failed r{server_round}: {exc}")
            return

        is_attacker = [bool(m) for m in gt_malicious_batch]
        n_attacker  = sum(is_attacker)

        # ── B: Momentum verify (every round 11-50) ───────────────────────────
        try:
            if historical_momentum is not None:
                current_median = np.median(grad_matrix, axis=0)
                bg_idx = [i for i, cid in enumerate(seq_cids)
                          if self.BENIGN_GT_MIN <= cid <= self.BENIGN_GT_MAX]
                if bg_idx:
                    cleanagg = np.mean(grad_matrix[bg_idx], axis=0)
                    cos_mom  = self._cosine(historical_momentum, cleanagg)
                    cos_med  = self._cosine(current_median, cleanagg)
                    mom_cleaner = 1 if cos_mom > cos_med else 0
                    self._b_w.writerow([
                        self.attack, self.alpha, server_round,
                        f"{cos_mom:.6f}", f"{cos_med:.6f}", mom_cleaner,
                    ])
                    self._b_f.flush()
        except Exception as exc:
            print(f"   [PROBE-AB] B failed r{server_round}: {exc}")

        # ── A: GAS probe (only at GAS_ROUNDS) ───────────────────────────────
        if server_round not in self.GAS_ROUNDS:
            return

        try:
            if n_attacker == 0:
                print(f"   [PROBE-AB] A: r{server_round} no attackers → skip GAS")
                return

            # L1L2_caught: attackers in all_filtered (the final pipeline rejected set)
            L1L2_caught = sum(
                1 for i, cid in enumerate(seq_cids)
                if is_attacker[i] and cid in all_filtered
            )
            DR_probe = L1L2_caught / n_attacker

            # Sanity: DR_probe must match DR_log (both from all_filtered)
            valid = True
            if abs(DR_probe - DR_log) > 0.05:
                valid = False
                self._s_w.writerow([
                    self.attack, self.alpha, server_round,
                    f"{DR_probe:.4f}", f"{DR_log:.4f}", "MISMATCH",
                ])
                self._s_f.flush()
                print(
                    f"   [PROBE-AB] ⚠️  SANITY MISMATCH r{server_round}: "
                    f"DR_probe={DR_probe:.4f} DR_log={DR_log:.4f} → INVALID"
                )

            if valid:
                # GAS: split d dimensions into GAS_SPLITS chunks, sum ||sub - ref||
                n, d = grad_matrix.shape
                rng     = np.random.RandomState(self.GAS_SEED)
                indices = np.arange(d)
                rng.shuffle(indices)
                chunks  = np.array_split(indices, self.GAS_SPLITS)

                gas_scores = np.zeros(n)
                for chunk in chunks:
                    sub = grad_matrix[:, chunk]        # (n, chunk_size)
                    ref = np.median(sub, axis=0)       # coordinate-wise median
                    gas_scores += np.linalg.norm(sub - ref, axis=1)

                labels = np.array(is_attacker, dtype=int)
                gas_AUC: float
                try:
                    from sklearn.metrics import roc_auc_score
                    if 0 < labels.sum() < len(labels):
                        gas_AUC = float(roc_auc_score(labels, gas_scores))
                    else:
                        gas_AUC = float("nan")
                except Exception:
                    gas_AUC = float("nan")

                # top-k where k = n_attacker
                topk_idx    = set(int(i) for i in np.argsort(gas_scores)[-n_attacker:])
                gas_caught_topk = sum(1 for i in topk_idx if is_attacker[i])
                gas_gain    = gas_caught_topk - L1L2_caught

                self._a_w.writerow([
                    self.attack, self.alpha, server_round,
                    f"{gas_AUC:.6f}", n_attacker, L1L2_caught, gas_caught_topk,
                    gas_gain, f"{DR_probe:.4f}", f"{DR_log:.4f}", "True",
                ])
                self._a_f.flush()

                print(
                    f"   [PROBE-AB] A r{server_round}: AUC={gas_AUC:.3f} "
                    f"L1L2={L1L2_caught}/{n_attacker} gas_top={gas_caught_topk} "
                    f"gain={gas_gain}"
                )
            else:
                # Write invalid row (no GAS computed)
                self._a_w.writerow([
                    self.attack, self.alpha, server_round,
                    "nan", n_attacker, L1L2_caught, "nan",
                    "nan", f"{DR_probe:.4f}", f"{DR_log:.4f}", "False",
                ])
                self._a_f.flush()

        except Exception as exc:
            print(f"   [PROBE-AB] A failed r{server_round}: {exc}")

    # ── cleanup ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        for f in (self._a_f, self._b_f, self._s_f):
            try:
                f.close()
            except Exception:
                pass
