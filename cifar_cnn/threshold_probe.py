"""
ThresholdProbe — logging-only instrumentation for the L1 (median+k×MAD) threshold
portability analysis. NO defense logic touched: reads Layer1Detector.last_result and
Layer2Detector.last_stats (already computed each round for the real decision) and
writes the RAW per-client scores to CSV, before/alongside the real threshold decision.

Opt-in only: activated by the THRESHOLD_PROBE_OUTPUT env var (see server_app.py).
Works for ANY dataset (unlike GDSProbe, which is CIFAR-10-specific) — this probe
has no dataset-specific reference tables, it only reads generic detector state.
"""

import csv
import os
from typing import Dict, List, Optional


RAW_HEADERS = [
    "dataset", "attack", "alpha", "seed", "round", "is_warmup", "client_id", "is_attacker",
    "l1_score", "l1_median", "l1_mad", "l1_rejected",
    "l2_score", "l2_distance", "l2_rejected",
    "final_rejected",
]


class ThresholdProbe:
    def __init__(self, dataset: str, attack: str, alpha: float, seed: int, output_path: str):
        self.dataset = dataset
        self.attack = attack
        self.alpha = alpha
        self.seed = seed
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        write_header = not os.path.exists(output_path)
        self._f = open(output_path, "a", newline="")
        self._w = csv.writer(self._f)
        if write_header:
            self._w.writerow(RAW_HEADERS)

    def record_round(
        self,
        server_round: int,
        seq_cids: List[int],
        gt_malicious: List[bool],
        layer1_detector,
        layer2_detector,
        final_rejected_set,
    ) -> None:
        l1_result = layer1_detector.get_result()
        if l1_result is None:
            return
        mag = l1_result.magnitude_stats
        norms: Dict[int, float] = mag.get("norms", {})
        l1_median = mag.get("median_norm")
        l1_mad = mag.get("effective_mad", mag.get("mad"))
        l1_status = l1_result.status  # client_id -> "REJECTED"/"FLAGGED"/"ACCEPTED"

        l2_decisions = {}
        try:
            for d in layer2_detector.last_stats.get("decisions", []):
                l2_decisions[d["client_id"]] = d
        except Exception:
            pass

        for i, cid in enumerate(seq_cids):
            l1_rejected = 1 if l1_status.get(cid) == "REJECTED" else 0
            l2d = l2_decisions.get(cid, {})
            l2_score = l2d.get("cosine")
            l2_distance = l2d.get("distance")
            l2_rejected = 1 if l2d.get("final") == "REJECTED" else 0
            final_rejected = 1 if cid in final_rejected_set else 0

            self._w.writerow([
                self.dataset, self.attack, self.alpha, self.seed, server_round, 0, cid,
                int(gt_malicious[i]) if i < len(gt_malicious) else 0,
                norms.get(cid), l1_median, l1_mad, l1_rejected,
                l2_score, l2_distance, l2_rejected,
                final_rejected,
            ])
        self._f.flush()

    def record_warmup_round(
        self,
        server_round: int,
        client_ids: List[int],
        magnitude_stats: Dict,
    ) -> None:
        """
        Log raw L1 magnitude-filter scores during warmup (rounds 1-warmup_rounds).
        L1/L2 are BYPASSED in warmup (only trusted/benign clients participate, no
        defense decision is actually enforced) — this call computes the SAME
        median+MAD formula on warmup gradients PURELY for calibration data (Phân
        tích C: k_warmup). is_attacker is always 0 by construction (warmup only
        uses trusted clients — guaranteed benign, checked before this is called).
        l2_* / final_rejected are left blank: Layer2 never runs in warmup, there is
        no "final decision" to log (nothing is actually filtered).
        """
        if not magnitude_stats:
            return
        norms: Dict[int, float] = magnitude_stats.get("norms", {})
        l1_median = magnitude_stats.get("median_norm")
        l1_mad = magnitude_stats.get("effective_mad", magnitude_stats.get("mad"))
        k_reject = magnitude_stats.get("k_reject")
        thr_reject = magnitude_stats.get("threshold_reject")

        for cid in client_ids:
            score = norms.get(cid)
            l1_rejected = (
                1 if (score is not None and thr_reject is not None and score > thr_reject) else 0
            )
            self._w.writerow([
                self.dataset, self.attack, self.alpha, self.seed, server_round, 1, cid,
                0,  # is_attacker — always 0 in warmup (trusted clients only, guaranteed benign)
                score, l1_median, l1_mad, l1_rejected,
                None, None, None,   # l2_* — Layer2 never runs in warmup
                None,                # final_rejected — nothing is actually filtered in warmup
            ])
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
