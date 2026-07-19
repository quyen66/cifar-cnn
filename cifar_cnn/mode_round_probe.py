"""
ModeRoundProbe — per-round summary logger for the Stage B cosine-threshold mode
comparison (A_FIX / B_WARMUP / C_ROUND k2.5 / C_ROUND k3.5). ONE row per round:
DR, FPR, cos_threshold_actual, n_rejected_L1, n_rejected_L2, plus the requested
per-round percentile values (cos_median, cos_mad, cos_benign_p5, cos_attacker_p50)
so post-hoc analysis never needs to re-run anything.

Opt-in via MODE_ROUND_PROBE_OUTPUT env var (see server_app.py).
"""

import csv
import os
from typing import Optional


HEADERS = [
    "mode", "dataset", "attack", "alpha", "seed", "round",
    "DR", "FPR", "cos_threshold_actual",
    "n_rejected_L1", "n_rejected_L2",
    "cos_median", "cos_mad", "cos_benign_p5", "cos_attacker_p50",
]


class ModeRoundProbe:
    def __init__(self, mode: str, dataset: str, attack: str, alpha: float, seed: int, output_path: str):
        self.mode = mode
        self.dataset = dataset
        self.attack = attack
        self.alpha = alpha
        self.seed = seed
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        write_header = not os.path.exists(output_path)
        self._f = open(output_path, "a", newline="")
        self._w = csv.writer(self._f)
        if write_header:
            self._w.writerow(HEADERS)

    def record_round(
        self,
        server_round: int,
        DR: Optional[float],
        FPR: Optional[float],
        cos_threshold_actual: Optional[float],
        n_rejected_L1: int,
        n_rejected_L2: int,
        cos_median: Optional[float],
        cos_mad: Optional[float],
        cos_benign_p5: Optional[float],
        cos_attacker_p50: Optional[float],
    ) -> None:
        self._w.writerow([
            self.mode, self.dataset, self.attack, self.alpha, self.seed, server_round,
            DR, FPR, cos_threshold_actual,
            n_rejected_L1, n_rejected_L2,
            cos_median, cos_mad, cos_benign_p5, cos_attacker_p50,
        ])
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
