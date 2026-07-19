"""
WarmupCosineProbe — logging-only. Records per-client cosine similarity during
warmup rounds (1..warmup_rounds), computed EXACTLY the way Layer2Detector would
(reuses its pure `_compute_metrics` method, reference=None -> median of the given
gradients, matching its own internal fallback formula) — but WITHOUT ever calling
Layer2Detector.detect(), so nothing about the warmup pipeline's behavior changes.

Scope: only ever fed `trusted_grads` (guaranteed-benign clients — see A0
verification in THRESH_COMPARE/version_snapshot.txt) — no ground-truth column
needed, warmup cosine is always benign-only by construction.

Opt-in via WARMUP_COSINE_PROBE_OUTPUT env var (see server_app.py).
"""

import csv
import os
from typing import List


HEADERS = ["dataset", "alpha", "seed", "round", "is_warmup", "client_id", "cosine"]


class WarmupCosineProbe:
    def __init__(self, dataset: str, alpha: float, seed: int, output_path: str):
        self.dataset = dataset
        self.alpha = alpha
        self.seed = seed
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        write_header = not os.path.exists(output_path)
        self._f = open(output_path, "a", newline="")
        self._w = csv.writer(self._f)
        if write_header:
            self._w.writerow(HEADERS)

    def record_round(self, server_round: int, client_ids: List[int], cosines) -> None:
        for cid, cos in zip(client_ids, cosines):
            self._w.writerow([self.dataset, self.alpha, self.seed, server_round, 1, cid, float(cos)])
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
