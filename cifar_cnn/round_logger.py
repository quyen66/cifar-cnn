"""
Round Logger — Per-Round Detection Detail Tables
=================================================
Format: 5 bảng (1 per module) + Option C summary bar.

  Table 1  — L1 Magnitude Filter      (norms vs thresholds)
  Table 2  — L1 DBSCAN Clustering     (cluster labels, outliers)
  Table 3  — L2 Distance + Cosine     (decision matrix, dist/cos per client)
  Table 4  — L2 Drift Detection       (drift trend, override)
  Table 5  — Two-Stage Filter         (ci, R, theta_hard/theta_soft/tau, Tier 1/2)
  Summary  — Option C bar

USAGE (trong server_app.py):
    self.round_logger.log_round(
        round_num=server_round,
        mode=current_mode,
        H=H,
        seq_cids=seq_cids,
        gt_malicious=gt_malicious_batch,           # [bool] parallel to seq_cids
        layer1_result=self.layer1_detector.last_result,
        layer2_stats=self.layer2_detector.last_stats,
        layer2_drift_info=self.layer2_detector.last_drift_info,
        layer2_status=layer2_status,               # Dict[cid, str] — AFTER drift override
        suspicion_levels=suspicion_levels,
        confidence_scores=confidence_scores,
        reputations=current_reputations,
        filter_stats=filter_stats,
        ref_alpha=self.ref_tracker.last_alpha_used,
        accuracy=None  # optional
    )
"""

from typing import Dict, List, Optional, Set


class RoundLogger:
    """Format + print per-round detection tables."""

    WIDTH = 108   # total line width (matches sample)

    def __init__(self, log_file: Optional[str] = None):
        """
        Args:
            log_file: If set, append each round's output to this file too.
        """
        self.log_file = log_file

    # =========================================================================
    # MAIN ENTRY
    # =========================================================================

    def log_round(
        self,
        round_num: int,
        mode: str,
        H: float,
        seq_cids: List[int],
        gt_malicious: List[bool],
        layer1_result,                       # Layer1Result dataclass
        layer2_stats: Dict,                  # layer2_detector.last_stats
        layer2_drift_info: Dict,             # layer2_detector.last_drift_info
        layer2_status: Dict[int, str],       # final L2 status (post drift-override)
        suspicion_levels: Dict[int, Optional[str]],
        confidence_scores: Dict[int, float],
        reputations: Dict[int, float],
        filter_stats: Dict,
        ref_alpha: float = 0.0,
        accuracy: Optional[float] = None,
    ) -> str:
        """Generate + print + optionally save all tables for this round."""

        # ── ground truth lookup ──
        mal_set: Set[int] = set()
        for i, cid in enumerate(seq_cids):
            if gt_malicious[i]:
                mal_set.add(cid)

        # ── final rejection set (L2 REJECTED union filter rejected) ──
        hard_ids = set(filter_stats.get('hard_filtered_ids', []))
        soft_ids = set(filter_stats.get('soft_filtered_ids', []))
        final_rejected: Set[int] = set()
        for cid in seq_cids:
            if layer2_status.get(cid) == "REJECTED" or cid in hard_ids or cid in soft_ids:
                final_rejected.add(cid)

        # ── GT label helper ──
        def gt(cid: int) -> str:
            mal = cid in mal_set
            det = cid in final_rejected
            if mal and det:      return "MAL \u2713det"
            if mal and not det:  return "MAL \u2717FN"
            if not mal and det:  return "BEN \u2717FP"
            return "BEN"

        # ── extract layer1 internals ──
        l1_status   = layer1_result.status          # Dict[int, str]
        mag         = layer1_result.magnitude_stats
        db          = layer1_result.dbscan_stats

        # ── extract layer2 per-client decisions (BEFORE drift override) ──
        l2_dec = {}
        for d in layer2_stats.get("decisions", []):
            l2_dec[d["client_id"]] = d

        # ── build output ──
        L: List[str] = []
        self._header(L, round_num, mode, H, seq_cids, mal_set)
        self._table1_magnitude(L, seq_cids, mag, l1_status, gt)
        self._table2_dbscan(L, seq_cids, db, l1_status, mal_set, gt)
        self._table3_dist_cos(L, seq_cids, l1_status, l2_dec, layer2_stats, suspicion_levels, ref_alpha, gt)
        self._table4_drift(L, seq_cids, l2_dec, layer2_drift_info, layer2_status, gt)
        self._table5_filter(L, seq_cids, layer2_status, confidence_scores, reputations, filter_stats, mal_set, gt)
        self._summary(L, round_num, mode, H, seq_cids, mal_set, l1_status,
                      layer2_status, layer2_drift_info, filter_stats, final_rejected, accuracy)

        output = "\n".join(L)
        print(output)

        # ── optional file save ──
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(output)
                f.write("\n")

        return output

    # =========================================================================
    # HEADER
    # =========================================================================

    def _header(self, L, round_num, mode, H, seq_cids, mal_set):
        n_mal = len(mal_set)
        n_ben = len(seq_cids) - n_mal
        L.append("=" * self.WIDTH)
        L.append(
            f"  ROUND {round_num}  \u00b7  Mode: {mode}  \u00b7  H: {H:.3f}  \u00b7  "
            f"Sampled: {len(seq_cids)} clients  \u00b7  "
            f"Ground Truth: {n_mal} MAL, {n_ben} BEN"
        )
        L.append("=" * self.WIDTH)

    # =========================================================================
    # TABLE 1 — L1 MAGNITUDE
    # =========================================================================

    def _table1_magnitude(self, L, seq_cids, mag, l1_status, gt):
        W = self.WIDTH - 2   # inner box width
        median  = mag.get("median_norm", 0.0)
        mad     = mag.get("mad", 0.0)
        t_flag  = mag.get("threshold_flag", 0.0)
        t_rej   = mag.get("threshold_reject", 0.0)
        k_flag  = mag.get("k_flag", 4.0)
        k_rej   = mag.get("k_reject", 15.0)
        norms   = mag.get("norms", {})

        L.append("")
        L.append(f"  \u250c\u2500 Layer 1 \u2014 Magnitude Filter {'\u2500' * (W - 33)}\u2510")
        L.append(f"  \u2502  Ph\u1ea5t hi\u1ec7n d\u1ee5a tr\u00ean: ||g_i|| vs Median + k\u00d7MAD c\u1ee7a sample{' ' * (W - 62)}\u2502")
        L.append(f"  \u2502{' ' * W}\u2502")
        L.append(f"  \u2502    Median Norm :  {median:.4f}{' ' * (W - 25)}\u2502")
        L.append(f"  \u2502    MAD          :    {mad:.4f}{' ' * (W - 28)}\u2502")

        flag_line = f"    FLAG  thresh  (k={k_flag})  : {t_flag:.4f}   \u2190 norm > n\u00e0y \u2192 FLAGGED"
        L.append(f"  \u2502{flag_line}{' ' * (W - len(flag_line))}\u2502")
        rej_line  = f"    REJECT thresh (k={k_rej}) : {t_rej:.4f}   \u2190 norm > n\u00e0y \u2192 REJECTED"
        L.append(f"  \u2502{rej_line}{' ' * (W - len(rej_line))}\u2502")
        L.append(f"  \u2514{'\u2500' * W}\u2518")
        L.append("")

        # Sort: REJECTED first, then FLAGGED, then ACCEPTED; within each: norm desc
        order = {"REJECTED": 0, "FLAGGED": 1, "ACCEPTED": 2}
        sorted_cids = sorted(seq_cids,
            key=lambda c: (order.get(l1_status.get(c, "ACCEPTED"), 2), -norms.get(c, 0.0)))

        L.append(f"    ID |       Norm |  vs Median | Status    | Reason                                           | GT")
        L.append("  " + "\u2500" * 106)

        for c in sorted_cids:
            n  = norms.get(c, 0.0)
            st = l1_status.get(c, "ACCEPTED")
            vs = f"{n - median:+.4f}"
            if   st == "REJECTED": rsn = f"norm > REJECT thresh ({t_rej})"
            elif st == "FLAGGED":  rsn = f"norm > FLAG thresh ({t_flag})"
            else:                  rsn = "norm \u2264 FLAG thresh \u2192 pass"
            L.append(f"  {c:>4} |  {n:>8.4f} | {vs:>10} | {st:<9} | {rsn:<48} | {gt(c)}")

    # =========================================================================
    # TABLE 2 — L1 DBSCAN
    # =========================================================================

    def _table2_dbscan(self, L, seq_cids, db, l1_status, mal_set, gt):
        W       = self.WIDTH - 2
        pca_act = db.get("pca_dims_actual", 0)
        pca_tgt = db.get("pca_dims_target", 20)
        eps     = db.get("eps", 0.0)
        med_d   = db.get("median_dist", 0.0)
        minPts  = db.get("min_samples", 3)
        n_clust = db.get("cluster_count", 0)
        n_outli = db.get("outlier_count", 0)
        labels  = db.get("labels", {})
        n_total = len(seq_cids)

        L.append("")
        L.append("")
        L.append(f"  \u250c\u2500 Layer 1 \u2014 DBSCAN Clustering {'\u2500' * (W - 34)}\u2510")
        L.append(f"  \u2502  Gi\u1ea3m chi\u1ec1u PCA \u2192 DBSCAN t\u00ecm outliers. Outlier (label=-1) \u2192 FLAGGED{' ' * (W - 72)}\u2502")
        L.append(f"  \u2502{' ' * W}\u2502")

        pca_line = f"    PCA dims : {pca_act} (target={pca_tgt}, actual = min({pca_tgt}, floor(0.5\u00d7{n_total})))"
        L.append(f"  \u2502{pca_line}{' ' * (W - len(pca_line))}\u2502")
        eps_line = f"    eps      : {eps:.4f}  (= multiplier \u00d7 median_dist={med_d:.4f})"
        L.append(f"  \u2502{eps_line}{' ' * (W - len(eps_line))}\u2502")
        pts_line = f"    minPts   : {minPts}"
        L.append(f"  \u2502{pts_line}{' ' * (W - len(pts_line))}\u2502")

        if n_outli == 0:
            cl_line = f"    Clusters : {n_clust}  \u00b7  Cluster 0 = nh\u00f3m benign  \u00b7  Cluster 1 = nh\u00f3m malicious"
            L.append(f"  \u2502{cl_line}{' ' * (W - len(cl_line))}\u2502")
            ol_line = f"    Outliers : 0  \u2192 kh\u00f4ng c\u00f3 thay \u0111\u1ed5i status t\u1eeb b\u01b0\u1edbc magnitude"
            L.append(f"  \u2502{ol_line}{' ' * (W - len(ol_line))}\u2502")
        else:
            cl_line = f"    Clusters : {n_clust}  \u00b7  Outliers : {n_outli}"
            L.append(f"  \u2502{cl_line}{' ' * (W - len(cl_line))}\u2502")
        L.append(f"  \u2514{'\u2500' * W}\u2518")
        L.append("")

        # Sort: FLAGGED first, then within each group: MAL before BEN, then by ID
        order = {"REJECTED": 0, "FLAGGED": 1, "ACCEPTED": 2}
        sorted_cids = sorted(seq_cids, key=lambda c: (
            order.get(l1_status.get(c, "ACCEPTED"), 2),
            0 if c in mal_set else 1,
            c
        ))

        L.append(f"    ID | Cluster | Outlier | L1 Status | Note                                                     | GT")
        L.append("  " + "\u2500" * 113)

        for c in sorted_cids:
            cl  = labels.get(c, 0)
            st  = l1_status.get(c, "ACCEPTED")
            is_out = (cl == -1)
            cl_s = "outlier" if is_out else str(cl)
            ol_s = "Yes" if is_out else "No"

            if is_out:
                note = "Outlier \u2192 n\u00e2ng l\u00ean FLAGGED"
            elif st == "FLAGGED":
                note = f"Cluster {cl} \u00b7 flagged t\u1eeb magnitude, kh\u00f4ng ph\u1ea3i outlier \u2192 gi\u1eef FLAGGED"
            else:
                note = f"Cluster {cl}"

            L.append(f"  {c:>4} | {cl_s:>7} | {ol_s:>7} | {st:<9} | {note:<56} | {gt(c)}")

    # =========================================================================
    # TABLE 3 — L2 DISTANCE + COSINE
    # =========================================================================

    def _table3_dist_cos(self, L, seq_cids, l1_status, l2_dec, layer2_stats, suspicion_levels, ref_alpha, gt):
        W           = self.WIDTH - 2
        dist_thresh = layer2_stats.get("distance_threshold", 0.0)
        cos_thresh  = layer2_stats.get("cosine_threshold", 0.3)
        rescue_thresh = layer2_stats.get("rescue_cosine_threshold", 0.8)

        L.append("")
        L.append("")
        L.append(f"  \u250c\u2500 Layer 2 \u2014 Distance + Cosine Analysis {'\u2500' * (W - 43)}\u2510")
        L.append(f"  \u2502  So s\u00e1nh gradient vs reference (AdaptiveRef blend). Quy\u1ebft \u0111\u1ecfnh theo decision matrix.{' ' * (W - 88)}\u2502")
        L.append(f"  \u2502{' ' * W}\u2502")

        alpha_line = f"AdaptiveRef \u03b1={ref_alpha:.3f} (Historical {ref_alpha*100:.1f}% + Current {(1-ref_alpha)*100:.1f}%)"
        ref_row    = f"    Reference        : {alpha_line}"
        L.append(f"  \u2502{ref_row}{' ' * (W - len(ref_row))}\u2502")
        dt_row     = f"    Distance Thresh  : {dist_thresh:<82.1f}"
        L.append(f"  \u2502{dt_row}\u2502")
        cr_row     = f"    Cos Reject       :   {cos_thresh}   \u2190 cos \u2264 n\u00e0y \u2192 REJECTED (h\u01b0\u1edbng sai)"
        L.append(f"  \u2502{cr_row}{' ' * (W - len(cr_row))}\u2502")
        rs_row     = f"    Cos Rescue       :   {rescue_thresh}   \u2190 cos \u2265 n\u00e0y + dist OK \u2192 RESCUE L1-FLAGGED"
        L.append(f"  \u2502{rs_row}{' ' * (W - len(rs_row))}\u2502")
        L.append(f"  \u2502{' ' * W}\u2502")
        L.append(f"  \u2502  Decision Matrix:{' ' * (W - 18)}\u2502")
        dm_lines = [
            f"    L1=REJECTED                            \u2192 REJECTED (carry forward)",
            f"    cos \u2264 0.3                              \u2192 REJECTED (direction fail)",
            f"    L1=FLAGGED + cos \u2265 {rescue_thresh} + dist OK      \u2192 ACCEPTED (suspicious) \u2190 RESCUED",
            f"    L1=FLAGGED + otherwise                 \u2192 REJECTED (not rescued)",
            f"    L1=ACCEPTED + cos OK + dist fail       \u2192 ACCEPTED (suspicious) *",
            f"    L1=ACCEPTED + cos OK + dist OK         \u2192 ACCEPTED (clean)",
            f"    * Tr\u1eeb: dist > 2\u00d7thresh AND cos < 0.85  \u2192 REJECTED (extreme)",
        ]
        for dm in dm_lines:
            L.append(f"  \u2502{dm}{' ' * (W - len(dm))}\u2502")
        L.append(f"  \u2514{'\u2500' * W}\u2518")
        L.append("")

        # Sort: RESCUED first, then suspicious (MAL first), then clean
        def _sort(c):
            d  = l2_dec.get(c, {})
            l1 = l1_status.get(c, "ACCEPTED")
            fin = d.get("final", "ACCEPTED")
            sus = d.get("suspicion")
            if l1 == "FLAGGED" and fin == "ACCEPTED":         return (0, 0 if gt(c).startswith("MAL") else 1, c)
            if fin == "ACCEPTED" and sus == "suspicious":     return (1, 0 if gt(c).startswith("MAL") else 1, c)
            if fin == "ACCEPTED":                             return (2, 0 if gt(c).startswith("MAL") else 1, c)
            return (3, 0 if gt(c).startswith("MAL") else 1, c)

        sorted_cids = sorted(seq_cids, key=_sort)

        L.append(f"    ID |       L1 |    Dist |   Cos |  D>T | Decision  | Suspicion  | Reason                                               | GT")
        L.append("  " + "\u2500" * 132)

        for c in sorted_cids:
            d    = l2_dec.get(c, {})
            l1   = l1_status.get(c, "ACCEPTED")
            dist = d.get("distance", 0.0)
            cos  = d.get("cosine", 0.0)
            dec  = d.get("final", "ACCEPTED")
            sus  = d.get("suspicion") or "\u2014"
            d_flag = "\u2717" if d.get("fail_distance", False) else "\u2713"

            # Reason text
            if l1 == "REJECTED":
                rsn = "L1 REJECTED \u2192 carry forward"
            elif d.get("fail_cosine", False):
                rsn = f"cos {cos:.3f} \u2264 {cos_thresh} \u2192 direction fail"
            elif l1 == "FLAGGED" and dec == "ACCEPTED":
                rsn = f"RESCUED: cos {cos:.3f} \u2265 {rescue_thresh}, dist {dist:.2f} \u2264 {dist_thresh}"
            elif l1 == "FLAGGED" and dec == "REJECTED":
                rsn = "FLAGGED not rescued"
            elif dec == "ACCEPTED" and sus == "suspicious":
                rsn = f"dist {dist:.2f} > {dist_thresh:.1f} but cos {cos:.3f} OK, not extreme \u2192 suspicious"
            elif dec == "REJECTED":
                rsn = f"extreme dist {dist:.2f} + marginal cos {cos:.3f}"
            else:
                rsn = f"dist {dist:.2f} \u2264 {dist_thresh:.1f}, cos {cos:.3f} OK \u2192 clean"

            L.append(f"  {c:>4} | {l1:<8} | {dist:>6.2f} | {cos:.3f} | {d_flag:>4} | {dec:<9} | {sus:<10} | {rsn:<52} | {gt(c)}")

    # =========================================================================
    # TABLE 4 — L2 DRIFT DETECTION
    # =========================================================================

    def _table4_drift(self, L, seq_cids, l2_dec, drift_info, layer2_status, gt):
        W          = self.WIDTH - 2
        drift_cids = set(drift_info.keys())
        n_drift    = len(drift_cids)
        drift_window = max((len(v.get("trend", [])) for v in drift_info.values()), default=5)

        # Drift client summary for header
        drift_mal = sorted(c for c in drift_cids if gt(c).startswith("MAL"))
        drift_ben = sorted(c for c in drift_cids if gt(c).startswith("BEN"))
        drift_summary = ""
        if drift_mal or drift_ben:
            parts = []
            if drift_mal: parts.append(f"{{{', '.join(str(c) for c in drift_mal)}}} MAL")
            if drift_ben: parts.append(f"{{{', '.join(str(c) for c in drift_ben)}}} BEN")
            drift_summary = f"  \u2192  {' + '.join(parts)}"

        L.append("")
        L.append("")
        L.append(f"  \u250c\u2500 Layer 2 \u2014 Drift Detection (Slow Poison Detector) {'\u2500' * (W - 56)}\u2510")
        L.append(f"  \u2502  Theo d\u00f5i xu h\u01b0\u1edbng cosine similarity qua nhi\u1ec1u round. Trend gi\u1ea3m li\u00ean t\u1ee5c = drift \u2192 REJECTED{' ' * (W - 97)}\u2502")
        L.append(f"  \u2502  Override: N\u1ebfu client drifting \u2192 L2 Final = REJECTED (b\u1ea5t k\u1ec3 L2 dist+cos k\u1ebft qu\u1ea3 g\u00ec){' ' * (W - 92)}\u2502")
        L.append(f"  \u2502{' ' * W}\u2502")
        dw_line = f"    Drift window    : {drift_window} rounds (c\u1ea7n {drift_window} round li\u00ean t\u1ee5c decreasing)"
        L.append(f"  \u2502{dw_line}{' ' * (W - len(dw_line))}\u2502")
        dc_line = f"    Clients drifting: {n_drift}{drift_summary}"
        L.append(f"  \u2502{dc_line}{' ' * (W - len(dc_line))}\u2502")
        L.append(f"  \u2514{'\u2500' * W}\u2518")
        L.append("")

        # Sort: drifting first (MAL before BEN), then non-drifting (by ID)
        sorted_cids = sorted(seq_cids, key=lambda c: (
            0 if c in drift_cids else 1,
            0 if gt(c).startswith("MAL") else 1,
            c
        ))

        L.append(f"    ID | Drift | Cosine Trend (last {drift_window} rounds)      |     L2 dc | L2 Final  | Reason                             | GT")
        L.append("  " + "\u2500" * 130)

        for c in sorted_cids:
            l2_dc  = l2_dec.get(c, {}).get("final", "ACCEPTED")
            l2_fin = layer2_status.get(c, l2_dc)

            if c in drift_cids:
                trend = drift_info[c].get("trend", [])
                trend_s = " \u2192 ".join(f"{v:.3f}" for v in trend)
                L.append(f"  {c:>4} |   Yes | {trend_s:<40} | {l2_dc:>9} | {'REJECTED':<9} | {'drift override \u2192 REJECTED':<34} | {gt(c)}")
            else:
                L.append(f"  {c:>4} |    No | {'\u2014':<40} | {l2_dc:>9} | {l2_fin:<9} | {'no drift':<34} | {gt(c)}")

    # =========================================================================
    # TABLE 5 — TWO-STAGE FILTER
    # =========================================================================

    def _table5_filter(self, L, seq_cids, layer2_status, confidence_scores, reputations, filter_stats, mal_set, gt):
        W        = self.WIDTH - 2
        mode     = filter_stats.get("mode", "?")
        H        = filter_stats.get("H", 0.0)
        th_hard  = filter_stats.get("theta_hard", 0.0)
        th_soft  = filter_stats.get("theta_soft", 0.0)
        tau      = filter_stats.get("tau_mode", 0.0)
        hard_ids = set(filter_stats.get("hard_filtered_ids", []))
        soft_ids = set(filter_stats.get("soft_filtered_ids", []))
        tier2_t  = th_soft * 0.5

        L.append("")
        L.append("")
        L.append(f"  \u250c\u2500 Two-Stage Filter {'\u2500' * (W - 22)}\u2510")
        L.append(f"  \u2502  L\u1ecfc cu\u1ed1i d\u1ee5a tr\u00ean ci (confidence) v\u00e0 R (reputation). Ch\u1ec9 x\u1eed l\u00fd L2-ACCEPTED clients.{' ' * (W - 88)}\u2502")
        L.append(f"  \u2502{' ' * W}\u2502")
        m_line  = f"    Mode    : {mode}  \u00b7  H = {H:.3f}"
        L.append(f"  \u2502{m_line}{' ' * (W - len(m_line))}\u2502")
        h_line  = f"    \u03b8_hard  : {th_hard:.3f}  \u2190 ci > n\u00e0y \u2192 Hard reject"
        L.append(f"  \u2502{h_line}{' ' * (W - len(h_line))}\u2502")
        s_line  = f"    \u03b8_soft  : {th_soft:.3f}  \u2190 Tier 1: ci > n\u00e0y \u2192 Soft reject"
        L.append(f"  \u2502{s_line}{' ' * (W - len(s_line))}\u2502")
        t_line  = f"    \u03c4_mode  : {tau:.1f}   \u2190 reputation threshold ({mode} mode)"
        L.append(f"  \u2502{t_line}{' ' * (W - len(t_line))}\u2502")
        t2_line = f"    Tier 2  : ci > \u03b8_soft/2 ({tier2_t:.3f}) AND R < \u03c4 ({tau}) \u2192 Soft reject"
        L.append(f"  \u2502{t2_line}{' ' * (W - len(t2_line))}\u2502")
        L.append(f"  \u2502{' ' * W}\u2502")
        n_line  = f"  Note: \u03b8_adj = clip(\u03b8base + (0.5 - H) \u00d7 factor, 0.5, 0.9)"
        L.append(f"  \u2502{n_line}{' ' * (W - len(n_line))}\u2502")
        L.append(f"  \u2514{'\u2500' * W}\u2518")
        L.append("")

        # Sort: L2-rejected (MAL first), then TS-rejected, then TS-accepted
        def _sort(c):
            if layer2_status.get(c) == "REJECTED": return (0, 0 if c in mal_set else 1, c)
            if c in hard_ids or c in soft_ids:     return (1, 0 if c in mal_set else 1, c)
            return (2, 0 if c in mal_set else 1, c)

        sorted_cids = sorted(seq_cids, key=_sort)

        L.append(f"    ID |       ci |         R |        TS | Reason                                                   | GT")
        L.append("  " + "\u2500" * 112)

        for c in sorted_cids:
            ci = confidence_scores.get(c, 0.0)
            R  = reputations.get(c, 0.0)

            if layer2_status.get(c) == "REJECTED":
                ts_s = "\u2014"
                rsn  = "skipped (L2 rejected)"
            elif c in hard_ids:
                ts_s = "REJECTED"
                rsn  = f"ci {ci:.3f} > \u03b8_hard {th_hard}"
            elif c in soft_ids:
                ts_s = "REJECTED"
                if ci > th_soft:
                    rsn = f"Tier 1: ci {ci:.3f} > \u03b8_soft {th_soft}"
                else:
                    rsn = f"Tier 2: ci {ci:.3f} > {tier2_t:.3f} AND R {R:.3f} < \u03c4 {tau}"
            else:
                ts_s = "ACCEPTED"
                rsn  = "ci OK, R OK"

            L.append(f"  {c:>4} |  {ci:>5.3f} |  {R:>5.3f} | {ts_s:>9} | {rsn:<56} | {gt(c)}")

    # =========================================================================
    # OPTION C SUMMARY BAR
    # =========================================================================

    def _summary(self, L, round_num, mode, H, seq_cids, mal_set, l1_status,
                 layer2_status, drift_info, filter_stats, final_rejected, accuracy):

        n_mal = len(mal_set)
        n_ben = len(seq_cids) - n_mal

        # L1
        l1_rej  = sum(1 for c in seq_cids if l1_status.get(c) == "REJECTED")
        l1_flag = sum(1 for c in seq_cids if l1_status.get(c) == "FLAGGED")

        # L2: drift / rescued / confirmed
        drift_n = len(drift_info)
        res_n   = sum(1 for c in seq_cids
                      if l1_status.get(c) == "FLAGGED" and layer2_status.get(c) == "ACCEPTED")
        con_n   = sum(1 for c in seq_cids
                      if l1_status.get(c) == "FLAGGED"
                      and layer2_status.get(c) == "REJECTED"
                      and c not in drift_info)

        # Filter
        hard_n = filter_stats.get("hard_filtered_count", 0)
        soft_n = filter_stats.get("soft_filtered_count", 0)

        # DR / FPR
        tp = sum(1 for c in mal_set if c in final_rejected)
        fp = sum(1 for c in seq_cids if c not in mal_set and c in final_rejected)
        dr_pct  = (tp / n_mal * 100) if n_mal > 0 else 0.0
        fpr_pct = (fp / n_ben * 100) if n_ben > 0 else 0.0

        acc_s = f" | Acc={accuracy:.4f}" if accuracy is not None else ""

        bar = (
            f"  \u25c6 Rnd {round_num} | {mode} | H={H:.3f}"
            f" | L1[{l1_rej}|{l1_flag}]"
            f" | L2[drift={drift_n}|res={res_n}|con={con_n}]"
            f" | Filt[{hard_n}|{soft_n}]"
            f" | DR {tp}/{n_mal} ({dr_pct:.1f}%)"
            f" | FPR {fp}/{n_ben} ({fpr_pct:.1f}%)"
            f"{acc_s}"
        )

        L.append("")
        L.append("")
        L.append("  " + "\u2500" * 106)
        L.append(bar)
        L.append("  " + "\u2500" * 106)