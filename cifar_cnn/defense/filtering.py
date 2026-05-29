"""
Two-Stage Filtering (V2 - main.pdf) - REFACTORED
==================================================
Giai đoạn 3: Lọc hai giai đoạn

Pipeline:
1. Nhận confidence scores (ci) từ giai đoạn tính điểm bất thường
2. Lọc cứng: ci > θ_adj(θbase = 0.85)
3. Lọc mềm: ci > θ_adj(θbase = 0.06) AND R < R_floor (mode-independent)

REFACTORED:
- Bỏ _compute_adaptive_threshold() (đã có trong NonIIDHandler)
- Nhận noniid_handler để gọi compute_adaptive_threshold()
- Bỏ mode-dependent τ_mode; thay bằng R_floor cố định

Output:
    trusted_clients: Set máy khách vượt qua cả 2 lớp lọc
    filtered_clients: Set máy khách bị loại (hard + soft)
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, TYPE_CHECKING
from dataclasses import dataclass

# Type hint để tránh circular import
if TYPE_CHECKING:
    from .noniid_handler import NonIIDHandler


@dataclass
class FilterStats:
    """Thống kê kết quả lọc."""
    hard_filtered_count: int
    soft_filtered_count: int
    trusted_count: int
    theta_hard: float
    theta_soft: float
    rep_floor: float
    H: float


class TwoStageFilter:
    """
    Two-Stage Filter theo main.pdf (REFACTORED).

    Lọc cứng: Loại bỏ ngay các máy khách có điểm bất thường quá cao
    Lọc mềm: Loại bỏ các máy khách có điểm nghi ngờ trung bình VÀ uy tín thấp
              (mode-independent: dùng R_floor cố định thay vì τ_mode)

    NOTE: θ_adj được tính bởi NonIIDHandler, không tự tính ở đây.
    """

    def __init__(
        self,
        hard_threshold_base: float = 0.85,
        soft_threshold_base: float = 0.06,
        soft_rep_floor: float = 0.30,
    ):
        """
        Initialize Two-Stage Filter.

        Args:
            hard_threshold_base: θbase cho lọc cứng (0.85)
            soft_threshold_base: θbase cho lọc mềm (0.06)
            soft_rep_floor: Ngưỡng reputation floor cho soft filter (0.30)

        NOTE: adjustment_factor, theta_adj_clip_min/max đã được chuyển sang NonIIDHandler
        """
        self.hard_threshold_base = hard_threshold_base
        self.soft_threshold_base = soft_threshold_base
        self.soft_rep_floor = soft_rep_floor

        # Stats
        self.last_stats: Optional[FilterStats] = None

        print(f"✅ TwoStageFilter V2 initialized (REFACTORED):")
        print(f"   Hard filter: θbase = {hard_threshold_base}")
        print(f"   Soft filter: θbase = {soft_threshold_base}")
        print(f"   R_floor = {soft_rep_floor} (mode-independent)")
        print(f"   NOTE: θ_adj computed by NonIIDHandler")

    def filter_clients(
        self,
        client_ids: List[int],
        confidence_scores: Dict[int, float],
        reputations: Dict[int, float],
        H: float,                            # theta_signal: GDS (Variant B) hoặc H (Variant A)
        noniid_handler: "NonIIDHandler",
        gt_malicious: Optional[Dict[int, bool]] = None  # Ground truth để debug (có thể None)
    ) -> Tuple[Set[int], Set[int], Dict]:
        """
        Lọc máy khách qua hai giai đoạn.

        Pipeline:
        1. Gọi noniid_handler.compute_adaptive_threshold(H) để tính θ_adj
           H ở đây là theta_signal (GDS hoặc H tùy variant)
        2. Lọc cứng: ci > θ_adj_hard → loại ngay
        3. Lọc mềm (trên remaining): ci > θ_adj_soft AND R < R_floor → loại

        Args:
            client_ids:        List client IDs (đã qua L1+L2)
            confidence_scores: Dict[cid, ci]
            reputations:       Dict[cid, R]
            H:                 theta_signal (GDS hoặc H behavioral tùy variant)
            noniid_handler:    NonIIDHandler để tính θ_adj
            gt_malicious:      Optional Dict[cid, bool] — nếu có, in GT cho từng client

        Returns:
            trusted, filtered, stats
        """
        # =========================================================
        # STEP 1: Tính ngưỡng thích ứng
        # =========================================================
        theta_hard = noniid_handler.compute_adaptive_threshold(H, self.hard_threshold_base)
        theta_soft = noniid_handler.compute_adaptive_threshold(H, self.soft_threshold_base)
        rep_floor  = self.soft_rep_floor

        # --- Header ---
        print(f"\n📊 Two-Stage Filtering:")
        print(f"   θ_signal={H:.4f} (used for θ_adj) | R_floor={rep_floor:.2f}")
        print(f"   θ_hard = {theta_hard:.3f}  (base={self.hard_threshold_base}  → ci > {theta_hard:.3f} → filtered)")
        print(f"   θ_soft = {theta_soft:.3f}  (base={self.soft_threshold_base}  → ci > {theta_soft:.3f} AND R < R_floor → filtered)")

        # =========================================================
        # STEP 2: Lọc cứng
        # =========================================================
        hard_filtered = set()
        for cid in client_ids:
            ci = confidence_scores.get(cid, 0.0)
            if ci > theta_hard:
                hard_filtered.add(cid)

        print(f"\n   🔴 Hard Filter (ci > {theta_hard:.3f}): {len(hard_filtered)} clients filtered")
        for cid in sorted(hard_filtered):
            ci  = confidence_scores.get(cid, 0)
            R   = reputations.get(cid, 0.0)
            gt  = "⚠️MAL" if gt_malicious and gt_malicious.get(cid) else "✅BEN" if gt_malicious else "  ?"
            print(f"      [{gt}] Client {cid:3d}: ci={ci:.3f}  R={R:.4f}")

        # =========================================================
        # STEP 3: Lọc mềm
        # =========================================================
        remaining     = set(client_ids) - hard_filtered
        soft_filtered = set()

        for cid in remaining:
            ci = confidence_scores.get(cid, 0.0)
            R  = reputations.get(cid, 0.8)
            if ci > theta_soft and R < rep_floor:
                soft_filtered.add(cid)

        print(f"\n   🟡 Soft Filter (ci > {theta_soft:.3f} AND R < {rep_floor:.2f}): {len(soft_filtered)} clients filtered")
        for cid in sorted(soft_filtered):
            ci  = confidence_scores.get(cid, 0)
            R   = reputations.get(cid, 0.8)
            gt  = "⚠️MAL" if gt_malicious and gt_malicious.get(cid) else "✅BEN" if gt_malicious else "  ?"
            print(f"      [{gt}] Client {cid:3d}: ci={ci:.3f}  R={R:.4f}")

        # =========================================================
        # STEP 4: Trusted set + Summary table
        # =========================================================
        all_filtered = hard_filtered | soft_filtered
        trusted      = set(client_ids) - all_filtered

        print(f"\n   ✅ Trusted (passed filter): {len(trusted)} | "
              f"🚫 Filtered: {len(all_filtered)} "
              f"(hard={len(hard_filtered)}, soft={len(soft_filtered)})")

        # Summary table — in tất cả clients với ci, R, outcome
        print(f"\n   {'─'*74}")
        print(f"   {'ID':>4} │ {'GT':^6} │ {'ci':>6} │ {'R':>7} │ {'ci>θhard':^8} │ {'ci>θsoft':^8} │ {'R<Rfl':^6} │ Outcome")
        print(f"   {'─'*74}")
        for cid in sorted(client_ids):
            ci        = confidence_scores.get(cid, 0.0)
            R         = reputations.get(cid, 0.0)
            gt_str    = "⚠️MAL" if gt_malicious and gt_malicious.get(cid) else ("✅BEN" if gt_malicious else "   ?")
            ci_h      = "✓" if ci > theta_hard else "✗"
            ci_s      = "✓" if ci > theta_soft else "✗"
            r_lt_fl   = "✓" if R < rep_floor  else "✗"
            if cid in hard_filtered:
                outcome = "HARD-FILTER"
            elif cid in soft_filtered:
                outcome = "SOFT-FILTER"
            else:
                outcome = "trusted"
            print(f"   {cid:>4} │ {gt_str:^6} │ {ci:>6.3f} │ {R:>7.4f} │ {ci_h:^8} │ {ci_s:^8} │ {r_lt_fl:^6} │ {outcome}")
        print(f"   {'─'*74}")

        # Store stats
        self.last_stats = FilterStats(
            hard_filtered_count=len(hard_filtered),
            soft_filtered_count=len(soft_filtered),
            trusted_count=len(trusted),
            theta_hard=theta_hard,
            theta_soft=theta_soft,
            rep_floor=rep_floor,
            H=H
        )

        # Return stats dict for compatibility
        stats = {
            'hard_filtered_count': len(hard_filtered),
            'soft_filtered_count': len(soft_filtered),
            'trusted_count': len(trusted),
            'theta_hard': theta_hard,
            'theta_soft': theta_soft,
            'rep_floor': rep_floor,
            'H': H,
            'hard_filtered_ids': list(hard_filtered),
            'soft_filtered_ids': list(soft_filtered),
            'trusted_ids': list(trusted)
        }

        return trusted, all_filtered, stats

    def get_stats(self) -> Dict:
        """Get last filter stats."""
        if self.last_stats is None:
            return {}
        return {
            'hard_filtered_count': self.last_stats.hard_filtered_count,
            'soft_filtered_count': self.last_stats.soft_filtered_count,
            'trusted_count': self.last_stats.trusted_count,
            'theta_hard': self.last_stats.theta_hard,
            'theta_soft': self.last_stats.theta_soft,
            'rep_floor': self.last_stats.rep_floor,
            'H': self.last_stats.H
        }


# ============================================================
# TEST CODE
# ============================================================

def test_two_stage_filter():
    """Test Two-Stage Filter với NonIIDHandler integration."""
    print("\n" + "="*70)
    print("🧪 TESTING TWO-STAGE FILTER V2 (REFACTORED)")
    print("="*70)

    # Import NonIIDHandler for testing
    # (Trong thực tế, import từ package)
    from noniid_handler import NonIIDHandler

    # Initialize components
    noniid_handler = NonIIDHandler(
        adjustment_factor=0.4,
        theta_adj_clip_min=0.5,
        theta_adj_clip_max=0.9
    )

    filter_obj = TwoStageFilter(
        hard_threshold_base=0.85,
        soft_threshold_base=0.06,
        soft_rep_floor=0.30,
    )

    all_pass = True

    # =========================================================
    # TEST 1: Verify θ_adj comes from NonIIDHandler
    # =========================================================
    print("\n" + "-"*70)
    print("TEST 1: θ_adj from NonIIDHandler")
    print("-"*70)

    test_cases = [
        (0.5, 0.85, 0.85),   # H=0.5 → no change
        (0.7, 0.85, 0.93),   # H=0.7 → increase (0.85 + 0.2*0.4 = 0.93, clip to 0.9)
        (0.3, 0.85, 0.77),   # H=0.3 → decrease (0.85 - 0.2*0.4 = 0.77)
    ]

    for H, base, expected_raw in test_cases:
        expected = np.clip(expected_raw, 0.5, 0.9)
        actual = noniid_handler.compute_adaptive_threshold(H, base)
        passed = abs(actual - expected) < 0.01
        status = "✅" if passed else "❌"
        print(f"   {status} H={H:.1f}, θbase={base:.2f} → θadj={actual:.3f} (expected: {expected:.3f})")
        if not passed:
            all_pass = False

    # =========================================================
    # TEST 2: R_floor is mode-independent
    # =========================================================
    print("\n" + "-"*70)
    print("TEST 2: R_floor (mode-independent)")
    print("-"*70)

    expected_floor = 0.30
    actual_floor = filter_obj.soft_rep_floor
    status = "✅" if actual_floor == expected_floor else "❌"
    print(f"   {status} R_floor={actual_floor:.2f} (expected: {expected_floor:.2f})")
    if status == "❌":
        all_pass = False

    # =========================================================
    # TEST 3: Full filtering with NonIIDHandler
    # =========================================================
    print("\n" + "-"*70)
    print("TEST 3: Full Filtering with NonIIDHandler")
    print("-"*70)

    # Scenario: 20 clients với H = 0.6
    client_ids = list(range(20))

    # Confidence scores (ci)
    confidence_scores = {
        **{i: 0.2 + np.random.uniform(0, 0.2) for i in range(15)},
        15: 0.95,  # Hard filter (> θ_hard)
        16: 0.92,  # Hard filter
        17: 0.91,  # Hard filter
        18: 0.75,  # Soft filter candidate
        19: 0.70,  # Soft filter candidate
    }

    # Reputations
    reputations = {
        **{i: 0.7 + np.random.uniform(0, 0.2) for i in range(15)},
        15: 0.5,
        16: 0.6,
        17: 0.4,
        18: 0.15,  # R < R_floor → soft filter
        19: 0.35,  # R > R_floor → not soft filtered
    }

    # Test với H = 0.6
    H = 0.6

    # θ_hard = clip(0.85 + (0.6-0.5)*0.4, 0.5, 0.9) = clip(0.89, 0.5, 0.9) = 0.89
    # θ_soft = clip(0.06 + (0.6-0.5)*0.4, 0.5, 0.9) = clip(0.10, 0.5, 0.9) = 0.5
    expected_theta_hard = 0.89

    trusted, filtered, stats = filter_obj.filter_clients(
        client_ids=client_ids,
        confidence_scores=confidence_scores,
        reputations=reputations,
        H=H,
        noniid_handler=noniid_handler  # PASSING NONIID HANDLER
    )

    print(f"\n   Verification:")
    print(f"      θ_hard = {stats['theta_hard']:.3f} (expected: {expected_theta_hard:.2f})")

    theta_hard_correct = abs(stats['theta_hard'] - expected_theta_hard) < 0.01
    print(f"      θ_hard correct: {'✅' if theta_hard_correct else '❌'}")

    if not theta_hard_correct:
        all_pass = False

    # Hard filter: ci > 0.89 → clients 15, 16, 17 (0.95, 0.92, 0.91)
    hard_expected = {15, 16, 17}
    hard_actual = set(stats['hard_filtered_ids'])
    hard_correct = hard_actual == hard_expected
    print(f"      Hard filtered: {hard_actual} (expected: {hard_expected}): {'✅' if hard_correct else '❌'}")
    if not hard_correct:
        all_pass = False

    # =========================================================
    # TEST 4: Different H values affect θ_adj
    # =========================================================
    print("\n" + "-"*70)
    print("TEST 4: H affects θ_adj dynamically")
    print("-"*70)

    for H in [0.3, 0.5, 0.8]:
        _, _, stats = filter_obj.filter_clients(
            client_ids=[0, 1, 2],
            confidence_scores={0: 0.5, 1: 0.6, 2: 0.9},
            reputations={0: 0.8, 1: 0.8, 2: 0.8},
            H=H,
            noniid_handler=noniid_handler
        )

        expected_hard = noniid_handler.compute_adaptive_threshold(H, 0.85)
        expected_soft = noniid_handler.compute_adaptive_threshold(H, 0.06)

        match_hard = abs(stats['theta_hard'] - expected_hard) < 0.001
        match_soft = abs(stats['theta_soft'] - expected_soft) < 0.001

        status = "✅" if (match_hard and match_soft) else "❌"
        print(f"   {status} H={H:.1f}: θ_hard={stats['theta_hard']:.3f}, θ_soft={stats['theta_soft']:.3f}")

        if not (match_hard and match_soft):
            all_pass = False

    # =========================================================
    # TEST 5: Edge cases
    # =========================================================
    print("\n" + "-"*70)
    print("TEST 5: Edge Cases")
    print("-"*70)

    # Empty clients
    trusted, filtered, stats = filter_obj.filter_clients(
        client_ids=[],
        confidence_scores={},
        reputations={},
        H=0.5,
        noniid_handler=noniid_handler
    )
    empty_pass = len(trusted) == 0 and len(filtered) == 0
    print(f"   Empty clients: {'✅' if empty_pass else '❌'}")
    if not empty_pass:
        all_pass = False

    # All clean
    trusted, filtered, stats = filter_obj.filter_clients(
        client_ids=list(range(10)),
        confidence_scores={i: 0.1 for i in range(10)},
        reputations={i: 0.9 for i in range(10)},
        H=0.5,
        noniid_handler=noniid_handler
    )
    all_trusted = len(trusted) == 10 and len(filtered) == 0
    print(f"   All clean clients: {'✅' if all_trusted else '❌'}")
    if not all_trusted:
        all_pass = False

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "="*70)
    if all_pass:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️ SOME TESTS FAILED")
    print("="*70 + "\n")

    return all_pass


if __name__ == "__main__":
    test_two_stage_filter()
