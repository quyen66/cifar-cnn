"""
Two-Stage Filtering (V2 - main.pdf) - REFACTORED
==================================================
Giai đoạn 3: Lọc hai giai đoạn

Pipeline:
1. Nhận confidence scores (ci) từ giai đoạn tính điểm bất thường
2. Lọc cứng: ci > θ_adj(θbase = 0.85)
3. Lọc mềm: ci > θ_adj(θbase = 0.65) AND R < τ_mode

REFACTORED: 
- Bỏ _compute_adaptive_threshold() (đã có trong NonIIDHandler)
- Nhận noniid_handler để gọi compute_adaptive_threshold()

τ_mode (reputation threshold theo mode):
    - NORMAL: τ = 0.2 (dễ tính, chỉ filter R < 0.2)
    - ALERT: τ = 0.4 (cảnh giác)
    - DEFENSE: τ = 0.6 (khắt khe)

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
    tau_mode: float
    mode: str
    H: float


class TwoStageFilter:
    """
    Two-Stage Filter theo main.pdf (REFACTORED).
    
    Lọc cứng: Loại bỏ ngay các máy khách có điểm bất thường quá cao
    Lọc mềm: Loại bỏ các máy khách có điểm nghi ngờ trung bình VÀ uy tín thấp
    
    NOTE: θ_adj được tính bởi NonIIDHandler, không tự tính ở đây.
    """
    
    def __init__(
        self,
        hard_threshold_base: float = 0.85,
        soft_threshold_base: float = 0.65,
        soft_rep_threshold_normal: float = 0.2,
        soft_rep_threshold_alert: float = 0.4,
        soft_rep_threshold_defense: float = 0.6
    ):
        """
        Initialize Two-Stage Filter.
        
        Args:
            hard_threshold_base: θbase cho lọc cứng (0.85)
            soft_threshold_base: θbase cho lọc mềm (0.65)
            soft_rep_threshold_normal: τ_mode cho NORMAL (0.2)
            soft_rep_threshold_alert: τ_mode cho ALERT (0.4)
            soft_rep_threshold_defense: τ_mode cho DEFENSE (0.6)
            
        NOTE: adjustment_factor, theta_adj_clip_min/max đã được chuyển sang NonIIDHandler
        """
        self.hard_threshold_base = hard_threshold_base
        self.soft_threshold_base = soft_threshold_base
        self.soft_rep_threshold_normal = soft_rep_threshold_normal
        self.soft_rep_threshold_alert = soft_rep_threshold_alert
        self.soft_rep_threshold_defense = soft_rep_threshold_defense
        
        # Stats
        self.last_stats: Optional[FilterStats] = None
        
        print(f"✅ TwoStageFilter V2 initialized (REFACTORED):")
        print(f"   Hard filter: θbase = {hard_threshold_base}")
        print(f"   Soft filter: θbase = {soft_threshold_base}")
        print(f"   τ_mode: NORMAL={soft_rep_threshold_normal}, "
              f"ALERT={soft_rep_threshold_alert}, DEFENSE={soft_rep_threshold_defense}")
        print(f"   NOTE: θ_adj computed by NonIIDHandler")

    def _get_tau_mode(self, mode: str) -> float:
        """
        Lấy ngưỡng reputation τ_mode theo chế độ hiện tại.
        
        Logic (PDF trang 13):
        - NORMAL (an toàn): τ = 0.2 → chỉ filter R < 0.2 (rất thấp)
        - ALERT (cảnh giác): τ = 0.4 → filter R < 0.4
        - DEFENSE (khắt khe): τ = 0.6 → filter R < 0.6 (khắt khe)
        
        Args:
            mode: Chế độ hiện tại
        
        Returns:
            τ_mode
        """
        if mode == "NORMAL":
            return self.soft_rep_threshold_normal
        elif mode == "ALERT":
            return self.soft_rep_threshold_alert
        else:  # DEFENSE
            return self.soft_rep_threshold_defense

    def filter_clients(
        self,
        client_ids: List[int],
        confidence_scores: Dict[int, float],
        reputations: Dict[int, float],
        mode: str,
        H: float,
        noniid_handler: "NonIIDHandler"
    ) -> Tuple[Set[int], Set[int], Dict]:
        """
        Lọc máy khách qua hai giai đoạn.
        
        Pipeline:
        1. Gọi noniid_handler.compute_adaptive_threshold() để tính θ_adj
        2. Lọc cứng: ci > θ_adj_hard → loại ngay
        3. Lọc mềm (trên remaining): ci > θ_adj_soft AND R < τ_mode → loại
        
        Args:
            client_ids: List client IDs (đã được L1+L2 ACCEPTED)
            confidence_scores: Dict[cid, ci] - điểm bất thường
            reputations: Dict[cid, R] - điểm danh tiếng
            mode: Chế độ hiện tại (NORMAL/ALERT/DEFENSE)
            H: Heterogeneity score
            noniid_handler: NonIIDHandler instance để tính θ_adj
        
        Returns:
            trusted: Set clients vượt qua cả 2 lọc
            filtered: Set clients bị loại
            stats: Dict thống kê
        """
        # =========================================================
        # STEP 1: Tính các ngưỡng thích ứng (DÙNG NONIID HANDLER)
        # =========================================================
        theta_hard = noniid_handler.compute_adaptive_threshold(H, self.hard_threshold_base)
        theta_soft = noniid_handler.compute_adaptive_threshold(H, self.soft_threshold_base)
        tau_mode = self._get_tau_mode(mode)
        
        print(f"\n📊 Two-Stage Filtering:")
        print(f"   Mode: {mode}, H: {H:.3f}")
        print(f"   θ_hard = {theta_hard:.3f} (base={self.hard_threshold_base})")
        print(f"   θ_soft = {theta_soft:.3f} (base={self.soft_threshold_base})")
        print(f"   τ_mode = {tau_mode:.2f}")
        
        # =========================================================
        # STEP 2: Lọc cứng - ci > θ_hard
        # =========================================================
        hard_filtered = set()
        for cid in client_ids:
            ci = confidence_scores.get(cid, 0.0)
            if ci > theta_hard:
                hard_filtered.add(cid)
        
        print(f"\n   🔴 Hard Filter (ci > {theta_hard:.3f}):")
        print(f"      Filtered: {len(hard_filtered)} clients")
        if hard_filtered and len(hard_filtered) <= 10:
            for cid in hard_filtered:
                ci = confidence_scores.get(cid, 0)
                print(f"         Client {cid}: ci={ci:.3f}")
        
        # =========================================================
        # STEP 3: Lọc mềm - ci > θ_soft AND R < τ_mode
        # =========================================================
        remaining = set(client_ids) - hard_filtered
        soft_filtered = set()
        
        for cid in remaining:
            ci = confidence_scores.get(cid, 0.0)
            R = reputations.get(cid, 0.8)  # Default R = 0.8 nếu chưa có
            if ci > theta_soft:
                soft_filtered.add(cid)
                continue
            # Điều kiện lọc mềm: PHẢI thỏa cả 2
            # 1. ci > θ_soft (có điểm nghi ngờ)
            # 2. R < τ_mode (uy tín thấp theo mode)
            if ci > theta_soft * 0.5 and R < tau_mode:
                soft_filtered.add(cid)
        
        # print(f"\n   🟡 Soft Filter (ci > {theta_soft:.3f} AND R < {tau_mode:.2f}):")
        # print(f"      Filtered: {len(soft_filtered)} clients")
        # if soft_filtered and len(soft_filtered) <= 10:
        #     for cid in soft_filtered:
        #         ci = confidence_scores.get(cid, 0)
        #         R = reputations.get(cid, 0.8)
        #         print(f"         Client {cid}: ci={ci:.3f}, R={R:.3f}")
        print(f"\n   🟡 Soft Filter (TIERED - ci>{theta_soft:.3f} OR [R<{tau_mode:.2f} AND ci>{theta_soft*0.5:.3f}]):")
        print(f"      Filtered: {len(soft_filtered)} clients")
        if soft_filtered and len(soft_filtered) <= 10:
            for cid in soft_filtered:
                ci = confidence_scores.get(cid, 0)
                R = reputations.get(cid, 0.8)
                # Determine which tier triggered
                tier = "Tier 1 (ci)" if ci > theta_soft else "Tier 2 (R+ci)"
                print(f"         Client {cid}: ci={ci:.3f}, R={R:.3f} [{tier}]")
 
        # =========================================================
        # STEP 4: Tập tin cậy cuối cùng
        # =========================================================
        all_filtered = hard_filtered | soft_filtered
        trusted = set(client_ids) - all_filtered
        
        print(f"\n   ✅ Trusted Clients: {len(trusted)}")
        
        # Store stats
        self.last_stats = FilterStats(
            hard_filtered_count=len(hard_filtered),
            soft_filtered_count=len(soft_filtered),
            trusted_count=len(trusted),
            theta_hard=theta_hard,
            theta_soft=theta_soft,
            tau_mode=tau_mode,
            mode=mode,
            H=H
        )
        
        # Return stats dict for compatibility
        stats = {
            'hard_filtered_count': len(hard_filtered),
            'soft_filtered_count': len(soft_filtered),
            'trusted_count': len(trusted),
            'theta_hard': theta_hard,
            'theta_soft': theta_soft,
            'tau_mode': tau_mode,
            'mode': mode,
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
            'tau_mode': self.last_stats.tau_mode,
            'mode': self.last_stats.mode,
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
        soft_threshold_base=0.65,
        soft_rep_threshold_normal=0.2,
        soft_rep_threshold_alert=0.4,
        soft_rep_threshold_defense=0.6
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
    # TEST 2: τ_mode by mode (unchanged)
    # =========================================================
    print("\n" + "-"*70)
    print("TEST 2: τ_mode by Mode")
    print("-"*70)
    
    tau_tests = [
        ("NORMAL", 0.2),
        ("ALERT", 0.4),
        ("DEFENSE", 0.6),
    ]
    
    for mode, expected in tau_tests:
        actual = filter_obj._get_tau_mode(mode)
        status = "✅" if actual == expected else "❌"
        print(f"   {status} {mode}: τ_mode={actual:.2f} (expected: {expected:.2f})")
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
        18: 0.15,  # R < τ → soft filter
        19: 0.35,  # depends on mode
    }
    
    # Test với H = 0.6, mode = ALERT
    H = 0.6
    mode = "ALERT"
    
    # θ_hard = clip(0.85 + (0.6-0.5)*0.4, 0.5, 0.9) = clip(0.89, 0.5, 0.9) = 0.89
    # θ_soft = clip(0.65 + (0.6-0.5)*0.4, 0.5, 0.9) = clip(0.69, 0.5, 0.9) = 0.69
    expected_theta_hard = 0.89
    expected_theta_soft = 0.69
    
    trusted, filtered, stats = filter_obj.filter_clients(
        client_ids=client_ids,
        confidence_scores=confidence_scores,
        reputations=reputations,
        mode=mode,
        H=H,
        noniid_handler=noniid_handler  # PASSING NONIID HANDLER
    )
    
    print(f"\n   Verification:")
    print(f"      θ_hard = {stats['theta_hard']:.3f} (expected: {expected_theta_hard:.2f})")
    print(f"      θ_soft = {stats['theta_soft']:.3f} (expected: {expected_theta_soft:.2f})")
    
    theta_hard_correct = abs(stats['theta_hard'] - expected_theta_hard) < 0.01
    theta_soft_correct = abs(stats['theta_soft'] - expected_theta_soft) < 0.01
    
    print(f"      θ_hard correct: {'✅' if theta_hard_correct else '❌'}")
    print(f"      θ_soft correct: {'✅' if theta_soft_correct else '❌'}")
    
    if not (theta_hard_correct and theta_soft_correct):
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
            mode="NORMAL",
            H=H,
            noniid_handler=noniid_handler
        )
        
        expected_hard = noniid_handler.compute_adaptive_threshold(H, 0.85)
        expected_soft = noniid_handler.compute_adaptive_threshold(H, 0.65)
        
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
        mode="NORMAL",
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
        mode="DEFENSE",
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