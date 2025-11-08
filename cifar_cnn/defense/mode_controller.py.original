"""
Mode Controller - Week 3
=========================
Quản lý 3 modes: NORMAL, ALERT, DEFENSE với Hysteresis và Reputation Gates.

Components:
1. Mode Decision: Quyết định mode dựa trên threat ratio
2. Hysteresis: Chỉ chuyển mode khi ổn định ≥2 rounds
3. Reputation Gates: Emergency mode switch dựa trên reputation anomalies

Author: Week 3 Implementation
"""

import numpy as np
from typing import Dict, List, Optional


class ModeController:
    """
    Controller quản lý 3 modes với Hysteresis và Reputation Gates.
    
    Modes:
    - NORMAL (ρ ≤ 0.15): Low threat → Weighted Average
    - ALERT (0.15 < ρ ≤ 0.30): Medium threat → Trimmed Mean
    - DEFENSE (ρ > 0.30): High threat → Coordinate Median
    
    Philosophy:
    - Hysteresis: Tránh oscillation, chỉ chuyển khi ổn định
    - Reputation Gates: Emergency override khi phát hiện anomaly
    """
    
    def __init__(self,
                 threshold_normal: float = 0.15,
                 threshold_defense: float = 0.30,
                 stability_required: int = 2,
                 gate1_min_high_rep: int = 3,
                 gate1_rep_threshold: float = 0.85,
                 gate2_drop_threshold: float = 0.05):
        """
        Args:
            threshold_normal: Ngưỡng chuyển từ NORMAL→ALERT (default=0.15)
            threshold_defense: Ngưỡng chuyển từ ALERT→DEFENSE (default=0.30)
            stability_required: Số rounds cần ổn định trước khi chuyển (default=2)
            gate1_min_high_rep: Số lượng high-rep clients bị flag để kích hoạt gate 1 (default=3)
            gate1_rep_threshold: Reputation threshold cho gate 1 (default=0.85)
            gate2_drop_threshold: % drop của mean reputation để kích hoạt gate 2 (default=0.05)
        """
        self.threshold_normal = threshold_normal
        self.threshold_defense = threshold_defense
        self.stability_required = stability_required
        
        # Reputation Gates
        self.gate1_min_high_rep = gate1_min_high_rep
        self.gate1_rep_threshold = gate1_rep_threshold
        self.gate2_drop_threshold = gate2_drop_threshold
        
        # State
        self.current_mode = 'NORMAL'
        self.mode_history = []  # History of suggested modes
        self.rep_mean_history = []  # History of mean reputation
        
        print(f"✅ ModeController initialized:")
        print(f"   - Thresholds: NORMAL≤{threshold_normal}, ALERT≤{threshold_defense}")
        print(f"   - Hysteresis: {stability_required} rounds stability")
        print(f"   - Reputation Gates enabled")
    
    def update_mode(self,
                   threat_ratio: float,
                   flagged_clients: List[int],
                   reputations: Dict[int, float],
                   current_round: int) -> str:
        """
        Cập nhật mode dựa trên threat ratio và reputation.
        
        Args:
            threat_ratio: Tỷ lệ đe dọa có trọng số ρ ∈ [0,1]
            flagged_clients: List of client IDs bị đánh dấu
            reputations: Dict mapping client_id -> reputation score
            current_round: Current training round
        
        Returns:
            New mode: 'NORMAL', 'ALERT', or 'DEFENSE'
        """
        # Suggest mode based on threat ratio
        suggested_mode = self._suggest_mode(threat_ratio)
        
        # Add to history
        self.mode_history.append(suggested_mode)
        
        # Calculate mean reputation
        if reputations:
            mean_rep = np.mean(list(reputations.values()))
            self.rep_mean_history.append(mean_rep)
        
        # Check Reputation Gates (emergency override)
        gate_triggered, gate_reason = self._check_reputation_gates(
            flagged_clients, reputations
        )
        
        if gate_triggered:
            print(f"   🚨 Reputation Gate triggered: {gate_reason}")
            print(f"      Emergency switch to DEFENSE")
            self.current_mode = 'DEFENSE'
            return 'DEFENSE'
        
        # Check Hysteresis
        if self._is_stable(suggested_mode):
            # Ổn định ≥ stability_required rounds → chuyển mode
            old_mode = self.current_mode
            self.current_mode = suggested_mode
            
            if old_mode != suggested_mode:
                print(f"   🔄 Mode transition: {old_mode} → {suggested_mode}")
                print(f"      (ρ={threat_ratio:.3f}, stable for {self.stability_required} rounds)")
        else:
            # Chưa ổn định → giữ mode cũ
            pass
        
        return self.current_mode
    
    def _suggest_mode(self, threat_ratio: float) -> str:
        """
        Suggest mode dựa trên threat ratio.
        
        Args:
            threat_ratio: ρ ∈ [0,1]
        
        Returns:
            Suggested mode
        """
        if threat_ratio <= self.threshold_normal:
            return 'NORMAL'
        elif threat_ratio <= self.threshold_defense:
            return 'ALERT'
        else:
            return 'DEFENSE'
    
    def _is_stable(self, suggested_mode: str) -> bool:
        """
        Kiểm tra xem suggested mode có ổn định không.
        
        Ổn định = suggested_mode giống nhau trong ≥ stability_required rounds gần nhất.
        
        Args:
            suggested_mode: Mode được đề xuất
        
        Returns:
            True if stable, False otherwise
        """
        if len(self.mode_history) < self.stability_required:
            return False
        
        # Check last N rounds
        recent = self.mode_history[-self.stability_required:]
        
        # All same as suggested_mode?
        return all(m == suggested_mode for m in recent)
    
    def _check_reputation_gates(self,
                                flagged_clients: List[int],
                                reputations: Dict[int, float]) -> tuple:
        """
        Kiểm tra Reputation Gates để emergency override.
        
        Gate 1: Nhiều high-reputation clients bị flag
        Gate 2: Mean reputation giảm đột ngột
        
        Args:
            flagged_clients: List of flagged client IDs
            reputations: Dict mapping client_id -> reputation
        
        Returns:
            (triggered: bool, reason: str)
        """
        # Gate 1: High-reputation clients flagged
        high_rep_flagged = [
            cid for cid in flagged_clients
            if reputations.get(cid, 0) > self.gate1_rep_threshold
        ]
        
        if len(high_rep_flagged) >= self.gate1_min_high_rep:
            return True, f"Gate 1: {len(high_rep_flagged)} high-rep clients flagged"
        
        # Gate 2: Mean reputation drop
        if len(self.rep_mean_history) >= 2:
            prev_mean = self.rep_mean_history[-2]
            curr_mean = self.rep_mean_history[-1]
            
            if prev_mean > 0:
                drop_ratio = (prev_mean - curr_mean) / prev_mean
                
                if drop_ratio > self.gate2_drop_threshold:
                    return True, f"Gate 2: {drop_ratio*100:.1f}% reputation drop"
        
        return False, ""
    
    def get_mode_thresholds(self, mode: str) -> Dict:
        """
        Get các thresholds cho mode hiện tại.
        
        Returns:
            Dict with thresholds for detection, filtering, etc.
        """
        if mode == 'NORMAL':
            return {
                'rep_threshold': 0.2,
                'confidence_threshold': 0.7,
                'layer1_k': 4.0,
                'layer2_distance_mult': 1.5,
                'layer2_cosine_threshold': 0.3
            }
        elif mode == 'ALERT':
            return {
                'rep_threshold': 0.4,
                'confidence_threshold': 0.6,
                'layer1_k': 3.5,
                'layer2_distance_mult': 1.3,
                'layer2_cosine_threshold': 0.4
            }
        else:  # DEFENSE
            return {
                'rep_threshold': 0.6,
                'confidence_threshold': 0.5,
                'layer1_k': 3.0,
                'layer2_distance_mult': 1.2,
                'layer2_cosine_threshold': 0.5
            }
    
    def get_stats(self) -> Dict:
        """Get controller statistics."""
        mode_counts = {}
        for mode in ['NORMAL', 'ALERT', 'DEFENSE']:
            mode_counts[mode] = self.mode_history.count(mode)
        
        return {
            'current_mode': self.current_mode,
            'total_rounds': len(self.mode_history),
            'mode_counts': mode_counts,
            'mean_rep_current': self.rep_mean_history[-1] if self.rep_mean_history else 0
        }
    
    def reset(self):
        """Reset controller state."""
        self.current_mode = 'NORMAL'
        self.mode_history = []
        self.rep_mean_history = []


# ============================================
# TESTING CODE
# ============================================

def test_mode_controller():
    """Test Mode Controller."""
    print("\n" + "="*70)
    print("🧪 TESTING MODE CONTROLLER")
    print("="*70)
    
    controller = ModeController()
    
    # Simulate reputations
    reputations = {i: 0.8 for i in range(20)}
    
    # Test 1: Normal → Alert transition
    print("\n📊 Test 1: NORMAL → ALERT Transition")
    
    for round_num in range(5):
        threat = 0.05 if round_num < 2 else 0.20  # Low then medium
        flagged = []
        
        mode = controller.update_mode(threat, flagged, reputations, round_num)
        print(f"   Round {round_num}: ρ={threat:.2f} → mode={mode}")
    
    # Test 2: Reputation Gate 1
    print("\n📊 Test 2: Reputation Gate 1 (Emergency DEFENSE)")
    
    controller.reset()
    
    # Many high-rep clients flagged
    flagged = [0, 1, 2, 3]  # 4 high-rep clients
    threat = 0.10  # Low threat
    
    mode = controller.update_mode(threat, flagged, reputations, 0)
    print(f"   ρ={threat:.2f}, {len(flagged)} high-rep flagged")
    print(f"   Result: mode={mode} (expect DEFENSE)")
    
    # Test 3: Reputation Gate 2
    print("\n📊 Test 3: Reputation Gate 2 (Mean Rep Drop)")
    
    controller.reset()
    
    # Simulate reputation drop
    reputations_t1 = {i: 0.8 for i in range(20)}
    reputations_t2 = {i: 0.7 for i in range(20)}  # 12.5% drop
    
    controller.update_mode(0.10, [], reputations_t1, 0)
    mode = controller.update_mode(0.10, [], reputations_t2, 1)
    
    print(f"   Rep: 0.80 → 0.70 (12.5% drop)")
    print(f"   Result: mode={mode} (expect DEFENSE)")
    
    print("\n" + "="*70)
    print("✅ MODE CONTROLLER TESTS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_mode_controller()