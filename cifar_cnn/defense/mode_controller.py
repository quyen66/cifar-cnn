# cifar_cnn/defense/mode_controller.py
"""
Mode Controller V2 - FULLY CONFIGURABLE VERSION
=================================================
Điều khiển chế độ hoạt động theo main.pdf.

ALL PARAMETERS ARE LOADED FROM pyproject.toml via constructor.

Cơ chế chính (từ PDF):
1. Mode Decision: Dựa trên ngưỡng cứng của Threat Ratio (ρ)
   - NORMAL: ρ ≤ threshold_normal (0.15)
   - ALERT: threshold_normal < ρ ≤ threshold_defense (0.30)
   - DEFENSE: ρ > threshold_defense (0.30)

2. Hysteresis: Chỉ chuyển mode nếu ổn định >= N vòng (PDF: 2)

3. Reputation Gates (Emergency Override):
   - Gate 1: ≥ trust_breach_count clients với R > trust_breach_threshold bị flag
   - Gate 2: Rep drop > rep_drop_threshold so với vòng trước

4. Aggregation Algorithms:
   - NORMAL: Weighted Average
   - ALERT: Trimmed Mean  
   - DEFENSE: Coordinate Median
"""

import numpy as np
from typing import List, Dict, Optional
from enum import Enum
from logging import INFO
from flwr.common.logger import log


class SystemMode(Enum):
    """Các chế độ hoạt động của hệ thống."""
    NORMAL = "NORMAL"      # An toàn - Weighted Average
    ALERT = "ALERT"        # Cảnh giác - Trimmed Mean
    DEFENSE = "DEFENSE"    # Nguy hiểm - Coordinate Median


class ModeController:
    """
    Mode Controller V2 - Fully Configurable.
    
    Tất cả tham số đều có thể tinh chỉnh qua constructor (load từ pyproject.toml).
    
    Parameters (from pyproject.toml [tool.flwr.app.config.defense.mode]):
        threshold_normal_to_alert: Ngưỡng ρ để chuyển NORMAL → ALERT (PDF: 0.15)
        threshold_alert_to_defense: Ngưỡng ρ để chuyển ALERT → DEFENSE (PDF: 0.30)
        hysteresis_rounds: Số vòng ổn định trước khi chuyển mode (PDF: 2)
        trust_breach_count: Số clients tin cậy bị flag để trigger Gate 1 (PDF: 3)
        trust_breach_threshold: Ngưỡng R để coi là trusted (PDF: 0.85)
        rep_drop_threshold: % drop rep trung bình để trigger Gate 2 (PDF: 0.10)
        initial_mode: Mode khởi tạo (PDF: NORMAL)
        warmup_rounds: Số vòng warmup (PDF: 10)
        safe_weight_epsilon: Epsilon cho safe aggregation (PDF: 1e-6)
    """
    
    def __init__(
        self,
        # === Threshold Parameters ===
        threshold_normal_to_alert: float = 0.15,
        threshold_alert_to_defense: float = 0.30,
        
        # === Hysteresis Parameters ===
        hysteresis_rounds: int = 2,
        
        # === Gate 1: Trust Breach ===
        trust_breach_count: int = 3,
        trust_breach_threshold: float = 0.85,
        
        # === Gate 2: Reputation Drop ===
        rep_drop_threshold: float = 0.10,
        
        # === Initial State ===
        initial_mode: str = "NORMAL",
        
        # === Warmup ===
        warmup_rounds: int = 10,
        
        # === Safe Aggregation ===
        safe_weight_epsilon: float = 1e-6,
        
        # === Backward Compatibility (ignored, kept for old code) ===
        rep_gate_defense: float = None,
        stability_required: int = None,
    ):
        """
        Initialize Mode Controller với tất cả tham số configurable.
        """
        # Store parameters
        self.threshold_normal = threshold_normal_to_alert
        self.threshold_defense = threshold_alert_to_defense
        self.hysteresis_required = hysteresis_rounds
        
        self.trust_breach_count = trust_breach_count
        self.trust_breach_threshold = trust_breach_threshold
        self.rep_drop_threshold = rep_drop_threshold
        
        self.warmup_rounds = warmup_rounds
        self.safe_weight_epsilon = safe_weight_epsilon
        
        # State
        self.current_mode = SystemMode[initial_mode.upper()]
        self.suggested_mode_history: List[SystemMode] = []
        self.last_avg_rep: float = 0.5
        
        # Log configuration
        self._log_config()
    
    def _log_config(self):
        """Log configuration for debugging."""
        print(f"\n{'='*60}")
        print(f"✅ ModeController Initialized (V2 - Fully Configurable)")
        print(f"{'='*60}")
        print(f"📊 Threshold Parameters:")
        print(f"   NORMAL → ALERT: ρ > {self.threshold_normal}")
        print(f"   ALERT → DEFENSE: ρ > {self.threshold_defense}")
        print(f"\n📊 Hysteresis:")
        print(f"   Stability required: {self.hysteresis_required} rounds")
        print(f"\n📊 Gate 1 (Trust Breach):")
        print(f"   Count threshold: {self.trust_breach_count} clients")
        print(f"   Rep threshold: R > {self.trust_breach_threshold}")
        print(f"\n📊 Gate 2 (Rep Drop):")
        print(f"   Drop threshold: > {self.rep_drop_threshold:.0%}")
        print(f"\n📊 Initial State:")
        print(f"   Mode: {self.current_mode.value}")
        print(f"   Warmup: {self.warmup_rounds} rounds")
        print(f"   Safe epsilon: {self.safe_weight_epsilon}")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # CORE LOGIC
    # =========================================================================
    
    def update_mode(
        self,
        threat_ratio: float,
        detected_clients: List[int],
        reputations: Dict[int, float],
        current_round: int
    ) -> str:
        """
        Cập nhật mode dựa trên threat ratio và reputation gates.
        
        Pipeline theo PDF:
        1. Check warmup phase
        2. Check Gate 1 (Trust Breach)
        3. Check Gate 2 (Rep Drop)
        4. Suggest mode based on ρ
        5. Apply hysteresis
        
        Args:
            threat_ratio: Tỷ lệ clients bị reject (ρ = |F_rejected| / N)
            detected_clients: List client IDs bị detect là malicious
            reputations: Dict[client_id, reputation]
            current_round: Vòng hiện tại
            
        Returns:
            Mode string: "NORMAL", "ALERT", hoặc "DEFENSE"
        """
        # === Step 1: Warmup Phase ===
        if current_round <= self.warmup_rounds:
            self.current_mode = SystemMode.NORMAL
            self.suggested_mode_history.append(SystemMode.NORMAL)
            self._update_last_avg_rep(reputations)
            return SystemMode.NORMAL.value
        
        # === Step 2: Gate 1 - Trust Breach ===
        gate1_triggered = self._check_trust_breach_gate(detected_clients, reputations)
        if gate1_triggered:
            self._force_switch(SystemMode.DEFENSE, 
                f"🚨 [GATE 1] >= {self.trust_breach_count} Trusted Clients (R>{self.trust_breach_threshold}) Flagged")
            self._update_last_avg_rep(reputations)
            return SystemMode.DEFENSE.value
        
        # === Step 3: Gate 2 - Rep Drop ===
        gate2_triggered, drop_rate = self._check_rep_drop_gate(reputations)
        if gate2_triggered:
            self._force_switch(SystemMode.DEFENSE,
                f"📉 [GATE 2] Rep Drop {drop_rate:.1%} > {self.rep_drop_threshold:.1%}")
            self._update_last_avg_rep(reputations)
            return SystemMode.DEFENSE.value
        
        # Update last avg rep for next round
        self._update_last_avg_rep(reputations)
        
        # === Step 4: Suggest Mode based on ρ ===
        suggested_mode = self._suggest_mode_from_rho(threat_ratio)
        self.suggested_mode_history.append(suggested_mode)
        
        # === Step 5: Apply Hysteresis ===
        final_mode = self._apply_hysteresis(suggested_mode, threat_ratio)
        
        return final_mode.value
    
    def _check_trust_breach_gate(
        self,
        detected_clients: List[int],
        reputations: Dict[int, float]
    ) -> bool:
        """
        Gate 1: Kiểm tra vi phạm niềm tin.
        
        Theo PDF: Trigger nếu >= trust_breach_count clients với R > trust_breach_threshold bị flag.
        """
        high_rep_flagged = sum(
            1 for cid in detected_clients 
            if reputations.get(cid, 0) > self.trust_breach_threshold
        )
        
        return high_rep_flagged >= self.trust_breach_count
    
    def _check_rep_drop_gate(
        self,
        reputations: Dict[int, float]
    ) -> tuple:
        """
        Gate 2: Kiểm tra sụt giảm danh tiếng.
        
        Theo PDF: Trigger nếu rep trung bình giảm > rep_drop_threshold (10%) so với vòng trước.
        
        Returns:
            (triggered: bool, drop_rate: float)
        """
        if not reputations:
            return False, 0.0
        
        current_avg_rep = np.mean(list(reputations.values()))
        
        # Tránh chia cho 0
        if self.last_avg_rep < 1e-6:
            return False, 0.0
        
        drop_rate = (self.last_avg_rep - current_avg_rep) / self.last_avg_rep
        
        return drop_rate > self.rep_drop_threshold, drop_rate
    
    def _suggest_mode_from_rho(self, threat_ratio: float) -> SystemMode:
        """
        Đề xuất mode dựa trên threat ratio (ρ).
        
        Theo PDF:
        - ρ ≤ 0.15 → NORMAL
        - 0.15 < ρ ≤ 0.30 → ALERT
        - ρ > 0.30 → DEFENSE
        """
        if threat_ratio <= self.threshold_normal:
            return SystemMode.NORMAL
        elif threat_ratio <= self.threshold_defense:
            return SystemMode.ALERT
        else:
            return SystemMode.DEFENSE
    
    def _apply_hysteresis(
        self,
        suggested_mode: SystemMode,
        threat_ratio: float
    ) -> SystemMode:
        """
        Áp dụng hysteresis: Chỉ chuyển mode nếu ổn định >= N vòng.
        
        Theo PDF: Yêu cầu ổn định trong 2 vòng.
        """
        if len(self.suggested_mode_history) >= self.hysteresis_required:
            recent = self.suggested_mode_history[-self.hysteresis_required:]
            
            if all(m == suggested_mode for m in recent):
                # Ổn định → Chấp nhận chuyển mode
                if self.current_mode != suggested_mode:
                    log(INFO, f"🔄 Mode stable for {self.hysteresis_required} rounds. "
                        f"Switching {self.current_mode.value} → {suggested_mode.value} (ρ={threat_ratio:.2f})")
                self.current_mode = suggested_mode
            else:
                # Chưa ổn định → Giữ mode cũ
                log(INFO, f"⏳ Threat unstable ({suggested_mode.value}). "
                    f"Holding {self.current_mode.value}.")
        else:
            # Chưa đủ lịch sử → Chấp nhận luôn
            self.current_mode = suggested_mode
        
        return self.current_mode
    
    def _force_switch(self, mode: SystemMode, reason: str):
        """Force switch mode và reset history."""
        self.current_mode = mode
        self.suggested_mode_history.append(mode)
        log(INFO, f"{reason} → Force Switch to {mode.value}")
    
    def _update_last_avg_rep(self, reputations: Dict[int, float]):
        """Update last average reputation for next round's Gate 2 check."""
        if reputations:
            self.last_avg_rep = np.mean(list(reputations.values()))
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_current_mode(self) -> str:
        """Get current mode as string."""
        return self.current_mode.value
    
    def get_aggregation_algorithm(self) -> str:
        """
        Get recommended aggregation algorithm for current mode.
        
        Theo PDF:
        - NORMAL: Weighted Average
        - ALERT: Trimmed Mean
        - DEFENSE: Coordinate Median
        """
        if self.current_mode == SystemMode.NORMAL:
            return "weighted_average"
        elif self.current_mode == SystemMode.ALERT:
            return "trimmed_mean"
        else:
            return "coordinate_median"
    
    def get_stats(self) -> Dict:
        """Get controller statistics."""
        return {
            'current_mode': self.current_mode.value,
            'last_avg_rep': self.last_avg_rep,
            'history_length': len(self.suggested_mode_history),
            'recent_suggestions': [m.value for m in self.suggested_mode_history[-5:]],
            'thresholds': {
                'normal_to_alert': self.threshold_normal,
                'alert_to_defense': self.threshold_defense
            }
        }
    
    def get_config(self) -> Dict:
        """Get current configuration."""
        return {
            'threshold_normal_to_alert': self.threshold_normal,
            'threshold_alert_to_defense': self.threshold_defense,
            'hysteresis_rounds': self.hysteresis_required,
            'trust_breach_count': self.trust_breach_count,
            'trust_breach_threshold': self.trust_breach_threshold,
            'rep_drop_threshold': self.rep_drop_threshold,
            'warmup_rounds': self.warmup_rounds,
            'safe_weight_epsilon': self.safe_weight_epsilon,
            'initial_mode': self.current_mode.value
        }