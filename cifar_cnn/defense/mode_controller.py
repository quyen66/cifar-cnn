# cifar_cnn/defense/mode_controller.py

from typing import List, Dict
from logging import INFO
from flwr.common.logger import log
import numpy as np

class ModeController:
    """
    Điều khiển chế độ hoạt động (Strict PDF Version).
    
    Cơ chế chuẩn theo PDF:
    1. Mode Decision: Dựa trên ngưỡng cứng của Threat Ratio (rho).
    2. Hysteresis (Trễ thời gian): Chỉ chuyển mode nếu ổn định >= 2 vòng.
    3. Reputation Gates: Emergency Override 
    """
    
    def __init__(
        self,
        threshold_normal_to_alert: float = 0.15,
        threshold_alert_to_defense: float = 0.30,
        stability_required: int = 2, 
        initial_mode: str = "NORMAL", 
        warmup_rounds: int = 10,
        rep_gate_defense: float = 0.05 # Gate 2 drop threshold
    ):
        self.threshold_normal = threshold_normal_to_alert
        self.threshold_defense = threshold_alert_to_defense
        self.stability_required = stability_required
        self.warmup_rounds = warmup_rounds
        
        # Reputation Gates
        self.rep_drop_threshold = rep_gate_defense 
        self.high_rep_threshold = 0.85 
        
        self.current_mode = initial_mode
        self.suggested_mode_history = [] # Lưu lịch sử đề xuất để check ổn định
        
        # Lưu trữ danh tiếng trung bình
        self.last_avg_rep = 0.5 
        
        log(INFO, f"🎛️ ModeController initialized (PDF Logic).")
        log(INFO, f"   Hysteresis: Require {self.stability_required} stable rounds")
        log(INFO, f"   Thresholds: NORMAL <= {self.threshold_normal} < ALERT <= {self.threshold_defense} < DEFENSE")

    def update_mode(
        self, 
        threat_ratio: float, 
        detected_clients: List[int],
        reputations: Dict[int, float],
        current_round: int
    ) -> str:
        
        # --- 1. Giai đoạn Warmup ---
        if current_round <= self.warmup_rounds:
            self.current_mode = "NORMAL"
            self.suggested_mode_history.append("NORMAL")
            if reputations:
                self.last_avg_rep = np.mean(list(reputations.values()))
            return "NORMAL"

        # --- 2. Reputation Gates (Emergency Override) ---
        # Gate 1: High Rep Clients Flagged
        high_rep_flagged_count = sum(1 for cid in detected_clients if reputations.get(cid, 0) > self.high_rep_threshold)
        
        if high_rep_flagged_count >= 3:
            self._force_switch("DEFENSE", "🚨 [GATE 1] >= 3 Trusted Clients Flagged")
            self._update_last_avg(reputations)
            return "DEFENSE"

        # Gate 2: Reputation Drop
        current_avg_rep = np.mean(list(reputations.values())) if reputations else 0.5
        drop_rate = 0.0
        if self.last_avg_rep > 1e-6:
            drop_rate = (self.last_avg_rep - current_avg_rep) / self.last_avg_rep
            
        if drop_rate > self.rep_drop_threshold:
            self._force_switch("DEFENSE", f"📉 [GATE 2] Rep Drop {drop_rate:.1%} > {self.rep_drop_threshold:.1%}")
            self.last_avg_rep = current_avg_rep
            return "DEFENSE"

        self.last_avg_rep = current_avg_rep

        # --- 3. Mode Suggestion (Based on Rho) ---
        if threat_ratio <= self.threshold_normal:
            suggested_mode = "NORMAL"
        elif threat_ratio <= self.threshold_defense:
            suggested_mode = "ALERT"
        else:
            suggested_mode = "DEFENSE"
            
        self.suggested_mode_history.append(suggested_mode)
        
        # --- 4. Hysteresis Check (Time-based Stability) ---
        # Kiểm tra xem mode đề xuất có giống nhau trong N vòng gần nhất không
        if len(self.suggested_mode_history) >= self.stability_required:
            recent_suggestions = self.suggested_mode_history[-self.stability_required:]
            if all(m == suggested_mode for m in recent_suggestions):
                # Ổn định -> Chấp nhận chuyển mode
                if self.current_mode != suggested_mode:
                    log(INFO, f"🔄 Mode stable for {self.stability_required} rounds. Switching {self.current_mode} -> {suggested_mode} (rho={threat_ratio:.2f})")
                self.current_mode = suggested_mode
            else:
                # Chưa ổn định -> Giữ mode cũ
                log(INFO, f"⏳ Threat unstable ({suggested_mode}). Holding {self.current_mode}.")
                pass
        else:
            # Chưa đủ lịch sử -> Chấp nhận luôn (hoặc giữ mặc định)
            self.current_mode = suggested_mode
            
        return self.current_mode

    def _force_switch(self, mode, reason):
        self.current_mode = mode
        # Reset lịch sử để tránh hysteresis block việc chuyển khẩn cấp này
        self.suggested_mode_history.append(mode)
        log(INFO, f"{reason} -> Force Switch to {mode}")

    def _update_last_avg(self, reputations):
        if reputations:
            self.last_avg_rep = np.mean(list(reputations.values()))

    def get_stats(self) -> Dict:
        return {
            "current_mode": self.current_mode,
            "last_avg_rep": self.last_avg_rep,
            "rho_thresholds": (self.threshold_normal, self.threshold_defense)
        }