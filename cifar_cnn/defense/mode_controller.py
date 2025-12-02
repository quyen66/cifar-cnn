# cifar_cnn/defense/mode_controller.py

from typing import List, Dict
from logging import INFO
from flwr.common.logger import log

class ModeController:
    """
    Điều khiển chế độ hoạt động của hệ thống phòng thủ.
    Cơ chế: Chuyển đổi giữa NORMAL, ALERT, DEFENSE dựa trên mức độ đe dọa (rho).
    """
    
    def __init__(
        self,
        threshold_normal_to_alert: float = 0.15,
        threshold_alert_to_defense: float = 0.30,
        hysteresis_normal: float = 0.05,
        hysteresis_defense: float = 0.10,
        rep_gate_defense: float = 0.5,
        initial_mode: str = "DEFENSE", # [FIX] Mặc định ban đầu là DEFENSE để an toàn
        warmup_rounds: int = 10        # [FIX] Thêm tham số Warmup
    ):
        self.threshold_normal = threshold_normal_to_alert
        self.threshold_defense = threshold_alert_to_defense
        self.hysteresis_normal = hysteresis_normal
        self.hysteresis_defense = hysteresis_defense
        self.rep_gate_defense = rep_gate_defense
        self.warmup_rounds = warmup_rounds
        
        self.current_mode = initial_mode
        self.mode_history = []
        
        log(INFO, f"🎛️ ModeController initialized. InitMode={self.current_mode}, Warmup={self.warmup_rounds}")

    def update_mode(
        self, 
        threat_ratio: float, 
        detected_clients: List[int],
        reputations: Dict[int, float],
        current_round: int
    ) -> str:
        """
        Quyết định chế độ dựa trên threat_ratio (rho) và giai đoạn huấn luyện.
        """
        # [FIX] LOGIC KHỞI ĐỘNG THẬN TRỌNG (Conservative Warm-up)
        # Trong giai đoạn warmup, luôn bắt buộc dùng chế độ DEFENSE
        if current_round <= self.warmup_rounds:
            self.current_mode = "DEFENSE"
            self.mode_history.append("DEFENSE")
            log(INFO, f"🛡️ [Warmup Round {current_round}/{self.warmup_rounds}] Force DEFENSE mode to prevent zero-day poisoning.")
            return "DEFENSE"

        # Logic chuyển đổi chế độ thông thường (sau warmup)
        next_mode = self.current_mode
        
        if self.current_mode == "NORMAL":
            if threat_ratio > self.threshold_normal:
                next_mode = "ALERT"
                log(INFO, f"⚠️ Threat level {threat_ratio:.2f} > {self.threshold_normal}. Switch NORMAL -> ALERT")
        
        elif self.current_mode == "ALERT":
            if threat_ratio > self.threshold_defense:
                next_mode = "DEFENSE"
                log(INFO, f"🚨 Threat level {threat_ratio:.2f} > {self.threshold_defense}. Switch ALERT -> DEFENSE")
            elif threat_ratio <= (self.threshold_normal - self.hysteresis_normal):
                next_mode = "NORMAL"
                log(INFO, f"✅ Threat level {threat_ratio:.2f} low enough. Switch ALERT -> NORMAL")
                
        elif self.current_mode == "DEFENSE":
            if threat_ratio <= (self.threshold_defense - self.hysteresis_defense):
                next_mode = "ALERT"
                log(INFO, f"⚠️ Threat level {threat_ratio:.2f} decreased. Switch DEFENSE -> ALERT")
        
        # Cập nhật trạng thái
        self.current_mode = next_mode
        self.mode_history.append(next_mode)
        
        return next_mode

    def get_stats(self) -> Dict:
        return {
            "current_mode": self.current_mode,
            "mode_history": self.mode_history[-10:] # Lấy 10 lịch sử gần nhất
        }