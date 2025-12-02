# cifar_cnn/defense/mode_controller.py

from typing import List, Dict
from logging import INFO
from flwr.common.logger import log

class ModeController:
    """
    Điều khiển chế độ hoạt động của hệ thống phòng thủ.
    
    Cơ chế: 
    1. Warmup (Vòng 1-10): Mặc định NORMAL (giả định Trusted Initialization).
    2. Reputation Gate: Nếu >= 3 client uy tín bị flag -> Force DEFENSE.
    3. Threat Ratio: Chuyển đổi NORMAL <-> ALERT <-> DEFENSE dựa trên rho và hysteresis.
    """
    
    def __init__(
        self,
        threshold_normal_to_alert: float = 0.15,
        threshold_alert_to_defense: float = 0.30,
        hysteresis_normal: float = 0.05,
        hysteresis_defense: float = 0.10,
        initial_mode: str = "NORMAL", 
        warmup_rounds: int = 10        
    ):
        self.threshold_normal = threshold_normal_to_alert
        self.threshold_defense = threshold_alert_to_defense
        self.hysteresis_normal = hysteresis_normal
        self.hysteresis_defense = hysteresis_defense
        self.warmup_rounds = warmup_rounds
        
        self.current_mode = initial_mode
        self.mode_history = []
        
        log(INFO, f"🎛️ ModeController initialized. Warmup={self.warmup_rounds} rounds.")

    def update_mode(
        self, 
        threat_ratio: float, 
        detected_clients: List[int],
        reputations: Dict[int, float],
        current_round: int
    ) -> str:
        """
        Quyết định chế độ dựa trên threat_ratio (rho), danh tiếng và giai đoạn huấn luyện.
        """
        
        # 1. Giai đoạn Warmup / Trusted Initialization (PDF Trang 13)
        # Trong 10 vòng đầu, giả định chạy trên tập Trusted Client -> Dùng NORMAL để hội tụ nhanh.
        if current_round <= self.warmup_rounds:
            self.current_mode = "NORMAL"
            self.mode_history.append("NORMAL")
            log(INFO, f"🛡️ [Warmup {current_round}/{self.warmup_rounds}] Trusted Phase -> Mode: NORMAL")
            return "NORMAL"

        # 2. Reputation Gates (PDF Trang 12 - Cổng 1)
        # Nếu có >= 3 client uy tín (R > 0.85) bị đánh dấu là tấn công -> Có biến lớn -> DEFENSE ngay.
        high_rep_flagged_count = 0
        for client_id in detected_clients:
            # Lấy reputation hiện tại, mặc định 0.5 nếu chưa có
            rep = reputations.get(client_id, 0.5)
            if rep > 0.85:
                high_rep_flagged_count += 1
        
        if high_rep_flagged_count >= 3:
            self.current_mode = "DEFENSE"
            self.mode_history.append("DEFENSE")
            log(INFO, f"🚨 [REP GATE TRIGGERED] {high_rep_flagged_count} Trusted Clients Flagged! -> Force DEFENSE")
            return "DEFENSE"

        # 3. Logic chuyển đổi chế độ dựa trên Threat Ratio (Hysteresis)
        next_mode = self.current_mode
        
        if self.current_mode == "NORMAL":
            if threat_ratio > self.threshold_normal:
                next_mode = "ALERT"
                log(INFO, f"⚠️ Threat {threat_ratio:.2f} > {self.threshold_normal}. Switch NORMAL -> ALERT")
        
        elif self.current_mode == "ALERT":
            if threat_ratio > self.threshold_defense:
                next_mode = "DEFENSE"
                log(INFO, f"🚨 Threat {threat_ratio:.2f} > {self.threshold_defense}. Switch ALERT -> DEFENSE")
            elif threat_ratio <= (self.threshold_normal - self.hysteresis_normal):
                next_mode = "NORMAL"
                log(INFO, f"✅ Threat {threat_ratio:.2f} low enough. Switch ALERT -> NORMAL")
                
        elif self.current_mode == "DEFENSE":
            if threat_ratio <= (self.threshold_defense - self.hysteresis_defense):
                next_mode = "ALERT"
                log(INFO, f"⚠️ Threat {threat_ratio:.2f} decreased. Switch DEFENSE -> ALERT")
        
        # Cập nhật trạng thái
        self.current_mode = next_mode
        self.mode_history.append(next_mode)
        
        return next_mode

    def get_stats(self) -> Dict:
        return {
            "current_mode": self.current_mode,
            "mode_history_last_10": self.mode_history[-10:]
        }