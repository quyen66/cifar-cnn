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
        warmup_rounds: int = 10,
        rep_gate_defense: float = 0.5
    ):
        self.threshold_normal = threshold_normal_to_alert
        self.threshold_defense = threshold_alert_to_defense
        self.hysteresis_normal = hysteresis_normal
        self.hysteresis_defense = hysteresis_defense
        self.warmup_rounds = warmup_rounds
        
        # Ngưỡng sụt giảm danh tiếng (PDF: 0.05 tức 5%)
        # Nếu config truyền vào 0.5 (sai), code sẽ dùng giá trị đó. 
        # Khuyên bạn nên sửa config thành 0.05.
        self.rep_drop_threshold = rep_gate_defense 
        
        # Ngưỡng xác định client uy tín (PDF: 0.85)
        self.high_rep_threshold = 0.85 
        
        self.current_mode = initial_mode
        self.mode_history = []
        
        # Lưu trữ danh tiếng trung bình của vòng trước để so sánh (Gate 2)
        self.last_avg_rep = 0.5 # Giá trị khởi tạo giả định
        
        log(INFO, f"🎛️ ModeController initialized.")
        log(INFO, f"   Warmup: {warmup_rounds} rounds")
        log(INFO, f"   Gate 1 (High Rep): Threshold > {self.high_rep_threshold}")
        log(INFO, f"   Gate 2 (Rep Drop): Threshold > {self.rep_drop_threshold:.2f} (Target: 0.05)")

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
        
        # --- 1. Giai đoạn Warmup / Trusted Initialization (PDF Trang 13) ---
        if current_round <= self.warmup_rounds:
            self.current_mode = "NORMAL"
            self.mode_history.append("NORMAL")
            # Cập nhật avg rep để chuẩn bị cho các vòng sau
            if reputations:
                self.last_avg_rep = np.mean(list(reputations.values()))
            return "NORMAL"

        # --- 2. Reputation Gate 1: High Rep Clients Flagged (PDF Trang 12) ---
        high_rep_flagged_count = 0
        for client_id in detected_clients:
            rep = reputations.get(client_id, 0.5)
            if rep > self.high_rep_threshold:
                high_rep_flagged_count += 1
        
        if high_rep_flagged_count >= 3:
            self._set_defense_mode("🚨 [GATE 1] >= 3 Trusted Clients Flagged")
            self._update_last_avg(reputations)
            return "DEFENSE"

        # --- 3. Reputation Gate 2: Average Reputation Drop (PDF Trang 12) ---
        # Tính R_bar_t (Danh tiếng trung bình hiện tại)
        current_avg_rep = np.mean(list(reputations.values())) if reputations else 0.5
        
        # Tránh chia cho 0
        if self.last_avg_rep > 1e-6:
            drop_rate = (self.last_avg_rep - current_avg_rep) / self.last_avg_rep
        else:
            drop_rate = 0.0
            
        # Kiểm tra sụt giảm > 0.05 (5%)
        if drop_rate > self.rep_drop_threshold:
            self._set_defense_mode(f"📉 [GATE 2] Rep Drop {drop_rate:.1%} > {self.rep_drop_threshold:.1%}")
            self.last_avg_rep = current_avg_rep
            return "DEFENSE"

        # Cập nhật last_avg_rep cho vòng kế tiếp
        self.last_avg_rep = current_avg_rep

        # --- 4. Logic Hysteresis dựa trên Threat Ratio (Bình thường) ---
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

    def _set_defense_mode(self, reason: str):
        """Helper để force chuyển sang DEFENSE và log lý do."""
        self.current_mode = "DEFENSE"
        self.mode_history.append("DEFENSE")
        log(INFO, f"{reason} -> Force Switch to DEFENSE")

    def _update_last_avg(self, reputations):
        """Helper để cập nhật avg rep."""
        if reputations:
            self.last_avg_rep = np.mean(list(reputations.values()))

    def get_stats(self) -> Dict:
        return {
            "current_mode": self.current_mode,
            "last_avg_rep": self.last_avg_rep,
            "mode_history_last_10": self.mode_history[-10:]
        }