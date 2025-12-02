"""
Reputation System (LOGIC FIXED - DYNAMIC INITIALIZATION)
=========================================================
UPDATES:
1. ✅ Dynamic Init: Sử dụng self.initial_reputation cho client mới (không fix cứng).
2. ✅ Fix Infinite Probation Loop: Chỉ bắt vào Probation nếu R < 0.2 VÀ đang giảm.
3. ✅ Probation Logic: Đóng băng EMA, đếm 5 vòng tốt liên tiếp.
"""
import numpy as np
from typing import Dict

class ReputationSystem:
    def __init__(self,
                 ema_alpha_increase: float = 0.15,
                 ema_alpha_decrease: float = 0.5,
                 penalty_flagged: float = 0.2,
                 penalty_variance: float = 0.1,
                 reward_clean: float = 0.1,
                 floor_warning_threshold: float = 0.2, # Ngưỡng vào Probation (PDF: 0.2)
                 probation_rounds: int = 5,            # Số vòng thử thách (PDF: 5)
                 initial_reputation: float = 0.1):     # Mặc định an toàn là 0.1 (Risk Dilution Fix)
        
        self.ema_alpha_increase = ema_alpha_increase
        self.ema_alpha_decrease = ema_alpha_decrease
        self.penalty_flagged = penalty_flagged
        self.penalty_variance = penalty_variance
        self.reward_clean = reward_clean
        
        self.floor_warning_threshold = floor_warning_threshold
        self.probation_rounds = probation_rounds
        self.initial_reputation = initial_reputation
        
        self.reputations = {}
        # Map: {client_id: consecutive_good_rounds}
        self.probation_list = {} 
        
        print(f"✅ ReputationSystem Initialized")
        print(f"   ► Initial Reputation: {self.initial_reputation}")
        print(f"   ► Probation Rule: If R < {floor_warning_threshold} AND dropping -> Freeze for {probation_rounds} rounds.")

    def initialize_client(self, client_id: int, is_trusted: bool = False):
        """Khởi tạo danh tiếng cho client mới."""
        if client_id not in self.reputations:
            if is_trusted:
                # Trusted nodes (Vòng 1-10) luôn bắt đầu max
                self.reputations[client_id] = 1.0
            else:
                # Client mới (Vòng 11+) dùng giá trị cấu hình (nên để thấp ~0.1)
                self.reputations[client_id] = self.initial_reputation

    def get_reputation(self, client_id: int) -> float:
        # Trả về giá trị khởi tạo nếu chưa có
        return self.reputations.get(client_id, self.initial_reputation)

    def update(self,
               client_id: int,
               gradient: np.ndarray,
               grad_median: np.ndarray,
               was_flagged: bool,
               current_round: int,
               baseline_deviation: float = 0.0) -> float:
        """
        Update reputation with Smart Probation Logic.
        """
        self.initialize_client(client_id)
        current_rep = self.reputations[client_id]
        
        # --- CASE 1: CLIENT ĐANG TRONG DANH SÁCH THEO DÕI ---
        if client_id in self.probation_list:
            if was_flagged:
                # Nếu hư trong lúc thử thách: Reset bộ đếm về 0
                self.probation_list[client_id] = 0
                # print(f"   Client {client_id} (Probation): Bad behavior! Counter reset to 0.")
                
                # Vẫn tính phạt để giảm điểm tiếp (răn đe)
                delta = -self.penalty_flagged
                alpha = self.ema_alpha_decrease
                new_rep = current_rep + alpha * delta
                new_rep = max(0.0, min(1.0, new_rep))
                self.reputations[client_id] = new_rep
                return new_rep
            else:
                # Nếu ngoan: Tăng bộ đếm
                self.probation_list[client_id] += 1
                count = self.probation_list[client_id]
                
                if count >= self.probation_rounds:
                    # Đủ 5 vòng -> Thoát Probation (Unlock)
                    del self.probation_list[client_id]
                    print(f"   Client {client_id}: 🎉 Exited Probation after {self.probation_rounds} good rounds.")
                    # Trả về điểm hiện tại (để vòng sau bắt đầu tăng)
                    return current_rep
                else:
                    # Chưa đủ -> Đóng băng (Freeze)
                    # Không cộng điểm thưởng, giữ nguyên điểm cũ
                    return current_rep

        # --- CASE 2: CLIENT BÌNH THƯỜNG (CẬP NHẬT EMA) ---
        # 1. Base Delta
        if was_flagged:
            delta = -self.penalty_flagged
            alpha = self.ema_alpha_decrease
        else:
            delta = self.reward_clean
            alpha = self.ema_alpha_increase
        
        # 2. Variance Penalty
        dist = np.linalg.norm(gradient.flatten() - grad_median)
        median_norm = np.linalg.norm(grad_median)
        norm_dist = dist / (median_norm + 1e-10)
        delta -= min(self.penalty_variance, norm_dist * self.penalty_variance)

        # 3. Baseline Penalty
        if baseline_deviation > 0.3:
            delta -= 0.1
        
        # 4. Calculate New Reputation
        new_rep = current_rep + alpha * delta
        new_rep = max(0.0, min(1.0, new_rep))
        
        # 5. Check Entry to Probation (CRITICAL FIX)
        # Chỉ vào tù nếu điểm thấp dưới ngưỡng VÀ điểm đang giảm (bị phạt).
        # Nếu điểm thấp (<0.2) nhưng đang tăng (do vừa thoát tù/mới vào round 11), thì KHÔNG bắt lại.
        is_dropping = (new_rep < current_rep)
        
        if new_rep < self.floor_warning_threshold and is_dropping and client_id not in self.probation_list:
            self.probation_list[client_id] = 0
            print(f"   Client {client_id}: 🚨 Entered Probation (R={new_rep:.3f} < {self.floor_warning_threshold})")
        
        self.reputations[client_id] = new_rep
        return new_rep

    def get_stats(self) -> Dict:
        if not self.reputations: return {}
        vals = list(self.reputations.values())
        return {
            'mean_reputation': np.mean(vals), 
            'min': np.min(vals), 
            'max': np.max(vals),
            'clients_in_probation': len(self.probation_list)
        }