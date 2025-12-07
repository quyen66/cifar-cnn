# cifar_cnn/defense/reputation.py

import numpy as np
from typing import Dict, List
from collections import deque

class ReputationSystem:
    """
    Reputation System với Adaptive Penalty cho Non-IID.
    """
    def __init__(self,
                 ema_alpha_increase: float = 0.15,
                 ema_alpha_decrease: float = 0.5,
                 penalty_flagged: float = 0.2,     
                 floor_warning_threshold: float = 0.2, 
                 probation_rounds: int = 5,            
                 initial_reputation: float = 0.1):     
        
        self.alpha_up = ema_alpha_increase
        self.alpha_down = ema_alpha_decrease
        self.penalty_flagged = penalty_flagged
        
        self.floor_threshold = floor_warning_threshold
        self.probation_rounds = probation_rounds
        self.initial_reputation = initial_reputation
        
        self.reputations = {}
        self.probation_list = {} 
        self.history_cosine = {} 
        
        print(f"✅ ReputationSystem Initialized (Adaptive Penalty Mode)")

    def initialize_client(self, client_id: int, is_trusted: bool = False):
        if client_id not in self.reputations:
            self.reputations[client_id] = 1.0 if is_trusted else self.initial_reputation
        if client_id not in self.history_cosine:
            self.history_cosine[client_id] = deque(maxlen=5)

    def get_reputation(self, client_id: int) -> float:
        return self.reputations.get(client_id, self.initial_reputation)

    def update(self,
               client_id: int,
               gradient: np.ndarray,
               grad_median: np.ndarray,
               was_flagged: bool,
               current_round: int,
               baseline_deviation: float = 0.0,
               heterogeneity_score: float = 0.0) -> float: 
        
        self.initialize_client(client_id)
        current_rep = self.reputations[client_id]
        
        # --- 1. Xử lý PROBATION ---
        if client_id in self.probation_list:
            if was_flagged:
                self.probation_list[client_id] = 0
                new_rep = max(0.0, current_rep - self.penalty_flagged)
                self.reputations[client_id] = new_rep
                return new_rep
            else:
                self.probation_list[client_id] += 1
                if self.probation_list[client_id] >= self.probation_rounds:
                    del self.probation_list[client_id]
                    return current_rep
                else:
                    return current_rep

        # --- 2. Tính toán điểm danh tiếng ---
        # A. Cosine & History
        g_flat = gradient.flatten()
        m_flat = grad_median.flatten()
        dot = np.dot(g_flat, m_flat)
        norm_g = np.linalg.norm(g_flat)
        norm_m = np.linalg.norm(m_flat)
        
        cosine_sim = dot / (norm_g * norm_m) if (norm_g > 1e-9 and norm_m > 1e-9) else 0.0
        self.history_cosine[client_id].append(cosine_sim)
        
        # B. Consistency & Participation (Công thức PDF)
        c_imm = cosine_sim
        c_hist = np.mean(self.history_cosine[client_id])
        
        participation_score = 0.5 * (cosine_sim + 1)
        consistency_score = 0.5 * (0.6 * c_imm + 0.4 * c_hist + 1)
        
        raw_score = 0.5 * consistency_score + 0.5 * participation_score
        
        # --- 3. XỬ LÝ PHẠT THÍCH ỨNG (ADAPTIVE PENALTY) ---
        if was_flagged:
            # Logic: 
            # H = 0 (IID)     -> Rất tin vào Flag -> Phạt nặng -> Hệ số nhân thấp (0.2)
            # H = 1 (Non-IID) -> Ít tin vào Flag  -> Phạt nhẹ  -> Hệ số nhân cao (0.6)
            
            # Công thức tuyến tính đơn giản:
            # penalty_multiplier đi từ 0.2 (khi H=0) đến 0.6 (khi H=1)
            penalty_multiplier = 0.2 + (0.4 * heterogeneity_score)
            
            # Clip để đảm bảo an toàn
            penalty_multiplier = np.clip(penalty_multiplier, 0.1, 0.8)
            
            # Áp dụng phạt
            raw_score = raw_score * penalty_multiplier
            
            # (Optional) Log để debug nếu cần
            # print(f"   Client {client_id} flagged. H={heterogeneity_score:.2f} -> Penalty mult={penalty_multiplier:.2f}")

        # E. Cập nhật EMA
        if raw_score < current_rep:
            alpha = self.alpha_down
        else:
            alpha = self.alpha_up
            
        new_rep = (1 - alpha) * current_rep + alpha * raw_score
        
        if baseline_deviation > 0.3:
            new_rep -= 0.1
            
        new_rep = np.clip(new_rep, 0.0, 1.0)

        # F. Kiểm tra vào Probation
        is_dropping = (new_rep < current_rep)
        if new_rep < self.floor_threshold and is_dropping and client_id not in self.probation_list:
            self.probation_list[client_id] = 0
        
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