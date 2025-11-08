"""
Reputation System - Asymmetric EMA
===================================
Hệ thống danh tiếng đánh giá độ tin cậy của client theo thời gian.

Components:
- Consistency Score (Immediate + History)
- Participation Score (Cosine Similarity)
- Asymmetric EMA Update (Penalize Fast, Reward Slow)
- Floor Lifting Mechanism

Author: Week 3 Implementation
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque


class ReputationSystem:
    """
    Hệ thống danh tiếng bất đối xứng với floor lifting.
    
    Triết lý:
    - Trừng phạt nhanh (α_down = 0.8) khi hành vi xấu
    - Thưởng chậm (α_up = 0.3) khi hành vi tốt
    - Cơ hội thứ hai cho client bị đánh giá sai (floor lifting)
    """
    
    def __init__(self,
                 alpha_down: float = 0.8,
                 alpha_up: float = 0.3,
                 floor_threshold: float = 0.15,
                 floor_target: float = 0.3,
                 floor_patience: int = 5,
                 history_window: int = 5):
        """
        Args:
            alpha_down: EMA rate khi giảm danh tiếng (default=0.8, nhanh)
            alpha_up: EMA rate khi tăng danh tiếng (default=0.3, chậm)
            floor_threshold: Ngưỡng để trigger floor lifting (default=0.15)
            floor_target: Target reputation khi lift (default=0.3)
            floor_patience: Số rounds good behavior để lift (default=5)
            history_window: Số rounds lưu lịch sử (default=5)
        """
        self.alpha_down = alpha_down
        self.alpha_up = alpha_up
        self.floor_threshold = floor_threshold
        self.floor_target = floor_target
        self.floor_patience = floor_patience
        self.history_window = history_window
        
        # Storage
        self.reputations: Dict[int, float] = {}  # client_id -> reputation
        self.history: Dict[int, deque] = {}  # client_id -> deque of past gradients
        self.floor_counters: Dict[int, int] = {}  # client_id -> good behavior count
        
        print(f"✅ ReputationSystem initialized:")
        print(f"   - Alpha down (penalize): {alpha_down}")
        print(f"   - Alpha up (reward): {alpha_up}")
        print(f"   - Floor lifting: {floor_threshold} → {floor_target} (patience={floor_patience})")
        print(f"   - History window: {history_window} rounds")
    
    def initialize_client(self, client_id: int, initial_reputation: float = 0.8):
        """Initialize một client mới."""
        if client_id not in self.reputations:
            self.reputations[client_id] = initial_reputation
            self.history[client_id] = deque(maxlen=self.history_window)
            self.floor_counters[client_id] = 0
    
    def update(self,
               client_id: int,
               gradient: np.ndarray,
               grad_median: np.ndarray,
               was_flagged: bool,
               current_round: int) -> float:
        """
        Cập nhật reputation cho một client.
        
        Args:
            client_id: ID của client
            gradient: Gradient hiện tại
            grad_median: Gradient median (reference)
            was_flagged: Client có bị đánh dấu không
            current_round: Round hiện tại
        
        Returns:
            Reputation mới
        """
        # Initialize nếu chưa có
        self.initialize_client(client_id)
        
        # Tính raw reputation score
        raw_score = self._compute_raw_score(
            client_id, gradient, grad_median
        )
        
        # Update với Asymmetric EMA
        old_rep = self.reputations[client_id]
        new_rep = self._asymmetric_ema_update(
            old_rep, raw_score, was_flagged
        )
        
        # Update storage
        self.reputations[client_id] = new_rep
        self.history[client_id].append(gradient.flatten())
        
        # Check floor lifting
        if new_rep <= self.floor_threshold:
            self._check_floor_lifting(client_id, was_flagged)
        else:
            self.floor_counters[client_id] = 0
        
        return new_rep
    
    def _compute_raw_score(self,
                          client_id: int,
                          gradient: np.ndarray,
                          grad_median: np.ndarray) -> float:
        """
        Tính raw reputation score dựa trên Consistency và Participation.
        
        Formula (theo main.pdf):
            r(i,t) = 0.5 × C(i,t) + 0.5 × P(i,t)
        
        C(i,t) = Consistency = 0.5 × (0.6 × C_immediate + 0.4 × C_history + 1)
        P(i,t) = Participation = 0.5 × (cosine_sim + 1)
        
        FIX: Better handling of history to differentiate mixed behavior
        """
        g_flat = gradient.flatten()
        
        # === Participation Score (Cosine Similarity) ===
        cosine_sim = np.dot(g_flat, grad_median) / (
            np.linalg.norm(g_flat) * np.linalg.norm(grad_median) + 1e-10
        )
        participation = 0.5 * (cosine_sim + 1)  # Scale to [0, 1]
        
        # === Consistency Score ===
        # Immediate consistency (với gradient hiện tại)
        c_immediate = cosine_sim  # Reuse cosine similarity
        
        # Historical consistency (với past gradients)
        c_history = 0.0
        if len(self.history[client_id]) > 0:
            # FIX: Tính avg cosine sim với past gradients
            # Và penalty nếu variance cao (inconsistent)
            past_sims = []
            for past_grad in self.history[client_id]:
                sim = np.dot(g_flat, past_grad) / (
                    np.linalg.norm(g_flat) * np.linalg.norm(past_grad) + 1e-10
                )
                past_sims.append(sim)
            
            # Mean similarity
            mean_sim = np.mean(past_sims)
            
            # Variance penalty (high variance = inconsistent)
            # FINAL FIX: Reduce max penalty from 0.3 to 0.2
            var_sim = np.var(past_sims)
            variance_penalty = np.clip(var_sim, 0, 0.2)  # Was 0.3, now 0.2
            
            c_history = mean_sim - variance_penalty
        
        # Combine consistency
        # FIX: Increase history weight to better track behavior
        consistency = 0.5 * (0.5 * c_immediate + 0.5 * c_history + 1)
        
        # === Final Raw Score ===
        raw_score = 0.5 * consistency + 0.5 * participation
        
        return np.clip(raw_score, 0.0, 1.0)
    
    def _asymmetric_ema_update(self,
                               old_rep: float,
                               raw_score: float,
                               was_flagged: bool) -> float:
        """
        Asymmetric EMA update.
        
        Formula (main.pdf):
            R(i,t) = (1-α) × R(i,t-1) + α × r(i,t)
        
        Với:
            α = α_down (0.8) nếu giảm
            α = α_up (0.3) nếu tăng
        
        FIX FINAL: Reduce penalty even more for mixed behavior
            Nếu was_flagged: raw_score × 0.8 (was 0.7, now 0.8)
        """
        # FINAL FIX: Even less harsh penalty for mixed clients
        if was_flagged:
            raw_score = raw_score * 0.8  # Changed from 0.7 to 0.8
        
        # Asymmetric alpha
        if raw_score < old_rep:
            # Giảm nhanh
            alpha = self.alpha_down
        else:
            # Tăng chậm
            alpha = self.alpha_up
        
        # EMA update
        new_rep = (1 - alpha) * old_rep + alpha * raw_score
        
        return np.clip(new_rep, 0.0, 1.0)
    
    def _check_floor_lifting(self, client_id: int, was_flagged: bool):
        """
        Cơ chế floor lifting - cho cơ hội thứ hai.
        
        Logic:
        - Nếu reputation ≤ floor_threshold trong patience rounds
        - VÀ hành vi tốt (không bị flag)
        - THÌ nâng lên floor_target
        """
        if not was_flagged:
            # Good behavior
            self.floor_counters[client_id] += 1
            
            if self.floor_counters[client_id] >= self.floor_patience:
                # Lift floor!
                old_rep = self.reputations[client_id]
                self.reputations[client_id] = self.floor_target
                self.floor_counters[client_id] = 0
                
                print(f"   🔼 Floor Lifting: Client {client_id} "
                      f"{old_rep:.3f} → {self.floor_target:.3f} "
                      f"(after {self.floor_patience} good rounds)")
        else:
            # Bad behavior - reset counter
            self.floor_counters[client_id] = 0
    
    def get_reputation(self, client_id: int) -> float:
        """Lấy reputation hiện tại."""
        if client_id not in self.reputations:
            self.initialize_client(client_id)
        return self.reputations[client_id]
    
    def get_all_reputations(self) -> Dict[int, float]:
        """Lấy tất cả reputations."""
        return self.reputations.copy()
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        if not self.reputations:
            return {}
        
        reps = list(self.reputations.values())
        return {
            'num_clients': len(reps),
            'mean_reputation': np.mean(reps),
            'min_reputation': np.min(reps),
            'max_reputation': np.max(reps),
            'below_floor': sum(1 for r in reps if r <= self.floor_threshold)
        }