# cifar_cnn/defense/reputation.py
"""
Reputation System V2 - FULLY CONFIGURABLE VERSION
===================================================
Cập nhật theo main.pdf với TẤT CẢ tham số có thể tinh chỉnh.

ALL PARAMETERS ARE LOADED FROM pyproject.toml via constructor.

Công thức chính (từ PDF):
1. Participation: P(i,t) = 0.5×(Cosine(gi, gmedian) + 1)
2. Consistency: C(i,t) = w_p×P(i,t) + w_hist×C(i,t-1)  [PDF: 0.6, 0.4]
3. Raw Score: rraw = w_c×C + w_p×P  [PDF: 0.5, 0.5]
4. Adjusted: radjusted = rraw × λ(H, status)
5. EMA Update: R(i,t) = (1-α)×R(i,t-1) + α×radjusted

Lambda theo status và H:
- Clean: λ = lambda_clean (1.0)
- Suspicious: λ = lambda_sus_base + lambda_sus_h_mult × H ∈ [0.6, 0.8]
- Rejected: λ = lambda_rej_base + lambda_rej_h_mult × H ∈ [0.1, 0.5]
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque
from enum import Enum


class ClientStatus(Enum):
    """Status của client sau detection pipeline."""
    CLEAN = "clean"           # Không bị flag ở bất kỳ layer nào
    SUSPICIOUS = "suspicious"  # Bị Layer 1 flag nhưng được Layer 2 rescue
    REJECTED = "rejected"      # Bị reject bởi detection hoặc filtering


class ReputationSystem:
    """
    Reputation System V2 - Fully Configurable.
    
    Đây là class chính thức (đã bỏ wrapper legacy).
    """
    
    def __init__(
        self,
        # === EMA Parameters ===
        ema_alpha_increase: float = 0.15,      # α khi điểm tăng (slow reward)
        ema_alpha_decrease: float = 0.5,       # α khi điểm giảm (fast penalty)
        
        # === Lambda Parameters ===
        lambda_clean: float = 1.0,             # λ cho clean clients
        lambda_suspicious_base: float = 0.6,   # λ base cho suspicious
        lambda_suspicious_h_mult: float = 0.2, # λ H multiplier cho suspicious
        lambda_rejected_base: float = 0.1,     # λ base cho rejected
        lambda_rejected_h_mult: float = 0.4,   # λ H multiplier cho rejected
        
        # === Probation Parameters ===
        floor_warning_threshold: float = 0.2,  # Ngưỡng vào probation
        probation_rounds: int = 5,             # Số vòng để thoát probation
        
        # === Initial Values ===
        initial_reputation: float = 0.25,      # R khởi tạo cho clients mới
        trusted_reputation: float = 1.0,       # R cho trusted clients (warmup)
        
        # === Formula Weights (Consistency) ===
        consistency_p_weight: float = 0.6,     # Weight cho P trong C formula
        consistency_history_weight: float = 0.4,  # Weight cho history trong C
        
        # === Formula Weights (Raw Score) ===
        raw_c_weight: float = 0.5,             # Weight cho C trong raw score
        raw_p_weight: float = 0.5,             # Weight cho P trong raw score
        
        # === History Parameters ===
        history_window_size: int = 5,          # Số vòng lưu lịch sử cosine
        
        # === Legacy Params (Ignored but accepted for compatibility) ===
        penalty_flagged: float = 0.2,
        **kwargs
    ):
        """Initialize Reputation System."""
        # Store all parameters
        self.alpha_up = ema_alpha_increase
        self.alpha_down = ema_alpha_decrease
        
        self.lambda_clean = lambda_clean
        self.lambda_sus_base = lambda_suspicious_base
        self.lambda_sus_h_mult = lambda_suspicious_h_mult
        self.lambda_rej_base = lambda_rejected_base
        self.lambda_rej_h_mult = lambda_rejected_h_mult
        
        self.floor_threshold = floor_warning_threshold
        self.probation_rounds = probation_rounds
        
        self.initial_reputation = initial_reputation
        self.trusted_reputation = trusted_reputation
        
        self.w_p_in_c = consistency_p_weight
        self.w_hist_in_c = consistency_history_weight
        self.w_c_in_raw = raw_c_weight
        self.w_p_in_raw = raw_p_weight
        
        self.history_window = history_window_size
        
        # Validate weights sum to 1.0
        self._validate_weights()
        
        # === State Storage ===
        self.reputations: Dict[int, float] = {}           # R(i,t)
        self.consistency_history: Dict[int, float] = {}   # C(i,t-1)
        self.cosine_history: Dict[int, deque] = {}        # Lịch sử cosine
        self.probation_list: Dict[int, int] = {}          # {cid: good_rounds_count}
        
        # Log configuration
        self._log_config()
    
    def _validate_weights(self):
        """Validate that formula weights sum to 1.0."""
        c_sum = self.w_p_in_c + self.w_hist_in_c
        if abs(c_sum - 1.0) > 1e-6:
            print(f"⚠️  Warning: Consistency weights sum to {c_sum}, not 1.0")
        
        r_sum = self.w_c_in_raw + self.w_p_in_raw
        if abs(r_sum - 1.0) > 1e-6:
            print(f"⚠️  Warning: Raw score weights sum to {r_sum}, not 1.0")
    
    def _log_config(self):
        """Log configuration for debugging."""
        print(f"\n{'='*60}")
        print(f"✅ ReputationSystem Initialized (V2 Logic)")
        print(f"{'='*60}")
        print(f"📊 EMA: up={self.alpha_up}, down={self.alpha_down}")
        print(f"📊 Lambda: Clean={self.lambda_clean}, "
              f"Sus={self.lambda_sus_base}+{self.lambda_sus_h_mult}H, "
              f"Rej={self.lambda_rej_base}+{self.lambda_rej_h_mult}H")
        print(f"📊 Init: R_new={self.initial_reputation}, R_trust={self.trusted_reputation}")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def initialize_client(self, client_id: int, is_trusted: bool = False):
        if client_id not in self.reputations:
            if is_trusted:
                self.reputations[client_id] = self.trusted_reputation
            else:
                self.reputations[client_id] = self.initial_reputation
        
        if client_id not in self.consistency_history:
            self.consistency_history[client_id] = 0.5
        
        if client_id not in self.cosine_history:
            self.cosine_history[client_id] = deque(maxlen=self.history_window)
    
    def get_reputation(self, client_id: int) -> float:
        return self.reputations.get(client_id, self.initial_reputation)
    
    # =========================================================================
    # CORE COMPUTATIONS
    # =========================================================================
    
    def _compute_participation(self, gradient: np.ndarray, grad_median: np.ndarray) -> float:
        g_flat = gradient.flatten()
        m_flat = grad_median.flatten()
        norm_g = np.linalg.norm(g_flat)
        norm_m = np.linalg.norm(m_flat)
        
        if norm_g < 1e-9 or norm_m < 1e-9:
            return 0.5
        
        cosine_sim = np.dot(g_flat, m_flat) / (norm_g * norm_m)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        return float(0.5 * (cosine_sim + 1.0))
    
    def _compute_consistency(self, client_id: int, participation: float) -> float:
        prev_consistency = self.consistency_history.get(client_id, 0.5)
        return float(self.w_p_in_c * participation + self.w_hist_in_c * prev_consistency)
    
    def _compute_lambda(self, status: ClientStatus, heterogeneity_score: float) -> float:
        H = np.clip(heterogeneity_score, 0.0, 1.0)
        
        if status == ClientStatus.CLEAN:
            return self.lambda_clean
        elif status == ClientStatus.SUSPICIOUS:
            lambda_val = self.lambda_sus_base + self.lambda_sus_h_mult * H
            return np.clip(lambda_val, self.lambda_sus_base, self.lambda_sus_base + self.lambda_sus_h_mult)
        else:  # REJECTED
            lambda_val = self.lambda_rej_base + self.lambda_rej_h_mult * H
            return np.clip(lambda_val, self.lambda_rej_base, self.lambda_rej_base + self.lambda_rej_h_mult)
    
    # =========================================================================
    # MAIN UPDATE
    # =========================================================================
    
    def update(
        self,
        client_id: int,
        gradient: np.ndarray,
        grad_median: np.ndarray,
        status: ClientStatus,
        heterogeneity_score: float = 0.0,
        current_round: int = 0,
        **kwargs  # Absorb any extra args from old calls
    ) -> float:
        """
        Update reputation V2.
        """
        # Step 1: Initialize
        self.initialize_client(client_id)
        current_rep = self.reputations[client_id]
        
        # Step 2: Handle probation
        was_flagged = (status == ClientStatus.REJECTED)
        
        if client_id in self.probation_list:
            if was_flagged:
                self.probation_list[client_id] = 0 # Reset
                return current_rep
            else:
                self.probation_list[client_id] += 1
                if self.probation_list[client_id] >= self.probation_rounds:
                    del self.probation_list[client_id] # Exit
                    # Continue to update
                else:
                    return current_rep # Still in probation
        
        # Step 3: Compute Scores
        P = self._compute_participation(gradient, grad_median)
        C = self._compute_consistency(client_id, P)
        self.consistency_history[client_id] = C
        
        raw_score = self.w_c_in_raw * C + self.w_p_in_raw * P
        
        # Step 4: Apply Penalty
        lambda_val = self._compute_lambda(status, heterogeneity_score)
        adjusted_score = raw_score * lambda_val
        
        # Step 5: EMA Update
        alpha = self.alpha_down if adjusted_score < current_rep else self.alpha_up
        new_rep = (1 - alpha) * current_rep + alpha * adjusted_score
        new_rep = np.clip(new_rep, 0.0, 1.0)
        
        # Step 6: Check Probation Entry
        if new_rep < self.floor_threshold and new_rep < current_rep:
            if client_id not in self.probation_list:
                self.probation_list[client_id] = 0
        
        self.reputations[client_id] = new_rep
        return new_rep
    
    def get_stats(self) -> Dict:
        vals = list(self.reputations.values())
        if not vals: return {}
        return {
            'mean_reputation': float(np.mean(vals)),
            'clients_in_probation': len(self.probation_list)
        }
    
    def get_all_reputations(self) -> Dict[int, float]:
        return self.reputations.copy()