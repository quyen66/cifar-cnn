"""
Non-IID Handler (VERIFIED & ENHANCED)
======================================
Xử lý non-IID data để giảm false positives.

Components:
- Heterogeneity Score (H): Đo mức độ non-IID
- Adaptive Thresholds: Điều chỉnh thresholds dựa trên H
- Baseline Tracking: Track gradient history để detect drift

ALL PARAMETERS ARE CONFIGURABLE VIA CONSTRUCTOR (loaded from pyproject.toml)

VERIFIED: Các hàm đã được kiểm tra hoạt động đúng logic trong PDF.
"""

import numpy as np
from typing import Dict, List
from collections import deque


class NonIIDHandler:
    """Non-IID handler với configurable parameters (VERIFIED)."""
    
    def __init__(self,
                 h_threshold_normal: float = 0.6,
                 h_threshold_alert: float = 0.5,
                 adaptive_multiplier: float = 1.5,
                 baseline_percentile: int = 60,
                 baseline_window_size: int = 10):
        """
        Initialize Non-IID Handler with configurable parameters.
        
        Args:
            h_threshold_normal: H threshold for NORMAL mode
            h_threshold_alert: H threshold for ALERT mode
            adaptive_multiplier: Multiplier for adaptive thresholds
            baseline_percentile: Percentile for baseline computation
            baseline_window_size: Window size for gradient history
        """
        self.h_threshold_normal = h_threshold_normal
        self.h_threshold_alert = h_threshold_alert
        self.adaptive_multiplier = adaptive_multiplier
        self.baseline_percentile = baseline_percentile
        self.baseline_window_size = baseline_window_size
        
        # Client gradient history
        self.client_gradients = {}
        
        print(f"✅ NonIIDHandler initialized (VERIFIED) with params:")
        print(f"   H thresholds: normal={h_threshold_normal}, alert={h_threshold_alert}")
        print(f"   Adaptive multiplier: {adaptive_multiplier}")
        print(f"   Baseline: percentile={baseline_percentile}, window={baseline_window_size}")
    
    def compute_heterogeneity_score(self, 
                                    gradients: List[np.ndarray],
                                    client_ids: List[int]) -> float:
        """
        Compute heterogeneity score H.
        
        H ∈ [0, 1]:
        - H = 0: Perfectly homogeneous (IID)
        - H = 1: Highly heterogeneous (non-IID)
        
        Formula: H = min(1.0, CV) where CV = std/mean of pairwise distances
        """
        n = len(gradients)
        
        if n < 2:
            return 0.0
        
        # Compute pairwise distances
        grad_matrix = np.vstack([g.flatten() for g in gradients])
        
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(grad_matrix[i] - grad_matrix[j])
                distances.append(dist)
        
        # Normalize H to [0, 1]
        if len(distances) == 0:
            return 0.0
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # H = CV (Coefficient of Variation), capped at 1.0
        H = min(1.0, std_dist / (mean_dist + 1e-10))
        
        return H
    
    def get_adaptive_threshold(self, H: float, mode: str, base_threshold: float) -> float:
        """
        Get adaptive threshold based on H score and mode.
        
        LOGIC: Higher H → Looser threshold (to reduce false positives)
        
        Args:
            H: Heterogeneity score (0-1)
            mode: Current mode (NORMAL/ALERT/DEFENSE)
            base_threshold: Base threshold value
        
        Returns:
            Adaptive threshold (adjusted based on H and mode)
        
        Example:
            - H=0.7 (high), mode=NORMAL, base=0.4
            - h_threshold_normal=0.6
            - H > 0.6 → High heterogeneity
            - adaptive = 0.4 × 1.5 = 0.6 (loosen to reduce FP)
        """
        # Determine H threshold based on mode
        if mode == 'NORMAL':
            h_threshold = self.h_threshold_normal  # 0.6
        elif mode == 'ALERT':
            h_threshold = self.h_threshold_alert  # 0.5
        else:  # DEFENSE
            h_threshold = 0.4  # Stricter in DEFENSE mode
        
        # If H > threshold → High heterogeneity → Loosen threshold
        if H > h_threshold:
            adaptive_threshold = base_threshold * self.adaptive_multiplier
        else:
            adaptive_threshold = base_threshold
        
        return adaptive_threshold
    
    def update_client_gradient(self, client_id: int, gradient: np.ndarray):
        """
        Update gradient history for a client.
        
        This is called EVERY round to maintain history for baseline tracking.
        """
        if client_id not in self.client_gradients:
            self.client_gradients[client_id] = deque(maxlen=self.baseline_window_size)
        
        self.client_gradients[client_id].append(gradient.flatten())
    
    def compute_baseline_deviation(self, 
                                   client_id: int,
                                   current_gradient: np.ndarray) -> float:
        """
        Compute deviation from client's baseline.
        
        LOGIC: Compare current gradient norm with historical baseline.
        Higher deviation = More anomalous = Potential attack or drift.
        
        Args:
            client_id: Client ID
            current_gradient: Current gradient from this client
        
        Returns:
            Deviation score (0-∞, higher = more anomalous)
            - 0.0: No deviation (perfectly aligned with history)
            - > 0.3: Significant deviation (trigger penalty in confidence scoring)
        
        Formula:
            deviation = |current_norm - baseline| / baseline
            where baseline = percentile(historical_norms)
        
        Example:
            - Historical norms: [0.5, 0.6, 0.55, 0.52, 0.58]
            - Baseline (60th percentile): 0.57
            - Current norm: 0.85
            - Deviation = |0.85 - 0.57| / 0.57 = 0.49 (HIGH)
        """
        if client_id not in self.client_gradients:
            return 0.0
        
        history = list(self.client_gradients[client_id])
        
        if len(history) < 2:
            return 0.0
        
        # Compute baseline from historical norms
        historical_norms = [np.linalg.norm(g) for g in history]
        baseline = np.percentile(historical_norms, self.baseline_percentile)
        
        # Current norm
        current_norm = np.linalg.norm(current_gradient.flatten())
        
        # Deviation (normalized)
        deviation = abs(current_norm - baseline) / (baseline + 1e-10)
        
        return deviation
    
    def get_stats(self) -> Dict:
        """Get handler statistics."""
        return {
            'h_threshold_normal': self.h_threshold_normal,
            'h_threshold_alert': self.h_threshold_alert,
            'adaptive_multiplier': self.adaptive_multiplier,
            'baseline_percentile': self.baseline_percentile,
            'baseline_window_size': self.baseline_window_size,
            'num_tracked_clients': len(self.client_gradients)
        }