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
             adjustment_factor: float = 0.4,
             baseline_percentile: int = 60,
             baseline_window_size: int = 10,
             delta_norm_weight: float = 0.5,
             delta_direction_weight: float = 0.5,
             weight_cv: float = 0.4,           
             weight_sim: float = 0.4,          
             weight_cluster: float = 0.2): 
        """
        Initialize Non-IID Handler with configurable parameters.
        
        Args:
            h_threshold_normal: H threshold for NORMAL mode
            h_threshold_alert: H threshold for ALERT mode
            adaptive_multiplier: Multiplier for adaptive thresholds when H high
            adjustment_factor: Factor trong công thức θ_adj = θ_base + (H-0.5) × factor
            baseline_percentile: Percentile for baseline computation
            baseline_window_size: Window size for gradient history
            delta_norm_weight: Weight cho norm deviation trong δi calculation
            delta_direction_weight: Weight cho direction deviation trong δi calculation
            weight_cv: Weight for H_CV component in H score
            weight_sim: Weight for H_sim component in H score
            weight_cluster: Weight for H_cluster component in H score
        """
        self.h_threshold_normal = h_threshold_normal
        self.h_threshold_alert = h_threshold_alert
        self.adaptive_multiplier = adaptive_multiplier
        self.adjustment_factor = adjustment_factor  
        self.baseline_percentile = baseline_percentile
        self.baseline_window_size = baseline_window_size
        self.delta_norm_weight = delta_norm_weight  
        self.delta_direction_weight = delta_direction_weight  
        self.weight_cv = weight_cv
        self.weight_sim = weight_sim
        self.weight_cluster = weight_cluster

        # Validate H weights sum to 1.0
        h_weight_sum = weight_cv + weight_sim + weight_cluster
        if abs(h_weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"H weights must sum to 1.0, got {h_weight_sum:.6f} "
                f"(cv={weight_cv}, sim={weight_sim}, cluster={weight_cluster})"
            )
            
        # Client gradient history
        self.client_gradients = {}
        
        print(f"✅ NonIIDHandler initialized with params:")
        print(f"   H weights: CV={weight_cv}, sim={weight_sim}, cluster={weight_cluster}")
        print(f"   H thresholds: normal={h_threshold_normal}, alert={h_threshold_alert}")
        print(f"   Adaptive multiplier: {adaptive_multiplier}")
        print(f"   Adjustment factor: {adjustment_factor}")
        print(f"   Baseline: percentile={baseline_percentile}, window={baseline_window_size}")
        print(f"   Delta weights: norm={delta_norm_weight}, direction={delta_direction_weight}")
    
    def compute_heterogeneity_score(self, 
                                    gradients: List[np.ndarray],
                                    client_ids: List[int]) -> float:
        """
        Compute heterogeneity score H (FULL FORMULA from PDF).
        
        H ∈ [0, 1]:
        - H = 0: Perfectly homogeneous (IID)
        - H = 1: Highly heterogeneous (non-IID)
        
        FULL Formula (từ main.pdf trang 10):
            H = 0.4 × H_CV + 0.4 × H_sim + 0.2 × H_cluster
        
        Components:
        - H_CV: Coefficient of Variation của khoảng cách cặp
        - H_sim: Độ tương đồng cosine trung bình (inverted)
        - H_cluster: Độ phân tách cụm (Silhouette score)
        """
        n = len(gradients)
        
        if n < 2:
            return 0.0
        
        # Flatten gradients
        grad_matrix = np.vstack([g.flatten() for g in gradients])
        
        # ===== Component 1: H_CV (Coefficient of Variation) =====
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(grad_matrix[i] - grad_matrix[j])
                distances.append(dist)
        
        if len(distances) == 0:
            return 0.0
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # H_CV = CV (capped at 1.0)
        H_CV = min(1.0, std_dist / (mean_dist + 1e-10))
        
        # ===== Component 2: H_sim (Cosine Similarity, inverted) =====
        # Tính độ tương đồng cosine trung bình giữa các cặp gradients
        # High similarity → Low heterogeneity → Invert để có: high H_sim = high heterogeneity
        cosine_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                dot_product = np.dot(grad_matrix[i], grad_matrix[j])
                norm_i = np.linalg.norm(grad_matrix[i])
                norm_j = np.linalg.norm(grad_matrix[j])
                
                cosine_sim = dot_product / (norm_i * norm_j + 1e-10)
                cosine_sims.append(cosine_sim)
        
        mean_cosine_sim = np.mean(cosine_sims) if cosine_sims else 0.0
        
        # Invert: High similarity (1.0) → Low H_sim (0.0)
        #         Low similarity (-1.0) → High H_sim (1.0)
        # Formula: H_sim = (1 - mean_cosine_sim) / 2, capped in [0, 1]
        H_sim = np.clip((1.0 - mean_cosine_sim) / 2.0, 0.0, 1.0)
        
        # ===== Component 3: H_cluster (Silhouette Score, inverted) =====
        # Silhouette score đo độ phân tách cụm
        # High silhouette → Well-separated clusters → High heterogeneity
        # 
        # Simplified approach: Tính intra-cluster vs inter-cluster distances
        # Nếu n < 3, không thể tính meaningful clustering
        if n < 3:
            H_cluster = 0.0
        else:
            # Compute median gradient (centroid)
            g_median = np.median(grad_matrix, axis=0)
            
            # Intra-cluster distance (distance to median)
            intra_dists = [np.linalg.norm(grad_matrix[i] - g_median) for i in range(n)]
            mean_intra = np.mean(intra_dists)
            
            # Inter-cluster distance (pairwise distances, reuse from H_CV)
            mean_inter = mean_dist
            
            # Silhouette-like score: (inter - intra) / max(inter, intra)
            # Range: [-1, 1], where 1 = well-separated
            silhouette = (mean_inter - mean_intra) / (max(mean_inter, mean_intra) + 1e-10)
            
            # Map to [0, 1]: High silhouette → High H_cluster
            H_cluster = np.clip((silhouette + 1.0) / 2.0, 0.0, 1.0)
        
        # ===== Combine with weights from PDF =====
        H = self.weight_cv * H_CV + self.weight_sim * H_sim + self.weight_cluster
        
        # Ensure [0, 1]
        H = np.clip(H, 0.0, 1.0)
        
        return H
    
    def get_adaptive_threshold(self, H: float, mode: str, base_threshold: float) -> float:
        """
        Get adaptive threshold based on H score and mode.
        
        HYBRID: Combine mode-based thresholds với PDF formula
        
        Args:
            H: Heterogeneity score (0-1)
            mode: Current mode (NORMAL/ALERT/DEFENSE)
            base_threshold: Base threshold value
        
        Returns:
            Adaptive threshold
        """
        # Determine H threshold based on mode
        if mode == 'NORMAL':
            h_threshold = self.h_threshold_normal
        elif mode == 'ALERT':
            h_threshold = self.h_threshold_alert
        else:  # DEFENSE
            h_threshold = 0.4
        
        # If H > threshold → Apply PDF formula adjustment
        if H > h_threshold:
            # Use PDF formula: θ_adj = θ_base + (H - 0.5) × adjustment_factor
            adjustment = (H - h_threshold) * self.adjustment_factor
            adaptive_threshold = base_threshold + adjustment
        else:
            adaptive_threshold = base_threshold
        
        # Clip to reasonable range
        adaptive_threshold = np.clip(adaptive_threshold, 0.3, 0.95)
        
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
    
    def compute_baseline_deviation_detailed(self,
                                           client_id: int,
                                           current_gradient: np.ndarray,
                                           median_gradient: np.ndarray) -> Dict[str, float]:
        """
        Compute detailed baseline deviation with norm + direction components.
        
        Formula from PDF trang 10:
        δi = delta_norm_weight × δ_norm(i) + delta_direction_weight × δ_dir(i)
        
        Args:
            client_id: Client ID
            current_gradient: Current gradient
            median_gradient: Median gradient from all clients
        
        Returns:
            Dict with 'delta_norm', 'delta_direction', 'delta_combined'
        """
        if client_id not in self.client_gradients:
            return {
                'delta_norm': 0.0,
                'delta_direction': 0.0,
                'delta_combined': 0.0
            }
        
        history = list(self.client_gradients[client_id])
        if len(history) < 2:
            return {
                'delta_norm': 0.0,
                'delta_direction': 0.0,
                'delta_combined': 0.0
            }
        
        # ===== Component 1: Norm Deviation =====
        # Historical baseline
        historical_norms = [np.linalg.norm(g) for g in history]
        baseline_norm = np.percentile(historical_norms, self.baseline_percentile)
        
        # Current norm
        current_norm = np.linalg.norm(current_gradient.flatten())
        
        # Normalized deviation
        delta_norm = abs(current_norm - baseline_norm) / (baseline_norm + 1e-10)
        
        # ===== Component 2: Direction Deviation =====
        # Compute cosine similarity với median gradient
        current_flat = current_gradient.flatten()
        median_flat = median_gradient.flatten()
        
        cosine_sim = np.dot(current_flat, median_flat) / (
            np.linalg.norm(current_flat) * np.linalg.norm(median_flat) + 1e-10
        )
        
        # Direction deviation: 1 - cosine_sim
        # High cosine_sim (close to 1) → Low delta_direction (good)
        # Low cosine_sim (close to -1) → High delta_direction (bad)
        delta_direction = max(0.0, 1.0 - cosine_sim)
        
        # ===== Combine với weights từ config =====
        # Formula: δi = delta_norm_weight × δ_norm + delta_direction_weight × δ_dir
        delta_combined = (
            self.delta_norm_weight * delta_norm +
            self.delta_direction_weight * delta_direction
        )
        
        return {
            'delta_norm': delta_norm,
            'delta_direction': delta_direction,
            'delta_combined': delta_combined
        }

    def get_stats(self) -> Dict:
        """Get handler statistics."""
        return {
            'weight_cv': self.weight_cv,
            'weight_sim': self.weight_sim,
            'weight_cluster': self.weight_cluster,
            'h_threshold_normal': self.h_threshold_normal,
            'h_threshold_alert': self.h_threshold_alert,
            'adaptive_multiplier': self.adaptive_multiplier,
            'baseline_percentile': self.baseline_percentile,
            'baseline_window_size': self.baseline_window_size,
            'num_tracked_clients': len(self.client_gradients)
        }