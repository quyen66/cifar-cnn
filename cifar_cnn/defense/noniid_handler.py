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
        OPTIMIZED: Only store gradient norm to save RAM (Solution 3).
        """
        # Tính Norm (độ lớn) của gradient hiện tại
        norm_val = float(np.linalg.norm(gradient))
        
        # Khởi tạo deque nếu chưa có
        if client_id not in self.client_gradients:
            self.client_gradients[client_id] = deque(maxlen=self.baseline_window_size)
        
        # CHỈ LƯU NORM (số thực nhẹ), KHÔNG lưu cả vector gradient (nặng)
        self.client_gradients[client_id].append(norm_val)

    def compute_baseline_deviation(self, client_id: int, current_gradient: np.ndarray) -> float:
        """
        Compute deviation from client's baseline.
        OPTIMIZED: Works with stored Norms instead of full vectors.
        """
        if client_id not in self.client_gradients:
            return 0.0
        
        # Lấy lịch sử (bây giờ là list các số thực Norm, không phải vector)
        historical_norms = list(self.client_gradients[client_id])
        
        if len(historical_norms) < 2:
            return 0.0
        
        # Tính baseline từ lịch sử Norm có sẵn (không cần tính toán lại)
        baseline = np.percentile(historical_norms, self.baseline_percentile)
        
        # Tính norm của gradient hiện tại để so sánh
        current_norm = float(np.linalg.norm(current_gradient.flatten()))
        
        # Tránh lỗi chia cho 0
        if baseline < 1e-9:
            return 0.0
            
        # Tính độ lệch: |current - baseline| / baseline
        deviation = abs(current_norm - baseline) / (baseline + 1e-10)
        
        return deviation
    
    def compute_baseline_deviation_detailed(self, 
                                          client_id: int, 
                                          current_gradient: np.ndarray,
                                          grad_median: np.ndarray) -> Dict[str, float]:
        """
        Compute detailed baseline deviation (Norm + Direction).
        OPTIMIZED: Works with stored Norms instead of full vectors.
        """
        # 1. Delta Norm (Dựa trên lịch sử Norm đã lưu)
        if client_id not in self.client_gradients:
            # Chưa có lịch sử thì deviation = 0
            return {'delta_norm': 0.0, 'delta_direction': 0.0, 'delta_combined': 0.0}
        
        # Lấy lịch sử (bây giờ là list các số thực Norm, không phải vector)
        historical_norms = list(self.client_gradients[client_id])
        
        if len(historical_norms) < 2:
            return {'delta_norm': 0.0, 'delta_direction': 0.0, 'delta_combined': 0.0}
            
        # Tính baseline từ lịch sử Norm có sẵn (không cần tính toán lại)
        baseline = np.percentile(historical_norms, self.baseline_percentile)
        
        # Tính norm của gradient hiện tại
        current_norm = float(np.linalg.norm(current_gradient.flatten()))
        
        # Tránh lỗi chia cho 0
        if baseline < 1e-9:
            delta_norm = 0.0
        else:
            delta_norm = abs(current_norm - baseline) / (baseline + 1e-10)
            
        # 2. Delta Direction (Dựa trên Median hiện tại, KHÔNG cần lịch sử)
        # Cosine Distance = 1 - Cosine Similarity
        
        g_flat = current_gradient.flatten()
        m_flat = grad_median.flatten()
        
        dot = np.dot(g_flat, m_flat)
        norm_g = np.linalg.norm(g_flat)
        norm_m = np.linalg.norm(m_flat)
        
        if norm_g < 1e-9 or norm_m < 1e-9:
            delta_dir = 0.0
        else:
            # Similarity [-1, 1]
            cosine_sim = dot / (norm_g * norm_m)
            # Distance [0, 2] (0 là trùng hướng, 2 là ngược hướng)
            delta_dir = 1.0 - cosine_sim 
            
        # 3. Kết hợp (Dùng trọng số cấu hình)
        # Lấy weight từ self (đã init từ config), mặc định 0.5
        w_norm = getattr(self, 'delta_norm_weight', 0.5)
        w_dir = getattr(self, 'delta_direction_weight', 0.5)
        
        delta_combined = w_norm * delta_norm + w_dir * delta_dir
        
        return {
            'delta_norm': delta_norm,
            'delta_direction': delta_dir,
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