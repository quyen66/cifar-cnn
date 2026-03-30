"""
Non-IID Handler (main.pdf Spec)
================================
Xử lý dữ liệu không đồng nhất để giảm false positives.

Components theo PDF:
1. Heterogeneity Score (H): Đo mức độ non-IID
   H = 0.4×H_CV + 0.4×H_sim + 0.2×H_cluster

2. Adaptive Threshold (θ_adj): Điều chỉnh ngưỡng dựa trên H
   θ_adj = clip(θbase + (H - 0.5) × 0.4, 0.5, 0.9)

3. Baseline Deviation (δi): Track hành vi từng client
   δi = 0.5 × |∥gi∥ - n̄i| / σn,i + 0.5 × (1 - Cosine(gi, ḡ_hist_i))

ALL PARAMETERS ARE CONFIGURABLE VIA CONSTRUCTOR (loaded from pyproject.toml)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class ClientHistory:
    """Lịch sử gradient của một client."""
    norms: deque  # Lịch sử ||g||
    running_mean_grad: Optional[np.ndarray]  # Running average của gradient (normalized)
    update_count: int  # Số lần cập nhật


class NonIIDHandler:
    """
    Non-IID Handler theo main.pdf spec.
    
    Chức năng chính:
    1. Tính H score để đo độ heterogeneity
    2. Tính θ_adj (adaptive threshold) theo công thức PDF
    3. Track baseline deviation δi cho từng client
    """
    
    def __init__(
        self,
        # H Score weights (phải sum = 1.0)
        weight_cv: float = 0.4,
        weight_sim: float = 0.4,
        weight_cluster: float = 0.2,
        # θ_adj parameters (theo PDF)
        adjustment_factor: float = 0.4,  # Hệ số trong công thức θ_adj
        theta_adj_clip_min: float = 0.5,  # Min của θ_adj
        theta_adj_clip_max: float = 0.9,  # Max của θ_adj
        # Baseline tracking
        baseline_window_size: int = 10,  # Số vòng lưu lịch sử
        # δi weights (theo PDF: 0.5 và 0.5)
        delta_norm_weight: float = 0.5,
        delta_direction_weight: float = 0.5,
        # EMA decay cho running mean gradient
        grad_ema_decay: float = 0.3
    ):
        """
        Initialize Non-IID Handler.
        
        Args:
            weight_cv: Weight cho H_CV (default: 0.4)
            weight_sim: Weight cho H_sim (default: 0.4)
            weight_cluster: Weight cho H_cluster (default: 0.2)
            adjustment_factor: Hệ số trong θ_adj = θbase + (H-0.5)×factor
            theta_adj_clip_min: Min clip cho θ_adj (default: 0.5)
            theta_adj_clip_max: Max clip cho θ_adj (default: 0.9)
            baseline_window_size: Số vòng lưu lịch sử norm
            delta_norm_weight: Weight cho phần norm trong δi
            delta_direction_weight: Weight cho phần direction trong δi
            grad_ema_decay: Decay rate cho running mean gradient
        """
        # Validate H weights
        h_weight_sum = weight_cv + weight_sim + weight_cluster
        if abs(h_weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"H weights must sum to 1.0, got {h_weight_sum:.6f} "
                f"(cv={weight_cv}, sim={weight_sim}, cluster={weight_cluster})"
            )
        
        # H Score weights
        self.weight_cv = weight_cv
        self.weight_sim = weight_sim
        self.weight_cluster = weight_cluster
        
        # θ_adj parameters
        self.adjustment_factor = adjustment_factor
        self.theta_adj_clip_min = theta_adj_clip_min
        self.theta_adj_clip_max = theta_adj_clip_max
        
        # Baseline tracking
        self.baseline_window_size = baseline_window_size
        self.delta_norm_weight = delta_norm_weight
        self.delta_direction_weight = delta_direction_weight
        self.grad_ema_decay = grad_ema_decay
        
        # Client histories: Dict[client_id, ClientHistory]
        self.client_histories: Dict[int, ClientHistory] = {}
        
        # Last computed H score
        self.last_h_score: float = 0.0
        
        print(f"✅ NonIIDHandler initialized (GDS — Gradient Dispersion Score):")
        print(f"   GDS weights: CV={weight_cv}, sim={weight_sim}, cluster={weight_cluster}")
        print(f"   θ_adj formula: clip(θbase + (0.5 - θ_signal) × {adjustment_factor}, {theta_adj_clip_min}, {theta_adj_clip_max})")
        print(f"   δi weights: norm={delta_norm_weight}, direction={delta_direction_weight}")
        print(f"   Baseline window: {baseline_window_size} rounds")

    # =========================================================================
    # HETEROGENEITY SCORE (H)
    # =========================================================================
    
    def compute_heterogeneity_score(
        self,
        gradients: List[np.ndarray],
        client_ids: List[int]
    ) -> float:
        """
        Tính Heterogeneity Score H với robust pre-filtering.
        
        Formula (PDF):
            H = 0.4×H_CV + 0.4×H_sim + 0.2×H_cluster
        
        Pre-filtering: Loại 30% gradient xa median nhất để tránh bị thao túng.
        
        Args:
            gradients: List of gradient arrays
            client_ids: List of client IDs
            
        Returns:
            H score trong [0, 1]
        """
        n = len(gradients)
        if n < 3:
            self.last_h_score = 0.0
            return 0.0
        
        # === ROBUST PRE-FILTERING ===
        # Loại 30% gradient xa median nhất (vì max attack ratio = 30%)
        grad_matrix = np.vstack([g.flatten() for g in gradients])
        g_median = np.median(grad_matrix, axis=0)
        
        dists_to_median = np.linalg.norm(grad_matrix - g_median, axis=1)
        
        keep_ratio = 0.7
        num_keep = max(2, int(n * keep_ratio))
        
        sorted_indices = np.argsort(dists_to_median)
        safe_indices = sorted_indices[:num_keep]
        safe_grad_matrix = grad_matrix[safe_indices]
        n_safe = len(safe_grad_matrix)
        
        # === TÍNH GDS COMPONENTS TRÊN TẬP SẠch ===

        # 1. D_CV: Coefficient of Variation của pairwise distances
        H_CV = self._compute_h_cv(safe_grad_matrix)

        # 2. D_sim: Cosine dissimilarity
        H_sim = self._compute_h_sim(safe_grad_matrix)

        # 3. D_cluster: Cluster dispersion
        H_cluster = self._compute_h_cluster(safe_grad_matrix)

        # Combine
        H = (self.weight_cv * H_CV +
             self.weight_sim * H_sim +
             self.weight_cluster * H_cluster)

        H = np.clip(H, 0.0, 1.0)
        self.last_h_score = H

        # Log GDS với component breakdown (compact, 1 dòng)
        print(f"   🔢 GDS={H:.4f}  "
              f"(D_CV={H_CV:.3f}×{self.weight_cv}  "
              f"D_sim={H_sim:.3f}×{self.weight_sim}  "
              f"D_cluster={H_cluster:.3f}×{self.weight_cluster}  "
              f"| n_safe={n_safe}/{n})")

        return H
    
    def _compute_h_cv(self, grad_matrix: np.ndarray) -> float:
        """
        Tính H_CV: Coefficient of Variation của pairwise distances.
        
        H_CV cao → distances vary nhiều → non-IID cao
        """
        n = len(grad_matrix)
        if n < 2:
            return 0.0
        
        # Pairwise distances
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(grad_matrix[i] - grad_matrix[j])
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # CV = std / mean (normalize to [0, 1])
        H_CV = min(1.0, std_dist / (mean_dist + 1e-10))
        
        return H_CV
    
    def _compute_h_sim(self, grad_matrix: np.ndarray) -> float:
        """
        Tính H_sim: Dissimilarity dựa trên cosine similarity.
        
        H_sim cao → cosine similarities thấp → non-IID cao
        """
        n = len(grad_matrix)
        if n < 2:
            return 0.0
        
        norms = np.linalg.norm(grad_matrix, axis=1)
        
        cosine_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                if norms[i] < 1e-9 or norms[j] < 1e-9:
                    continue
                dot_product = np.dot(grad_matrix[i], grad_matrix[j])
                sim = dot_product / (norms[i] * norms[j])
                cosine_sims.append(sim)
        
        if not cosine_sims:
            return 0.0
        
        mean_cosine_sim = np.mean(cosine_sims)
        
        # Convert similarity to dissimilarity, normalize to [0, 1]
        # sim ∈ [-1, 1] → dissim ∈ [0, 1]
        H_sim = np.clip((1.0 - mean_cosine_sim) / 2.0, 0.0, 1.0)
        
        return H_sim
    
    def _compute_h_cluster(self, grad_matrix: np.ndarray) -> float:
        """
        Tính H_cluster: Cluster dispersion (simplified).
        
        Dùng variance của distances to centroid.
        """
        n = len(grad_matrix)
        if n < 2:
            return 0.0
        
        centroid = np.mean(grad_matrix, axis=0)
        distances_to_centroid = np.linalg.norm(grad_matrix - centroid, axis=1)
        
        # Normalize variance
        mean_dist = np.mean(distances_to_centroid)
        std_dist = np.std(distances_to_centroid)
        
        if mean_dist < 1e-9:
            return 0.0
        
        # CV of distances to centroid
        H_cluster = min(1.0, std_dist / (mean_dist + 1e-10))
        
        return H_cluster

    # =========================================================================
    # ADAPTIVE THRESHOLD (θ_adj)
    # =========================================================================
    
    def compute_adaptive_threshold(self, H: float, theta_base: float) -> float:
        """
        Tính ngưỡng thích ứng theo công thức PDF.
        
        Formula (PDF Giai đoạn 3):
            θ_adj = clip(θbase + (0.5 - H) × 0.4, 0.5, 0.9)
        
        Logic:
        - H = 0.5 (trung bình) → θ_adj = θbase
        - H > 0.5 (non-IID cao) → θ_adj > θbase (nới lỏng, dễ accept hơn)
        - H < 0.5 (IID) → θ_adj < θbase (siết chặt, khó accept hơn)
        
        Args:
            H: Heterogeneity score [0, 1]
            theta_base: Base threshold (e.g., 0.85 for hard filter, 0.65 for soft)
            
        Returns:
            Adaptive threshold θ_adj
        """
        theta_adj = theta_base + (0.5 - H) * self.adjustment_factor
        
        # Clip theo range cho phép
        theta_adj = np.clip(theta_adj, self.theta_adj_clip_min, self.theta_adj_clip_max)
        
        return theta_adj
    
    def get_adaptive_threshold(
        self, 
        H: float, 
        mode: str, 
        base_threshold: float
    ) -> float:
        """
        Wrapper cho compute_adaptive_threshold (backward compatible).
        
        Args:
            H: Heterogeneity score
            mode: Current mode (NORMAL/ALERT/DEFENSE) - không dùng trong PDF formula
            base_threshold: Base threshold
            
        Returns:
            Adaptive threshold
        """
        # PDF formula không phụ thuộc mode, chỉ phụ thuộc H
        return self.compute_adaptive_threshold(H, base_threshold)

    # =========================================================================
    # BASELINE DEVIATION (δi)
    # =========================================================================
    
    def update_client_gradient(self, client_id: int, gradient: np.ndarray):
        """
        Cập nhật lịch sử gradient cho client.
        
        Lưu trữ:
        1. Lịch sử norm (để tính mean và std)
        2. Running mean gradient normalized (để tính cosine với lịch sử)
        
        Args:
            client_id: Client ID
            gradient: Current gradient vector
        """
        g_flat = gradient.flatten()
        norm_val = float(np.linalg.norm(g_flat))
        
        # Khởi tạo nếu chưa có
        if client_id not in self.client_histories:
            self.client_histories[client_id] = ClientHistory(
                norms=deque(maxlen=self.baseline_window_size),
                running_mean_grad=None,
                update_count=0
            )
        
        history = self.client_histories[client_id]
        
        # Cập nhật norm history
        history.norms.append(norm_val)
        
        # Cập nhật running mean gradient (normalized để tính direction)
        if norm_val > 1e-9:
            g_normalized = g_flat / norm_val
        else:
            g_normalized = g_flat
        
        if history.running_mean_grad is None:
            history.running_mean_grad = g_normalized.copy()
        else:
            # EMA update
            alpha = self.grad_ema_decay
            history.running_mean_grad = (
                (1 - alpha) * history.running_mean_grad + alpha * g_normalized
            )
        
        history.update_count += 1
    
    def compute_baseline_deviation(
        self, 
        client_id: int, 
        current_gradient: np.ndarray,
        grad_median: Optional[np.ndarray] = None
    ) -> float:
        """
        Tính độ lệch baseline δi theo công thức PDF.
        
        Formula (PDF):
            δi = 0.5 × |∥gi∥ - n̄i| / σn,i + 0.5 × (1 - Cosine(gi, ḡ_hist_i))
        
        Trong đó:
        - n̄i: Mean của lịch sử norm
        - σn,i: Std của lịch sử norm
        - ḡ_hist_i: Running mean gradient (normalized)
        
        Args:
            client_id: Client ID
            current_gradient: Current gradient vector
            grad_median: Global median gradient (không dùng trong PDF formula δi)
            
        Returns:
            Baseline deviation δi ∈ [0, 2] (có thể > 1 nếu cực lệch)
        """
        if client_id not in self.client_histories:
            return 0.0
        
        history = self.client_histories[client_id]
        
        # Cần ít nhất 2 samples để tính std
        if len(history.norms) < 2:
            return 0.0
        
        g_flat = current_gradient.flatten()
        current_norm = float(np.linalg.norm(g_flat))
        
        # === PHẦN 1: NORM DEVIATION ===
        # δ_norm = |∥gi∥ - n̄i| / σn,i
        norms_list = list(history.norms)
        mean_norm = np.mean(norms_list)
        std_norm = np.std(norms_list)
        
        if std_norm < 1e-9:
            delta_norm = 0.0
        else:
            delta_norm = abs(current_norm - mean_norm) / std_norm
        
        # Clip để tránh giá trị quá lớn
        delta_norm = min(delta_norm, 3.0)  # Cap at 3 std
        delta_norm = delta_norm / 3.0  # Normalize to [0, 1]
        
        # === PHẦN 2: DIRECTION DEVIATION ===
        # δ_dir = 1 - Cosine(gi, ḡ_hist_i)
        if history.running_mean_grad is None or current_norm < 1e-9:
            delta_dir = 0.0
        else:
            g_normalized = g_flat / current_norm
            hist_grad = history.running_mean_grad
            hist_norm = np.linalg.norm(hist_grad)
            
            if hist_norm < 1e-9:
                delta_dir = 0.0
            else:
                hist_normalized = hist_grad / hist_norm
                cosine_sim = np.dot(g_normalized, hist_normalized)
                cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                # Cosine ∈ [-1, 1] → δ_dir ∈ [0, 2], normalize to [0, 1]
                delta_dir = (1.0 - cosine_sim) / 2.0
        
        # === COMBINE ===
        delta_i = (self.delta_norm_weight * delta_norm + 
                   self.delta_direction_weight * delta_dir)
        
        return delta_i
    
    def compute_baseline_deviation_detailed(
        self,
        client_id: int,
        current_gradient: np.ndarray,
        grad_median: np.ndarray
    ) -> Dict[str, float]:
        """
        Tính chi tiết baseline deviation với breakdown.
        
        Returns:
            Dict với delta_norm, delta_direction, delta_combined
        """
        if client_id not in self.client_histories:
            return {
                'delta_norm': 0.0,
                'delta_direction': 0.0,
                'delta_combined': 0.0
            }
        
        history = self.client_histories[client_id]
        
        if len(history.norms) < 2:
            return {
                'delta_norm': 0.0,
                'delta_direction': 0.0,
                'delta_combined': 0.0
            }
        
        g_flat = current_gradient.flatten()
        current_norm = float(np.linalg.norm(g_flat))
        
        # PHẦN 1: NORM DEVIATION (theo PDF: dùng lịch sử cá nhân)
        norms_list = list(history.norms)
        mean_norm = np.mean(norms_list)
        std_norm = np.std(norms_list)
        
        if std_norm < 1e-9:
            delta_norm = 0.0
        else:
            delta_norm = abs(current_norm - mean_norm) / std_norm
            delta_norm = min(delta_norm, 3.0) / 3.0  # Normalize to [0, 1]
        
        # PHẦN 2: DIRECTION DEVIATION (so với running mean cá nhân)
        if history.running_mean_grad is None or current_norm < 1e-9:
            delta_dir = 0.0
        else:
            g_normalized = g_flat / current_norm
            hist_grad = history.running_mean_grad
            hist_norm = np.linalg.norm(hist_grad)
            
            if hist_norm < 1e-9:
                delta_dir = 0.0
            else:
                hist_normalized = hist_grad / hist_norm
                cosine_sim = np.dot(g_normalized, hist_normalized)
                cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                delta_dir = (1.0 - cosine_sim) / 2.0
        
        # COMBINE
        delta_combined = (self.delta_norm_weight * delta_norm +
                         self.delta_direction_weight * delta_dir)
        
        return {
            'delta_norm': delta_norm,
            'delta_direction': delta_dir,
            'delta_combined': delta_combined
        }

    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_last_h_score(self) -> float:
        """Get last computed score (GDS). Alias kept for backward compat."""
        return self.last_h_score

    def get_last_gds(self) -> float:
        """Get last computed GDS score (same as last_h_score)."""
        return self.last_h_score

    def compute_gds(
        self,
        gradients: List[np.ndarray],
        client_ids: List[int]
    ) -> float:
        """
        Alias của compute_heterogeneity_score() với tên rõ nghĩa hơn.
        GDS = w_cv × D_CV + w_sim × D_sim + w_cluster × D_cluster
        Đo sự phân tán hình học gradient → proxy cho môi trường non-IID.
        """
        return self.compute_heterogeneity_score(gradients, client_ids)

    def get_client_history_length(self, client_id: int) -> int:
        """Get number of rounds tracked for a client."""
        if client_id not in self.client_histories:
            return 0
        return len(self.client_histories[client_id].norms)
    
    def get_stats(self) -> Dict:
        """Get handler statistics."""
        return {
            'weight_cv': self.weight_cv,
            'weight_sim': self.weight_sim,
            'weight_cluster': self.weight_cluster,
            'adjustment_factor': self.adjustment_factor,
            'theta_adj_clip_min': self.theta_adj_clip_min,
            'theta_adj_clip_max': self.theta_adj_clip_max,
            'baseline_window_size': self.baseline_window_size,
            'delta_norm_weight': self.delta_norm_weight,
            'delta_direction_weight': self.delta_direction_weight,
            'num_tracked_clients': len(self.client_histories),
            'last_h_score': self.last_h_score
        }
    
    def reset_client_history(self, client_id: int):
        """Reset history for a specific client."""
        if client_id in self.client_histories:
            del self.client_histories[client_id]
    
    def reset_all_histories(self):
        """Reset all client histories."""
        self.client_histories.clear()



# =============================================================================
# BehavioralScorer — Behavioral Heterogeneity Score (H)
# =============================================================================

class BehavioralScorer:
    """
    Tính Behavioral Heterogeneity Score (H) từ gradient direction,
    loss, và accuracy của các clients trong round hiện tại.

    H = w_grad × h_grad + w_loss × h_loss + w_acc × h_acc

    Khác với GDS (NonIIDHandler):
      - GDS đo gradient geometry → proxy cho non-IID environment
      - H  đo behavioral divergence (gradient direction + loss + accuracy)
            → phản ánh threat level, detect được ALIE qua CV(loss)/CV(acc)

    Vai trò trong pipeline:
      - Variant A (θ_adj theo H):   dùng H cho cả filter threshold VÀ mode/reputation
      - Variant B (θ_adj theo GDS): dùng H cho mode/reputation, GDS cho filter threshold
    """

    def __init__(
        self,
        # H weights — phải sum = 1.0
        weight_grad: float = 0.4,   # Gradient direction divergence
        weight_loss: float = 0.3,   # Loss CV
        weight_acc:  float = 0.3,   # Accuracy CV
    ):
        h_sum = weight_grad + weight_loss + weight_acc
        if abs(h_sum - 1.0) > 1e-6:
            raise ValueError(
                f"H weights must sum to 1.0, got {h_sum:.6f} "
                f"(grad={weight_grad}, loss={weight_loss}, acc={weight_acc})"
            )
        self.w_grad = weight_grad
        self.w_loss = weight_loss
        self.w_acc  = weight_acc
        self.last_h_score: float = 0.0

        print(f"✅ BehavioralScorer initialized:")
        print(f"   H weights: grad={weight_grad}, loss={weight_loss}, acc={weight_acc}")

    def compute_h_score(
        self,
        gradients: List[np.ndarray],
        losses: Optional[List[float]] = None,
        accuracies: Optional[List[float]] = None
    ) -> float:
        """
        Tính H = w_grad × h_grad + w_loss × h_loss + w_acc × h_acc.

        Args:
            gradients:   Gradient arrays từ round hiện tại
            losses:      Local training losses từ client metrics (có thể None)
            accuracies:  Local training accuracies từ client metrics (có thể None)

        Returns:
            H score ∈ [0, 1]
        """
        n = len(gradients)
        if n < 2:
            self.last_h_score = 0.0
            return 0.0

        h_grad = self._compute_h_grad(gradients)
        h_loss = self._compute_h_cv_metric(losses, scale=0.5) if losses and len(losses) == n else 0.0
        h_acc  = self._compute_h_cv_metric(accuracies, scale=2.0) if accuracies and len(accuracies) == n else 0.0

        H = float(np.clip(
            self.w_grad * h_grad + self.w_loss * h_loss + self.w_acc * h_acc,
            0.0, 1.0
        ))
        self.last_h_score = H
        return H

    def _compute_h_grad(self, gradients: List[np.ndarray]) -> float:
        """
        h_grad: Mean pairwise cosine divergence = (1 - cosine_sim) / 2.
        Đo sự phân kỳ hướng gradient giữa các clients.
        """
        n = len(gradients)
        if n < 2:
            return 0.0
        # Normalize
        g_norm = []
        for g in gradients:
            gf = g.flatten()
            norm = np.linalg.norm(gf)
            g_norm.append(gf / norm if norm > 1e-9 else np.zeros_like(gf))
        # Pairwise cosine divergence
        divs = []
        for i in range(n):
            for j in range(i + 1, n):
                cos = float(np.clip(np.dot(g_norm[i], g_norm[j]), -1.0, 1.0))
                divs.append((1.0 - cos) / 2.0)  # ∈ [0, 1]
        return float(np.mean(divs)) if divs else 0.0

    def _compute_h_cv_metric(self, values: List[float], scale: float = 1.0) -> float:
        """
        Tính normalized CV của một metric (loss hoặc accuracy).
        scale: hệ số điều chỉnh để đưa CV về [0, 1]:
          - loss: scale=0.5  (CV loss thường 0–2 → × 0.5 → 0–1)
          - acc:  scale=2.0  (CV acc thường 0–0.5 → × 2.0 → 0–1)
        """
        if not values or len(values) < 2:
            return 0.0
        arr = np.array(values, dtype=float)
        mean_val = np.mean(arr)
        if mean_val < 1e-8:
            return 0.0
        cv = np.std(arr) / mean_val
        return float(min(1.0, cv * scale))

    def get_breakdown(
        self,
        gradients: List[np.ndarray],
        losses: Optional[List[float]] = None,
        accuracies: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Tính H và trả về breakdown từng component — dùng cho logging."""
        n = len(gradients)
        if n < 2:
            return {'h_grad': 0.0, 'h_loss': 0.0, 'h_acc': 0.0, 'H': 0.0}
        h_grad = self._compute_h_grad(gradients)
        h_loss = self._compute_h_cv_metric(losses, 0.5) if losses and len(losses) == n else 0.0
        h_acc  = self._compute_h_cv_metric(accuracies, 2.0) if accuracies and len(accuracies) == n else 0.0
        H = float(np.clip(self.w_grad * h_grad + self.w_loss * h_loss + self.w_acc * h_acc, 0.0, 1.0))
        self.last_h_score = H

        # Log H với component breakdown (compact, 1 dòng)
        loss_note = f"n={len(losses)}" if losses and len(losses) == n else "N/A (fallback)"
        acc_note  = f"n={len(accuracies)}" if accuracies and len(accuracies) == n else "N/A (fallback)"
        print(f"   🔢 H={H:.4f}  "
              f"(h_grad={h_grad:.3f}×{self.w_grad}  "
              f"h_loss={h_loss:.3f}×{self.w_loss} [{loss_note}]  "
              f"h_acc={h_acc:.3f}×{self.w_acc} [{acc_note}])")

        return {'h_grad': h_grad, 'h_loss': h_loss, 'h_acc': h_acc, 'H': H}

    def get_last_h_score(self) -> float:
        return self.last_h_score


# =============================================================================
# TESTING
# =============================================================================

def test_noniid_handler():
    """Test NonIIDHandler theo PDF spec."""
    print("\n" + "="*70)
    print("🧪 TESTING NONIID HANDLER (PDF SPEC)")
    print("="*70)
    
    np.random.seed(42)
    
    # Initialize handler
    handler = NonIIDHandler(
        weight_cv=0.4,
        weight_sim=0.4,
        weight_cluster=0.2,
        adjustment_factor=0.4,
        theta_adj_clip_min=0.5,
        theta_adj_clip_max=0.9
    )
    
    # =========================================================================
    # TEST 1: θ_adj Formula
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 1: θ_adj Formula (PDF: θbase + (H-0.5)×0.4)")
    print("-"*70)
    
    test_cases = [
        # (H, θbase, expected_θadj)
        (0.5, 0.85, 0.85),      # H=0.5 → no adjustment
        (0.5, 0.65, 0.65),
        (0.7, 0.85, 0.85 + 0.2*0.4),  # H>0.5 → increase
        (0.3, 0.85, 0.85 - 0.2*0.4),  # H<0.5 → decrease
        (1.0, 0.85, 0.9),       # H=1.0 → max clip
        (0.0, 0.85, 0.65),      # H=0.0 → θbase - 0.2
        (0.0, 0.5, 0.5),        # Min clip
    ]
    
    all_pass = True
    for H, theta_base, expected in test_cases:
        result = handler.compute_adaptive_threshold(H, theta_base)
        expected_clipped = np.clip(expected, 0.5, 0.9)
        passed = abs(result - expected_clipped) < 1e-6
        status = "✅" if passed else "❌"
        print(f"   {status} H={H:.1f}, θbase={theta_base:.2f} → θadj={result:.3f} (expected: {expected_clipped:.3f})")
        if not passed:
            all_pass = False
    
    # =========================================================================
    # TEST 2: H Score
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 2: H Score Computation")
    print("-"*70)
    
    # IID data: similar gradients
    iid_grads = [np.random.randn(1000) * 0.1 + np.ones(1000) for _ in range(20)]
    H_iid = handler.compute_heterogeneity_score(iid_grads, list(range(20)))
    print(f"   IID gradients (similar):     H = {H_iid:.3f} (expected: low, < 0.3)")
    
    # Non-IID data: diverse gradients
    noniid_grads = [np.random.randn(1000) * i * 0.5 for i in range(1, 21)]
    H_noniid = handler.compute_heterogeneity_score(noniid_grads, list(range(20)))
    print(f"   Non-IID gradients (diverse): H = {H_noniid:.3f} (expected: high, > 0.4)")
    
    if H_iid < H_noniid:
        print(f"   ✅ H_iid < H_noniid: Correct ordering")
    else:
        print(f"   ❌ H_iid >= H_noniid: Wrong ordering")
        all_pass = False
    
    # =========================================================================
    # TEST 3: Baseline Deviation δi
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 3: Baseline Deviation δi")
    print("-"*70)
    
    handler2 = NonIIDHandler()
    
    # Simulate consistent client
    base_grad = np.random.randn(1000)
    for i in range(10):
        noisy_grad = base_grad + np.random.randn(1000) * 0.01
        handler2.update_client_gradient(client_id=1, gradient=noisy_grad)
    
    # Test δi for similar gradient
    similar_grad = base_grad + np.random.randn(1000) * 0.01
    delta_similar = handler2.compute_baseline_deviation(1, similar_grad)
    print(f"   Similar gradient: δi = {delta_similar:.3f} (expected: low, < 0.2)")
    
    # Test δi for different gradient
    different_grad = -base_grad * 2  # Opposite direction, different magnitude
    delta_different = handler2.compute_baseline_deviation(1, different_grad)
    print(f"   Different gradient: δi = {delta_different:.3f} (expected: high, > 0.5)")
    
    if delta_similar < delta_different:
        print(f"   ✅ δ_similar < δ_different: Correct ordering")
    else:
        print(f"   ❌ δ_similar >= δ_different: Wrong ordering")
        all_pass = False
    
    # =========================================================================
    # TEST 4: Detailed Deviation
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 4: Detailed Baseline Deviation")
    print("-"*70)
    
    grad_median = np.median(np.vstack([base_grad, similar_grad]), axis=0)
    detail = handler2.compute_baseline_deviation_detailed(1, different_grad, grad_median)
    print(f"   δ_norm:      {detail['delta_norm']:.3f}")
    print(f"   δ_direction: {detail['delta_direction']:.3f}")
    print(f"   δ_combined:  {detail['delta_combined']:.3f}")
    
    if detail['delta_combined'] == (0.5 * detail['delta_norm'] + 0.5 * detail['delta_direction']):
        print(f"   ✅ Combined formula correct")
    else:
        print(f"   ❌ Combined formula wrong")
        all_pass = False
    
    # =========================================================================
    # TEST 5: Edge Cases
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 5: Edge Cases")
    print("-"*70)
    
    handler3 = NonIIDHandler()
    
    # New client (no history)
    delta_new = handler3.compute_baseline_deviation(999, np.random.randn(1000))
    print(f"   New client (no history): δi = {delta_new:.3f} (expected: 0.0)")
    if delta_new == 0.0:
        print(f"   ✅ Correct")
    else:
        print(f"   ❌ Wrong")
        all_pass = False
    
    # Client with 1 sample (not enough for std)
    handler3.update_client_gradient(998, np.random.randn(1000))
    delta_one = handler3.compute_baseline_deviation(998, np.random.randn(1000))
    print(f"   Client with 1 sample: δi = {delta_one:.3f} (expected: 0.0)")
    if delta_one == 0.0:
        print(f"   ✅ Correct")
    else:
        print(f"   ❌ Wrong")
        all_pass = False
    
    # Very few gradients for H
    H_few = handler3.compute_heterogeneity_score([np.ones(100)], [0])
    print(f"   H with 1 gradient: H = {H_few:.3f} (expected: 0.0)")
    if H_few == 0.0:
        print(f"   ✅ Correct")
    else:
        print(f"   ❌ Wrong")
        all_pass = False
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    if all_pass:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70 + "\n")
    
    return all_pass


if __name__ == "__main__":
    test_noniid_handler()