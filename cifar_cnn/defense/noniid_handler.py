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
        weight_h_grad: float = 0.2,
        weight_h_loss: float = 0.4,
        weight_h_acc: float = 0.4,
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
        grad_ema_decay: float = 0.3,
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
        h_weight_sum = weight_h_grad + weight_h_loss + weight_h_acc
        if abs(h_weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"H weights must sum to 1.0, got {h_weight_sum:.6f} "
                f"(cv={weight_h_grad}, sim={weight_h_loss}, cluster={weight_h_acc})"
            )
        
        self.w_grad = weight_h_grad
        self.w_loss = weight_h_loss
        self.w_acc = weight_h_acc
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
        self.h_score_history: List[float] = []
        
        print(f"✅ NonIIDHandler initialized (main.pdf spec):")
        print(f"   H weights: Accuracy={weight_h_acc}, Gradient={weight_h_grad}, Loss={weight_h_loss}")
        print(f"   θ_adj formula: clip(θbase + (H-0.5)×{adjustment_factor}, {theta_adj_clip_min}, {theta_adj_clip_max})")
        print(f"   δi weights: norm={delta_norm_weight}, direction={delta_direction_weight}")
        print(f"   Baseline window: {baseline_window_size} rounds")

    # =========================================================================
    # HETEROGENEITY SCORE (H)
    # =========================================================================
    
    def compute_heterogeneity_score(
        self,
        client_gradients: List[np.ndarray], 
        client_losses: List[float],     
        client_accuracies: List[float]
    ) -> float:
        """
        Tính H-score lai tạo: Gradient + Loss CV + Accuracy CV.
        """
        
        # 1. Component H_grad (Dựa trên Cosine Similarity - Giữ lại từ logic cũ nhưng đơn giản hơn)
        if not client_gradients:
            h_grad = 0.0
        else:
            # Flatten gradients để tính toán
            flat_grads = [np.concatenate([arr.flatten() for arr in g]) for g in client_gradients]
            mean_grad = np.mean(flat_grads, axis=0)
            norm_mean = np.linalg.norm(mean_grad)
            
            sims = []
            for g in flat_grads:
                norm_g = np.linalg.norm(g)
                if norm_g > 1e-9 and norm_mean > 1e-9:
                    sim = np.dot(g, mean_grad) / (norm_g * norm_mean)
                    sims.append(sim)
                else:
                    sims.append(0.0) # Gradient 0 coi như không giống
            
            # Mean sim càng thấp -> càng Non-IID. 
            # Chuyển đổi: Sim=1 -> H=0, Sim=0 -> H=0.5, Sim=-1 -> H=1
            avg_sim = np.mean(sims) if sims else 1.0
            h_grad = 0.5 * (1.0 - avg_sim) # Range [0, 1]

        # 2. Component H_loss (Coefficient of Variation of Loss)
        # Ổn định ngay cả khi Loss nhỏ (hội tụ)
        if not client_losses or len(client_losses) < 2:
            h_loss = 0.0
        else:
            losses = np.array(client_losses)
            mean_loss = np.mean(losses)
            if mean_loss < 1e-9: # Tránh chia cho 0
                h_loss = 0.0
            else:
                cv_loss = np.std(losses) / mean_loss
                # Dùng tanh để chuẩn hóa về [0, 1], scale=1.5 để kéo dãn
                h_loss = np.tanh(cv_loss * 1.5)

        # 3. Component H_acc (Coefficient of Variation of Accuracy)
        # Rất ổn định để đo độ khó dữ liệu
        if not client_accuracies or len(client_accuracies) < 2:
            h_acc = 0.0
        else:
            accs = np.array(client_accuracies)
            mean_acc = np.mean(accs)
            if mean_acc < 1e-9:
                h_acc = 0.0
            else:
                cv_acc = np.std(accs) / mean_acc
                # Acc thường lệch ít, nhân 3.0 để nhạy hơn
                h_acc = np.tanh(cv_acc * 3.0)

        # 4. Tổng hợp
        current_H = (self.w_grad * h_grad) + (self.w_loss * h_loss) + (self.w_acc * h_acc)
        
        # Clip an toàn
        current_H = float(np.clip(current_H, 0.0, 1.0))

        # EMA Smoothing (Làm mượt để threshold không nhảy loạn xạ)
        if self.h_score_history:
            prev_H = self.h_score_history[-1]
            smooth_H = 0.7 * current_H + 0.3 * prev_H
        else:
            smooth_H = current_H
            
        self.h_score_history.append(smooth_H)
        
        # Debug Log chi tiết để bạn dễ theo dõi
        print(f"   📊 H-Score Components: Grad={h_grad:.3f}, Loss={h_loss:.3f}, Acc={h_acc:.3f} -> H_final={smooth_H:.3f}")
        
        return smooth_H
    
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
        """Get last computed H score."""
        return self.last_h_score
    
    def get_client_history_length(self, client_id: int) -> int:
        """Get number of rounds tracked for a client."""
        if client_id not in self.client_histories:
            return 0
        return len(self.client_histories[client_id].norms)
    
    def get_stats(self) -> Dict:
        """Get handler statistics."""
        return {
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

