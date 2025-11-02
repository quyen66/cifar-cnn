"""
Non-IID Handler - Week 3
=========================
Xử lý dữ liệu không đồng nhất để giảm False Positive Rate.

Components:
1. Heterogeneity Score: Đo mức độ không đồng nhất
2. Adaptive Thresholds: Điều chỉnh ngưỡng dựa trên heterogeneity
3. Baseline Tracking: Theo dõi hành vi cơ bản của mỗi client

Author: Week 3 Implementation
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import deque


class NonIIDHandler:
    """
    Xử lý dữ liệu không đồng nhất trong Federated Learning.
    
    Giảm False Positive bằng cách:
    - Tính toán mức độ không đồng nhất của dữ liệu
    - Điều chỉnh ngưỡng phát hiện thích ứng
    - Theo dõi baseline behavior của từng client
    """
    
    def __init__(self, 
                 history_window: int = 5,
                 cv_weight: float = 0.4,
                 sim_weight: float = 0.4,
                 cluster_weight: float = 0.2):
        """
        Args:
            history_window: Số rounds lưu lịch sử gradient (default=5)
            cv_weight: Trọng số cho Coefficient of Variation (default=0.4)
            sim_weight: Trọng số cho Similarity variance (default=0.4)
            cluster_weight: Trọng số cho Cluster spread (default=0.2)
        """
        self.history_window = history_window
        self.cv_weight = cv_weight
        self.sim_weight = sim_weight
        self.cluster_weight = cluster_weight
        
        # Storage
        self.gradient_history: Dict[int, deque] = {}  # client_id -> deque of gradients
        self.baseline_norms: Dict[int, float] = {}  # client_id -> baseline norm
        self.baseline_directions: Dict[int, np.ndarray] = {}  # client_id -> baseline direction
        
        print(f"✅ NonIIDHandler initialized:")
        print(f"   - History window: {history_window} rounds")
        print(f"   - Weights: CV={cv_weight}, Sim={sim_weight}, Cluster={cluster_weight}")
    
    def initialize_client(self, client_id: int):
        """Initialize storage cho một client mới."""
        if client_id not in self.gradient_history:
            self.gradient_history[client_id] = deque(maxlen=self.history_window)
            self.baseline_norms[client_id] = 0.0
            self.baseline_directions[client_id] = None
    
    def update_client_gradient(self, 
                              client_id: int, 
                              gradient: np.ndarray):
        """
        Cập nhật gradient history cho một client.
        
        Args:
            client_id: ID của client
            gradient: Gradient mới
        """
        self.initialize_client(client_id)
        
        # Flatten gradient
        g_flat = gradient.flatten()
        
        # Add to history
        self.gradient_history[client_id].append(g_flat)
        
        # Update baseline nếu đủ history
        if len(self.gradient_history[client_id]) >= 3:
            self._update_baseline(client_id)
    
    def compute_heterogeneity_score(self, 
                                    gradients: List[np.ndarray],
                                    client_ids: List[int]) -> float:
        """
        Tính điểm không đồng nhất (Heterogeneity Score) của hệ thống.
        
        H = w1*H_cv + w2*H_sim + w3*H_cluster
        
        Args:
            gradients: List of gradient arrays
            client_ids: List of client IDs
        
        Returns:
            Heterogeneity score ∈ [0, 1]
        """
        n = len(gradients)
        if n < 3:
            return 0.0
        
        # Stack gradients
        grad_matrix = np.vstack([g.flatten() for g in gradients])
        
        # Component 1: Coefficient of Variation (CV) của norms
        norms = np.array([np.linalg.norm(g) for g in grad_matrix])
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        cv = std_norm / (mean_norm + 1e-10)
        H_cv = np.clip(cv, 0, 1)
        
        # Component 2: Variance của pairwise cosine similarities
        similarities = []
        for i in range(n):
            for j in range(i+1, n):
                sim = np.dot(grad_matrix[i], grad_matrix[j]) / (
                    np.linalg.norm(grad_matrix[i]) * np.linalg.norm(grad_matrix[j]) + 1e-10
                )
                similarities.append(sim)
        
        var_sim = np.var(similarities) if similarities else 0
        H_sim = np.clip(var_sim * 2, 0, 1)  # Scale to [0,1]
        
        # Component 3: Cluster spread (sử dụng median distance)
        median_grad = np.median(grad_matrix, axis=0)
        distances = np.array([
            np.linalg.norm(g - median_grad) 
            for g in grad_matrix
        ])
        median_dist = np.median(distances)
        mad_dist = np.median(np.abs(distances - median_dist))
        spread = mad_dist / (median_dist + 1e-10)
        H_cluster = np.clip(spread, 0, 1)
        
        # Weighted combination
        H = (self.cv_weight * H_cv + 
             self.sim_weight * H_sim + 
             self.cluster_weight * H_cluster)
        
        return np.clip(H, 0, 1)
    
    def adjust_threshold(self, 
                        base_threshold: float,
                        heterogeneity: float) -> float:
        """
        Điều chỉnh ngưỡng dựa trên heterogeneity score.
        
        Formula: θ_adj = clip(θ_base + (H - 0.5) * 0.4)
        
        Args:
            base_threshold: Ngưỡng cơ bản
            heterogeneity: Heterogeneity score [0,1]
        
        Returns:
            Adjusted threshold
        """
        # Tăng ngưỡng khi H cao (dữ liệu không đồng nhất)
        # Giảm ngưỡng khi H thấp (dữ liệu đồng nhất)
        adjustment = (heterogeneity - 0.5) * 0.4
        adjusted = base_threshold + adjustment
        
        return np.clip(adjusted, 0.3, 0.95)
    
    def compute_baseline_deviation(self,
                                   client_id: int,
                                   gradient: np.ndarray) -> float:
        """
        Tính độ lệch so với baseline behavior của client.
        
        δ_i = 0.5 * δ_norm + 0.5 * δ_dir
        
        Args:
            client_id: ID của client
            gradient: Gradient hiện tại
        
        Returns:
            Deviation score ∈ [0, 1]
        """
        if client_id not in self.baseline_norms:
            return 0.0
        
        if self.baseline_directions[client_id] is None:
            return 0.0
        
        g_flat = gradient.flatten()
        
        # Norm deviation
        current_norm = np.linalg.norm(g_flat)
        baseline_norm = self.baseline_norms[client_id]
        
        if baseline_norm > 0:
            delta_norm = abs(current_norm - baseline_norm) / baseline_norm
        else:
            delta_norm = 0.0
        
        # Direction deviation (1 - cosine similarity với baseline)
        baseline_dir = self.baseline_directions[client_id]
        cosine_sim = np.dot(g_flat, baseline_dir) / (
            np.linalg.norm(g_flat) * np.linalg.norm(baseline_dir) + 1e-10
        )
        delta_dir = (1 - cosine_sim) / 2  # Scale to [0,1]
        
        # Combined
        deviation = 0.5 * delta_norm + 0.5 * delta_dir
        
        return np.clip(deviation, 0, 1)
    
    def _update_baseline(self, client_id: int):
        """
        Cập nhật baseline behavior cho một client.
        
        Baseline = trung bình của 3 gradients gần nhất.
        """
        history = list(self.gradient_history[client_id])
        
        if len(history) < 3:
            return
        
        # Lấy 3 gradients gần nhất
        recent = history[-3:]
        
        # Baseline norm = median của norms
        norms = [np.linalg.norm(g) for g in recent]
        self.baseline_norms[client_id] = np.median(norms)
        
        # Baseline direction = normalized mean
        mean_grad = np.mean(recent, axis=0)
        self.baseline_directions[client_id] = mean_grad / (np.linalg.norm(mean_grad) + 1e-10)
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        return {
            'num_clients': len(self.gradient_history),
            'clients_with_baseline': sum(
                1 for cid in self.baseline_norms 
                if self.baseline_norms[cid] > 0
            )
        }


# ============================================
# TESTING CODE
# ============================================

def test_noniid_handler():
    """Test Non-IID Handler."""
    print("\n" + "="*70)
    print("🧪 TESTING NON-IID HANDLER")
    print("="*70)
    
    handler = NonIIDHandler()
    
    # Generate synthetic data
    np.random.seed(42)
    n_benign = 18
    n_malicious = 2
    dim = 1000
    
    # Scenario 1: IID data (low heterogeneity)
    print("\n📊 Test 1: IID Data (Low Heterogeneity)")
    iid_grads = [np.random.randn(dim) * 0.1 for _ in range(n_benign)]
    client_ids = list(range(n_benign))
    
    H_iid = handler.compute_heterogeneity_score(iid_grads, client_ids)
    print(f"   H_iid = {H_iid:.3f} (expect < 0.3)")
    
    # Scenario 2: Non-IID data (high heterogeneity)
    print("\n📊 Test 2: Non-IID Data (High Heterogeneity)")
    noniid_grads = []
    for i in range(n_benign):
        # Mỗi client có distribution khác nhau
        scale = 0.1 + i * 0.05
        noniid_grads.append(np.random.randn(dim) * scale)
    
    H_noniid = handler.compute_heterogeneity_score(noniid_grads, client_ids)
    print(f"   H_noniid = {H_noniid:.3f} (expect > 0.5)")
    
    # Scenario 3: Adaptive thresholds
    print("\n📊 Test 3: Adaptive Thresholds")
    base_threshold = 0.7
    
    adjusted_iid = handler.adjust_threshold(base_threshold, H_iid)
    adjusted_noniid = handler.adjust_threshold(base_threshold, H_noniid)
    
    print(f"   Base threshold: {base_threshold:.3f}")
    print(f"   IID adjusted:    {adjusted_iid:.3f} (expect lower)")
    print(f"   Non-IID adjusted: {adjusted_noniid:.3f} (expect higher)")
    
    # Scenario 4: Baseline tracking
    print("\n📊 Test 4: Baseline Tracking")
    
    client_id = 0
    
    # Add 5 consistent gradients
    print("   Adding 5 consistent gradients...")
    for i in range(5):
        grad = np.random.randn(dim) * 0.1
        handler.update_client_gradient(client_id, grad)
    
    # Test with consistent gradient
    consistent_grad = np.random.randn(dim) * 0.1
    dev_consistent = handler.compute_baseline_deviation(client_id, consistent_grad)
    print(f"   Consistent deviation: {dev_consistent:.3f} (expect < 0.3)")
    
    # Test with inconsistent gradient
    inconsistent_grad = np.random.randn(dim) * 2.0
    dev_inconsistent = handler.compute_baseline_deviation(client_id, inconsistent_grad)
    print(f"   Inconsistent deviation: {dev_inconsistent:.3f} (expect > 0.5)")
    
    # Check results
    print("\n" + "="*70)
    if H_noniid > H_iid and adjusted_noniid > adjusted_iid and dev_inconsistent > dev_consistent:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_noniid_handler()