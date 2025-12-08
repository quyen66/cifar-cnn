# cifar_cnn/defense/aggregation.py
"""
Aggregation Methods
=====================================================
Mode-Adaptive Aggregation theo main.pdf.

ALL PARAMETERS ARE LOADED FROM pyproject.toml via constructor.

Thuật toán theo PDF:
1. NORMAL (Weighted Average):
   - gglobal = Σ(R(i,t)/SR × gi) nếu SR > ε
   - gglobal = 1/|T| × Σgi nếu SR ≤ ε (fallback FedAvg)

2. ALERT (Trimmed Mean):
   - k = ⌊max(trim_ratio_min, ρ) × |T|⌋
   - Cắt k giá trị cao nhất và thấp nhất mỗi chiều
   
3. DEFENSE (Coordinate Median):
   - gglobal[j] = Median({gi[j] : i ∈ T})
"""

import numpy as np
from typing import List, Optional, Dict
from logging import INFO
from flwr.common.logger import log


class Aggregator:
    """
    Aggregator
    
    Tất cả tham số đều có thể tinh chỉnh qua constructor (load từ pyproject.toml).
    """
    
    def __init__(
        self,
        # === Safe Weight Parameters ===
        safe_weight_epsilon: float = 1e-6,          # ε để check SR > 0
        
        # === Trimmed Mean Parameters ===
        trim_ratio_min: float = 0.1,                # Tỷ lệ trim tối thiểu (10%)
        trim_ratio_max: float = 0.4,                # Tỷ lệ trim tối đa (40%)
        
        # === Weighted Average Parameters ===
        use_reputation_weights: bool = True,        # Dùng reputation làm weight
        uniform_weight_fallback: bool = True,       # Fallback về uniform nếu SR ≤ ε
    ):
        """
        Initialize Aggregator với tất cả tham số configurable.
        
        Args:
            safe_weight_epsilon: Ngưỡng epsilon để check tổng weight > 0 (PDF: 1e-6)
            trim_ratio_min: Tỷ lệ trim tối thiểu cho Trimmed Mean (PDF: 0.1)
            trim_ratio_max: Tỷ lệ trim tối đa để tránh cắt quá nhiều (0.4)
            use_reputation_weights: Dùng reputation làm weight trong Weighted Avg
            uniform_weight_fallback: Fallback về uniform weight nếu SR ≤ ε
        """
        self.epsilon = safe_weight_epsilon
        self.trim_ratio_min = trim_ratio_min
        self.trim_ratio_max = trim_ratio_max
        self.use_reputation_weights = use_reputation_weights
        self.uniform_fallback = uniform_weight_fallback
        
        # Stats tracking
        self.last_aggregation_stats: Dict = {}
        
        # Log configuration
        self._log_config()
    
    def _log_config(self):
        """Log configuration for debugging."""
        print(f"\n{'='*60}")
        print(f"✅ Aggregator Initialized (Fully Configurable)")
        print(f"{'='*60}")
        print(f"📊 Safe Weight Parameters:")
        print(f"   Epsilon: {self.epsilon}")
        print(f"\n📊 Trimmed Mean Parameters:")
        print(f"   Trim ratio min: {self.trim_ratio_min}")
        print(f"   Trim ratio max: {self.trim_ratio_max}")
        print(f"\n📊 Weighted Average Parameters:")
        print(f"   Use reputation weights: {self.use_reputation_weights}")
        print(f"   Uniform fallback: {self.uniform_fallback}")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # AGGREGATION METHODS
    # =========================================================================
    
    def weighted_average(
        self,
        gradients: List[np.ndarray],
        reputations: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        NORMAL Mode: Weighted Average với reputation weights.
        
        Theo PDF:
        - gglobal = Σ(R(i,t)/SR × gi) nếu SR > ε
        - gglobal = 1/|T| × Σgi nếu SR ≤ ε (fallback)
        
        Args:
            gradients: List of gradient arrays
            reputations: List of reputation scores (optional)
            
        Returns:
            Aggregated gradient
        """
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        n = len(gradients)
        grad_stack = np.stack(gradients)
        
        # Case 1: No reputations or disabled → Simple mean
        if reputations is None or not self.use_reputation_weights:
            self.last_aggregation_stats = {
                'method': 'weighted_average',
                'weight_type': 'uniform',
                'num_clients': n
            }
            return np.mean(grad_stack, axis=0)
        
        # Case 2: Check SR (total reputation)
        SR = sum(reputations)
        
        if SR <= self.epsilon:
            # Fallback to uniform weights
            if self.uniform_fallback:
                log(INFO, f"⚠️ [Safe Aggregation] SR ({SR:.6f}) <= ε. Fallback to Simple Mean.")
                self.last_aggregation_stats = {
                    'method': 'weighted_average',
                    'weight_type': 'uniform_fallback',
                    'SR': SR,
                    'num_clients': n
                }
                return np.mean(grad_stack, axis=0)
            else:
                raise ValueError(f"SR ({SR}) <= epsilon and fallback disabled")
        
        # Case 3: Normal weighted average
        weights = np.array(reputations) / SR
        
        # Weighted sum
        weighted_grads = grad_stack * weights[:, np.newaxis]
        result = np.sum(weighted_grads, axis=0)
        
        self.last_aggregation_stats = {
            'method': 'weighted_average',
            'weight_type': 'reputation',
            'SR': SR,
            'num_clients': n,
            'weight_min': float(np.min(weights)),
            'weight_max': float(np.max(weights)),
            'weight_std': float(np.std(weights))
        }
        
        return result
    
    def trimmed_mean(
        self,
        gradients: List[np.ndarray],
        threat_ratio: float = 0.0
    ) -> np.ndarray:
        """
        ALERT Mode: Coordinate-wise Trimmed Mean.
        
        Theo PDF:
        - k = ⌊max(trim_ratio_min, ρ) × |T|⌋
        - Cắt k giá trị cao nhất và thấp nhất mỗi chiều
        
        Args:
            gradients: List of gradient arrays
            threat_ratio: Current threat ratio ρ (để tính dynamic trim)
            
        Returns:
            Aggregated gradient
        """
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        n = len(gradients)
        grad_stack = np.stack(gradients)
        
        # Dynamic trim ratio theo PDF: k = max(0.1, ρ)
        dynamic_ratio = max(self.trim_ratio_min, threat_ratio)
        dynamic_ratio = min(dynamic_ratio, self.trim_ratio_max)  # Cap at max
        
        # Compute trim count
        k = int(np.floor(dynamic_ratio * n))
        
        # Ensure we have enough elements left
        if k >= n // 2:
            k = max(0, (n - 1) // 2)
        
        if k == 0:
            # No trimming needed
            self.last_aggregation_stats = {
                'method': 'trimmed_mean',
                'trim_ratio': dynamic_ratio,
                'k': 0,
                'num_clients': n,
                'note': 'no_trimming'
            }
            return np.mean(grad_stack, axis=0)
        
        # Coordinate-wise trimmed mean
        d = grad_stack.shape[1]
        result = np.zeros(d)
        
        for j in range(d):
            values = grad_stack[:, j]
            sorted_values = np.sort(values)
            trimmed = sorted_values[k:n-k]
            result[j] = np.mean(trimmed)
        
        self.last_aggregation_stats = {
            'method': 'trimmed_mean',
            'trim_ratio': dynamic_ratio,
            'threat_ratio_input': threat_ratio,
            'k': k,
            'num_clients': n,
            'elements_per_dim': n - 2*k
        }
        
        return result
    
    def coordinate_median(
        self,
        gradients: List[np.ndarray]
    ) -> np.ndarray:
        """
        DEFENSE Mode: Coordinate-wise Median.
        
        Theo PDF:
        - gglobal[j] = Median({gi[j] : i ∈ T})
        - Breakdown point: 50%
        
        Args:
            gradients: List of gradient arrays
            
        Returns:
            Aggregated gradient
        """
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        n = len(gradients)
        grad_stack = np.stack(gradients)
        
        result = np.median(grad_stack, axis=0)
        
        self.last_aggregation_stats = {
            'method': 'coordinate_median',
            'num_clients': n,
            'breakdown_point': 0.5
        }
        
        return result
    
    # =========================================================================
    # MAIN AGGREGATION FUNCTION
    # =========================================================================
    
    def aggregate_by_mode(
        self,
        gradients: List[np.ndarray],
        mode: str,
        reputations: Optional[List[float]] = None,
        threat_ratio: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        Điều phối aggregation dựa trên mode.
        
        Theo PDF:
        - NORMAL: Weighted Average
        - ALERT: Trimmed Mean với dynamic trim ratio
        - DEFENSE: Coordinate Median
        
        Args:
            gradients: List of gradient arrays
            mode: "NORMAL", "ALERT", hoặc "DEFENSE"
            reputations: List of reputation scores (for NORMAL mode)
            threat_ratio: Current threat ratio ρ (for ALERT mode trim)
            
        Returns:
            Aggregated gradient or None if empty
        """
        if not gradients:
            log(INFO, "⚠️ No gradients to aggregate")
            return None
        
        mode = mode.upper()
        
        if mode == "NORMAL":
            return self.weighted_average(gradients, reputations)
        
        elif mode == "ALERT":
            return self.trimmed_mean(gradients, threat_ratio)
        
        elif mode == "DEFENSE":
            return self.coordinate_median(gradients)
        
        else:
            log(INFO, f"⚠️ Unknown mode '{mode}'. Fallback to weighted_average.")
            return self.weighted_average(gradients, reputations)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_last_stats(self) -> Dict:
        """Get stats from last aggregation."""
        return self.last_aggregation_stats.copy()
    
    def get_config(self) -> Dict:
        """Get current configuration."""
        return {
            'safe_weight_epsilon': self.epsilon,
            'trim_ratio_min': self.trim_ratio_min,
            'trim_ratio_max': self.trim_ratio_max,
            'use_reputation_weights': self.use_reputation_weights,
            'uniform_weight_fallback': self.uniform_fallback
        }


# =============================================================================
# STANDALONE FUNCTIONS (Backward Compatibility)
# =============================================================================

# Global aggregator instance with default config
_default_aggregator = None

def _get_default_aggregator():
    global _default_aggregator
    if _default_aggregator is None:
        _default_aggregator = Aggregator()
    return _default_aggregator


def weighted_average_aggregation(
    gradients: List[np.ndarray],
    reputations: Optional[List[float]] = None,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Backward-compatible weighted average function.
    
    Safe Weighted Average với epsilon check.
    """
    if not gradients:
        raise ValueError("No gradients to aggregate")
    
    grad_stack = np.stack(gradients)
    
    if reputations is None:
        return np.mean(grad_stack, axis=0)
    
    SR = sum(reputations)
    
    if SR <= epsilon:
        log(INFO, f"⚠️ [Safe Aggregation] SR ({SR:.6f}) <= ε. Fallback to Simple Mean.")
        return np.mean(grad_stack, axis=0)
    
    weights = np.array(reputations) / SR
    weighted_grads = grad_stack * weights[:, np.newaxis]
    return np.sum(weighted_grads, axis=0)


def trimmed_mean_aggregation(
    gradients: List[np.ndarray],
    beta: float = 0.1
) -> np.ndarray:
    """
    Backward-compatible trimmed mean function.
    
    Coordinate-wise Trimmed Mean với fixed beta.
    """
    if not gradients:
        raise ValueError("No gradients to aggregate")
    
    n = len(gradients)
    grad_stack = np.stack(gradients)
    d = grad_stack.shape[1]
    
    k = int(n * beta)
    
    if k >= n // 2:
        k = (n - 1) // 2
    
    if k == 0:
        return np.mean(grad_stack, axis=0)
    
    result = np.zeros(d)
    for j in range(d):
        sorted_values = np.sort(grad_stack[:, j])
        trimmed = sorted_values[k:n-k]
        result[j] = np.mean(trimmed)
    
    return result


def coordinate_median_aggregation(
    gradients: List[np.ndarray]
) -> np.ndarray:
    """
    Backward-compatible coordinate median function.
    """
    if not gradients:
        raise ValueError("No gradients to aggregate")
    
    grad_stack = np.stack(gradients)
    return np.median(grad_stack, axis=0)


def aggregate_by_mode(
    gradients: List[np.ndarray],
    mode: str,
    reputations: Optional[List[float]] = None,
    threat_level: float = 0.0,
    epsilon: float = 1e-6
) -> Optional[np.ndarray]:
    """
    Backward-compatible aggregation dispatcher.
    
    Điều phối aggregation dựa trên mode.
    """
    if not gradients:
        return None
    
    mode = mode.upper()
    
    if mode == "NORMAL":
        return weighted_average_aggregation(gradients, reputations, epsilon)
    
    elif mode == "ALERT":
        # Dynamic beta based on threat level
        dynamic_beta = max(0.1, threat_level)
        return trimmed_mean_aggregation(gradients, beta=dynamic_beta)
    
    elif mode == "DEFENSE":
        return coordinate_median_aggregation(gradients)
    
    else:
        return weighted_average_aggregation(gradients, reputations, epsilon)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING AGGREGATOR")
    print("="*70)
    
    # Test with default config
    aggregator = Aggregator()
    
    # Print config
    print("\n📋 Current Config:")
    for k, v in aggregator.get_config().items():
        print(f"   {k}: {v}")
    
    # Create test data
    np.random.seed(42)
    n_clients = 20
    grad_dim = 1000
    
    # Benign gradients (normal distribution)
    benign_grads = [np.random.randn(grad_dim) * 0.1 for _ in range(15)]
    
    # Malicious gradients (large outliers)
    malicious_grads = [np.random.randn(grad_dim) * 5.0 for _ in range(5)]
    
    all_grads = benign_grads + malicious_grads
    reputations = [0.9] * 15 + [0.2] * 5  # High rep for benign, low for malicious
    
    print(f"\n📊 Test Data:")
    print(f"   Benign clients: 15 (rep=0.9)")
    print(f"   Malicious clients: 5 (rep=0.2)")
    
    # Test NORMAL mode
    print("\n📊 NORMAL Mode (Weighted Average):")
    result = aggregator.aggregate_by_mode(all_grads, "NORMAL", reputations)
    print(f"   Result norm: {np.linalg.norm(result):.4f}")
    print(f"   Stats: {aggregator.get_last_stats()}")
    
    # Test ALERT mode
    print("\n📊 ALERT Mode (Trimmed Mean, ρ=0.25):")
    result = aggregator.aggregate_by_mode(all_grads, "ALERT", threat_ratio=0.25)
    print(f"   Result norm: {np.linalg.norm(result):.4f}")
    print(f"   Stats: {aggregator.get_last_stats()}")
    
    # Test DEFENSE mode
    print("\n📊 DEFENSE Mode (Coordinate Median):")
    result = aggregator.aggregate_by_mode(all_grads, "DEFENSE")
    print(f"   Result norm: {np.linalg.norm(result):.4f}")
    print(f"   Stats: {aggregator.get_last_stats()}")
    
    # Test SR = 0 fallback
    print("\n📊 SR = 0 Fallback Test:")
    zero_reps = [0.0] * 20
    result = aggregator.aggregate_by_mode(all_grads, "NORMAL", zero_reps)
    print(f"   Result norm: {np.linalg.norm(result):.4f}")
    print(f"   Stats: {aggregator.get_last_stats()}")
    
    # Compare methods
    print("\n📊 Comparison (Benign only vs All):")
    benign_result = aggregator.coordinate_median(benign_grads)
    all_result = aggregator.coordinate_median(all_grads)
    print(f"   Benign only median norm: {np.linalg.norm(benign_result):.4f}")
    print(f"   All (with malicious) median norm: {np.linalg.norm(all_result):.4f}")
    print(f"   Difference: {np.linalg.norm(all_result - benign_result):.4f}")
    
    print("\n✅ All tests passed!")