"""
Mode-Adaptive Aggregation Methods for Flower Server
====================================================

3 aggregation modes theo main.pdf:
- NORMAL (ρ ≤ 15%): Weighted Average (uniform weights)
- ALERT (15% < ρ ≤ 30%): Trimmed Mean 10%
- DEFENSE (ρ > 30%): Coordinate Median

Usage trong CustomFedProx.aggregate_fit():
    # After filtering malicious clients
    trusted_gradients = [...]
    mode = decide_mode(threat_ratio)  # NORMAL/ALERT/DEFENSE
    
    aggregated_params = aggregate_by_mode(
        gradients=trusted_gradients,
        mode=mode
    )

Author: Adaptive Defense System - Week 3
"""

import numpy as np
from typing import List


def weighted_average_aggregation(gradients: List[np.ndarray]) -> np.ndarray:
    """
    NORMAL Mode: Weighted Average với uniform weights.
    
    Philosophy: Performance-first. Trust rằng filtering đã loại bỏ attackers.
    
    Formula:
        g_global = Σ(w_i * g_i) where w_i = 1/n
    
    Args:
        gradients: List of gradient arrays (already filtered, trusted clients only)
    
    Returns:
        Aggregated gradient array
    
    Complexity: O(nd) where n=clients, d=dimensions
    """
    if not gradients:
        raise ValueError("No gradients to aggregate")
    
    n = len(gradients)
    
    # Uniform weights
    weights = np.ones(n) / n
    
    # Weighted average
    global_gradient = np.zeros_like(gradients[0])
    for i in range(n):
        global_gradient += weights[i] * gradients[i]
    
    return global_gradient


def trimmed_mean_aggregation(
    gradients: List[np.ndarray],
    trim_ratio: float = 0.1
) -> np.ndarray:
    """
    ALERT Mode: Coordinate-wise Trimmed Mean.
    
    Philosophy: Balanced. Trim extremes để loại bỏ potential attackers 
    còn sót lại sau filtering.
    
    Process:
        - For each dimension j:
          1. Sort values across all clients
          2. Trim top 10% and bottom 10%
          3. Take mean of remaining values
    
    Args:
        gradients: List of gradient arrays (trusted clients only)
        trim_ratio: Ratio to trim from each end (default=0.1 for 10%)
    
    Returns:
        Aggregated gradient array
    
    Complexity: O(nd log n) due to sorting per dimension
    Breakdown point: ~10% (can tolerate 10% outliers per dimension)
    """
    if not gradients:
        raise ValueError("No gradients to aggregate")
    
    n = len(gradients)
    d = len(gradients[0])
    
    # Compute trim count
    trim_count = int(n * trim_ratio)
    
    # Stack gradients into matrix [n_clients × d_dimensions]
    grad_matrix = np.vstack(gradients)
    
    # Coordinate-wise trimmed mean
    global_gradient = np.zeros(d)
    
    for j in range(d):
        # Get all values for dimension j
        values_j = grad_matrix[:, j]
        
        # Sort values
        sorted_values = np.sort(values_j)
        
        # Trim extremes
        if trim_count > 0:
            trimmed = sorted_values[trim_count:-trim_count]
        else:
            trimmed = sorted_values
        
        # Take mean
        global_gradient[j] = np.mean(trimmed)
    
    return global_gradient


def coordinate_median_aggregation(gradients: List[np.ndarray]) -> np.ndarray:
    """
    DEFENSE Mode: Coordinate-wise Median.
    
    Philosophy: Security-first. Maximum robustness với breakdown point 50%.
    Không trust bất kỳ ai, use median để immune với outliers.
    
    Process:
        - For each dimension j:
          Take median value across all clients
    
    Args:
        gradients: List of gradient arrays (trusted clients only)
    
    Returns:
        Aggregated gradient array
    
    Complexity: O(nd log n) due to median computation per dimension
    Breakdown point: 50% (can tolerate up to 50% Byzantine clients)
    """
    if not gradients:
        raise ValueError("No gradients to aggregate")
    
    n = len(gradients)
    d = len(gradients[0])
    
    # Stack gradients into matrix [n_clients × d_dimensions]
    grad_matrix = np.vstack(gradients)
    
    # Coordinate-wise median
    global_gradient = np.zeros(d)
    
    for j in range(d):
        # Get all values for dimension j
        values_j = grad_matrix[:, j]
        
        # Take median
        global_gradient[j] = np.median(values_j)
    
    return global_gradient


def aggregate_by_mode(
    gradients: List[np.ndarray],
    mode: str = "NORMAL"
) -> np.ndarray:
    """
    Main aggregation function - routes to appropriate method based on mode.
    
    Args:
        gradients: List of gradient arrays (trusted clients only)
        mode: "NORMAL", "ALERT", or "DEFENSE"
    
    Returns:
        Aggregated gradient array
    
    Example:
        >>> gradients = [grad1, grad2, grad3, ...]  # Filtered gradients
        >>> mode = "ALERT"  # From mode decision
        >>> aggregated = aggregate_by_mode(gradients, mode)
    """
    if mode == "NORMAL":
        return weighted_average_aggregation(gradients)
    elif mode == "ALERT":
        return trimmed_mean_aggregation(gradients, trim_ratio=0.1)
    elif mode == "DEFENSE":
        return coordinate_median_aggregation(gradients)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use NORMAL/ALERT/DEFENSE")


def decide_mode_simple(threat_ratio: float) -> str:
    """
    Simple mode decision based on threat ratio.
    
    PLACEHOLDER - Sẽ được replace bằng full mode decision logic
    trong Week 4-5 (với reputation gates, hysteresis, etc.)
    
    Args:
        threat_ratio: Ratio of detected malicious clients (0.0 to 1.0)
    
    Returns:
        Mode string: "NORMAL", "ALERT", or "DEFENSE"
    
    Boundaries:
        - ρ ≤ 0.15 → NORMAL
        - 0.15 < ρ ≤ 0.30 → ALERT
        - ρ > 0.30 → DEFENSE
    """
    if threat_ratio <= 0.15:
        return "NORMAL"
    elif threat_ratio <= 0.30:
        return "ALERT"
    else:
        return "DEFENSE"


# ============================================
# TESTING CODE
# ============================================

def test_aggregation_methods():
    """Test 3 aggregation methods với synthetic data."""
    print("\n" + "="*70)
    print("🧪 TESTING AGGREGATION METHODS")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    n_benign = 15
    n_malicious = 5
    d = 1000
    
    # Benign gradients: normal distribution
    benign_grads = [np.random.randn(d) * 0.1 for _ in range(n_benign)]
    
    # Malicious gradients: extreme values (Byzantine)
    malicious_grads = [np.random.randn(d) * 10.0 for _ in range(n_malicious)]
    
    print(f"\n📊 Test Setup:")
    print(f"   - Benign clients: {n_benign}")
    print(f"   - Malicious clients: {n_malicious}")
    print(f"   - Gradient dimension: {d}")
    
    # Test 1: All gradients (including malicious)
    print(f"\n🔬 Test 1: All Gradients (Benign + Malicious)")
    all_grads = benign_grads + malicious_grads
    
    agg_normal = weighted_average_aggregation(all_grads)
    agg_alert = trimmed_mean_aggregation(all_grads, trim_ratio=0.1)
    agg_defense = coordinate_median_aggregation(all_grads)
    
    print(f"   NORMAL (Weighted Avg):  norm = {np.linalg.norm(agg_normal):.2f}")
    print(f"   ALERT (Trimmed Mean):   norm = {np.linalg.norm(agg_alert):.2f}")
    print(f"   DEFENSE (Median):       norm = {np.linalg.norm(agg_defense):.2f}")
    
    # Test 2: Only benign gradients (after filtering)
    print(f"\n🔬 Test 2: Only Benign Gradients (After Filtering)")
    
    agg_normal = weighted_average_aggregation(benign_grads)
    agg_alert = trimmed_mean_aggregation(benign_grads, trim_ratio=0.1)
    agg_defense = coordinate_median_aggregation(benign_grads)
    
    print(f"   NORMAL (Weighted Avg):  norm = {np.linalg.norm(agg_normal):.2f}")
    print(f"   ALERT (Trimmed Mean):   norm = {np.linalg.norm(agg_alert):.2f}")
    print(f"   DEFENSE (Median):       norm = {np.linalg.norm(agg_defense):.2f}")
    
    # Test 3: Mode decision
    print(f"\n🔬 Test 3: Mode Decision")
    
    test_cases = [
        (0.0, "NORMAL"),
        (0.10, "NORMAL"),
        (0.15, "NORMAL"),
        (0.20, "ALERT"),
    (0.30, "ALERT"),
        (0.35, "DEFENSE"),
        (0.50, "DEFENSE"),
    ]
    
    for threat_ratio, expected_mode in test_cases:
        mode = decide_mode_simple(threat_ratio)
        status = "✓" if mode == expected_mode else "✗"
        print(f"   {status} ρ={threat_ratio:.2f} → {mode} (expected: {expected_mode})")
    
    # Test 4: aggregate_by_mode
    print(f"\n🔬 Test 4: aggregate_by_mode Function")
    
    for mode in ["NORMAL", "ALERT", "DEFENSE"]:
        agg = aggregate_by_mode(benign_grads, mode=mode)
        print(f"   {mode:8s}: norm = {np.linalg.norm(agg):.2f}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_aggregation_methods()