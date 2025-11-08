"""
Mode-Adaptive Aggregation
==========================
Different aggregation methods for different modes.

No configurable parameters - uses algorithms directly.
"""

import numpy as np
from typing import List


def weighted_average_aggregation(gradients: List[np.ndarray],
                                 weights: List[float] = None) -> np.ndarray:
    """Weighted average aggregation (for NORMAL mode)."""
    if weights is None:
        weights = [1.0] * len(gradients)
    
    weighted_sum = np.zeros_like(gradients[0])
    total_weight = sum(weights)
    
    for grad, weight in zip(gradients, weights):
        weighted_sum += grad * weight
    
    return weighted_sum / total_weight


def trimmed_mean_aggregation(gradients: List[np.ndarray],
                             trim_ratio: float = 0.1) -> np.ndarray:
    """Trimmed mean aggregation (for ALERT mode)."""
    grad_matrix = np.array([g.flatten() for g in gradients])
    
    # Trim top and bottom trim_ratio
    lower = int(len(gradients) * trim_ratio)
    upper = int(len(gradients) * (1 - trim_ratio))
    
    sorted_grads = np.sort(grad_matrix, axis=0)
    trimmed = sorted_grads[lower:upper]
    
    result = np.mean(trimmed, axis=0)
    
    return result.reshape(gradients[0].shape)


def coordinate_median_aggregation(gradients: List[np.ndarray]) -> np.ndarray:
    """Coordinate-wise median aggregation (for DEFENSE mode)."""
    grad_matrix = np.array([g.flatten() for g in gradients])
    result = np.median(grad_matrix, axis=0)
    return result.reshape(gradients[0].shape)


def aggregate_by_mode(gradients: List[np.ndarray],
                     mode: str,
                     weights: List[float] = None) -> np.ndarray:
    """
    Select aggregation method based on mode.
    
    Args:
        gradients: List of gradients to aggregate
        mode: 'NORMAL', 'ALERT', or 'DEFENSE'
        weights: Optional weights for weighted average
    
    Returns:
        Aggregated gradient
    """
    if mode == 'NORMAL':
        return weighted_average_aggregation(gradients, weights)
    elif mode == 'ALERT':
        return trimmed_mean_aggregation(gradients, trim_ratio=0.1)
    else:  # DEFENSE
        return coordinate_median_aggregation(gradients)