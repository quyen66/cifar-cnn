# cifar_cnn/defense/aggregation.py

import numpy as np
from typing import List, Optional
from logging import INFO
from flwr.common.logger import log

def weighted_average_aggregation(
    gradients: List[np.ndarray], 
    reputations: Optional[List[float]] = None, 
    epsilon: float = 1e-6
) -> np.ndarray:
    """Safe Weighted Average Aggregation."""
    grads_stack = np.stack(gradients)
    
    if reputations is None:
        return np.mean(grads_stack, axis=0)
    
    total_reputation = sum(reputations)
    
    # Fail-safe check
    if total_reputation <= epsilon:
        log(INFO, f"⚠️ [Safe Aggregation] Total reputation ({total_reputation:.6f}) <= epsilon. Fallback to Simple Mean.")
        return np.mean(grads_stack, axis=0)
    
    weights = np.array(reputations) / total_reputation
    weighted_grads = grads_stack * weights[:, np.newaxis]
    return np.sum(weighted_grads, axis=0)

def trimmed_mean_aggregation(gradients: List[np.ndarray], beta: float = 0.1) -> np.ndarray:
    """
    Trimmed Mean Aggregation.
    Loại bỏ beta% giá trị lớn nhất và beta% giá trị nhỏ nhất trên mỗi chiều.
    """
    grads_stack = np.stack(gradients)
    n = grads_stack.shape[0]
    k = int(n * beta)
    
    # Nếu k quá lớn (ví dụ > 40%), giữ lại ít nhất 1 phần tử ở giữa
    if k >= n // 2:
        k = (n - 1) // 2
    
    if k == 0:
        return np.mean(grads_stack, axis=0)
    
    # Sort dọc theo trục client
    sorted_grads = np.sort(grads_stack, axis=0)
    
    # Cắt bỏ k phần tử đầu và cuối
    trimmed = sorted_grads[k : n - k]
    return np.mean(trimmed, axis=0)

def coordinate_median_aggregation(gradients: List[np.ndarray]) -> np.ndarray:
    """Coordinate-wise Median Aggregation."""
    grads_stack = np.stack(gradients)
    return np.median(grads_stack, axis=0)

def aggregate_by_mode(
    gradients: List[np.ndarray], 
    mode: str, 
    reputations: Optional[List[float]] = None,
    threat_level: float = 0.0,  # <--- [FIX] Thêm tham số rho
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Hàm điều phối aggregation dựa trên mode và threat_level.
    """
    if not gradients:
        return None
        
    if mode == "NORMAL":
        return weighted_average_aggregation(gradients, reputations, epsilon)
    
    elif mode == "ALERT":
        # [FIX] Dynamic Beta cho Trimmed Mean
        # Beta tối thiểu là 0.1 (10%), tối đa theo rho thực tế
        # Ví dụ: rho=0.25 (25% tấn công) -> cắt 25% biên
        dynamic_beta = max(0.1, threat_level)
        
        # Log để kiểm tra (chỉ hiện khi debug hoặc nếu cần)
        # log(INFO, f"   [Agg] ALERT mode: Using Trimmed Mean with beta={dynamic_beta:.2f} (rho={threat_level:.2f})")
        
        return trimmed_mean_aggregation(gradients, beta=dynamic_beta)
    
    elif mode == "DEFENSE":
        return coordinate_median_aggregation(gradients)
    
    else:
        # Default fallback
        return weighted_average_aggregation(gradients, reputations, epsilon)