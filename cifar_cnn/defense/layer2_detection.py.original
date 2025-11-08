"""
Layer 2: Distance + Direction Detection
========================================
Phân tích sâu các gradient chưa bị Layer 1 phát hiện.

Components:
- Distance Check: Euclidean distance to median
- Direction Check: Cosine similarity to median
- Adaptive thresholds based on data distribution

Author: Week 3 Implementation
"""

import numpy as np
from typing import Dict, List, Tuple


class Layer2Detector:
    """
    Layer 2: Deep analysis với Distance + Direction.
    
    Phát hiện các cuộc tấn công tinh vi mà Layer 1 bỏ sót,
    đặc biệt là các tấn công thích ứng (ALIE, Min-Max).
    """
    
    def __init__(self, 
                 distance_multiplier: float = 2.5,
                 cosine_threshold: float = 0.2,
                 warmup_rounds: int = 15):
        """
        Args:
            distance_multiplier: Multiplier cho ngưỡng khoảng cách (default=1.5)
            cosine_threshold: Ngưỡng cosine similarity (default=0.3)
            warmup_rounds: Số rounds warmup với ngưỡng loose hơn (default=15)
        """
        self.distance_multiplier = distance_multiplier
        self.cosine_threshold = cosine_threshold
        self.warmup_rounds = warmup_rounds
        
        print(f"✅ Layer2Detector initialized:")
        print(f"   - Distance multiplier: {distance_multiplier}")
        print(f"   - Cosine threshold: {cosine_threshold}")
        print(f"   - Warmup rounds: {warmup_rounds}")
    
    def detect(self, 
               gradients: List[np.ndarray],
               client_ids: List[int],
               current_round: int,
               layer1_flags: Dict[int, bool] = None) -> Dict[int, bool]:
        """
        Phát hiện malicious clients bằng Distance + Direction.
        
        Args:
            gradients: List of gradient arrays
            client_ids: List of client IDs
            current_round: Current training round
            layer1_flags: Flags from Layer 1 (optional, để skip các client đã bị đánh dấu)
        
        Returns:
            Dictionary mapping client_id -> is_malicious (True/False)
        """
        n = len(gradients)
        
        # Skip nếu quá ít clients
        if n < 3:
            print("   ⚠️  [Layer 2] Too few clients, skipping")
            return {client_ids[i]: False for i in range(n)}
        
        # Determine warmup mode
        is_warmup = current_round <= self.warmup_rounds
        
        # Tính gradient median (reference point)
        grad_matrix = np.vstack([g.flatten() for g in gradients])
        grad_median = np.median(grad_matrix, axis=0)
        
        # Distance Check
        distance_flags = self._distance_check(
            gradients, grad_median, is_warmup=is_warmup
        )
        
        # Direction Check
        direction_flags = self._direction_check(
            gradients, grad_median, is_warmup=is_warmup
        )
        
        # Combine: Flag nếu vi phạm ít nhất 1 trong 2
        layer2_flags = []
        for i in range(n):
            # Nếu Layer 1 đã flag, không cần check lại
            if layer1_flags and layer1_flags.get(client_ids[i], False):
                layer2_flags.append(False)  # Layer 2 không flag nữa
                continue
            
            # Flag nếu distance HOẶC direction đáng ngờ
            flag = distance_flags[i] or direction_flags[i]
            layer2_flags.append(flag)
        
        # Map to client IDs
        detection_results = {
            client_ids[i]: layer2_flags[i] 
            for i in range(n)
        }
        
        # Log
        num_detected = sum(layer2_flags)
        if num_detected > 0:
            detected_ids = [client_ids[i] for i in range(n) if layer2_flags[i]]
            print(f"   🔍 Layer 2 detected {num_detected}/{n} suspicious clients: {detected_ids}")
            
            # Breakdown
            num_distance = sum(distance_flags)
            num_direction = sum(direction_flags)
            print(f"      Distance violations: {num_distance}")
            print(f"      Direction violations: {num_direction}")
        
        return detection_results
    
    def _distance_check(self, 
                       gradients: List[np.ndarray],
                       grad_median: np.ndarray,
                       is_warmup: bool) -> List[bool]:
        """
        Distance Check: Euclidean distance to median.
        
        Formula:
            ||g_i - g_median|| > threshold
            threshold = distance_multiplier × median(||g_j - g_median||)
        
        Returns:
            List of boolean flags
        """
        n = len(gradients)
        
        # Tính khoảng cách tới median
        distances = np.array([
            np.linalg.norm(g.flatten() - grad_median) 
            for g in gradients
        ])
        
        # Tính ngưỡng adaptive
        median_dist = np.median(distances)
        
        # Warmup: loose hơn (2.0×), Normal: strict hơn (1.5×)
        multiplier = 2.0 if is_warmup else self.distance_multiplier
        threshold = multiplier * median_dist
        
        # Flag outliers
        flags = [dist > threshold for dist in distances]
        
        return flags
    
    def _direction_check(self,
                        gradients: List[np.ndarray],
                        grad_median: np.ndarray,
                        is_warmup: bool) -> List[bool]:
        """
        Direction Check: Cosine similarity to median.
        
        Formula:
            cosine_sim(g_i, g_median) < threshold
        
        Cosine similarity = 1 (cùng hướng) → 0 (vuông góc) → -1 (ngược hướng)
        
        FIX: Use adaptive threshold based on distribution
        
        Returns:
            List of boolean flags
        """
        n = len(gradients)
        
        # Tính cosine similarity
        similarities = []
        for g in gradients:
            g_flat = g.flatten()
            
            # Cosine similarity = dot product / (norm × norm)
            sim = np.dot(g_flat, grad_median) / (
                np.linalg.norm(g_flat) * np.linalg.norm(grad_median) + 1e-10
            )
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # FIX: Adaptive threshold based on distribution
        # Use percentile instead of fixed threshold
        if is_warmup:
            # Warmup: only flag extreme outliers (bottom 5%)
            threshold = np.percentile(similarities, 5)
        else:
            # Normal: flag bottom 10% or below absolute threshold
            percentile_threshold = np.percentile(similarities, 10)
            absolute_threshold = self.cosine_threshold
            # Use whichever is stricter
            threshold = min(percentile_threshold, absolute_threshold)
        
        # Flag nếu similarity < threshold (hướng quá khác)
        flags = [sim < threshold for sim in similarities]
        
        return flags
    
    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            'distance_multiplier': self.distance_multiplier,
            'cosine_threshold': self.cosine_threshold,
            'warmup_rounds': self.warmup_rounds
        }