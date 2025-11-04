"""
Two-Stage Filtering - Week 3
=============================
Lọc clients bằng 2 giai đoạn: Hard filtering và Soft filtering.

Components:
1. Hard Filtering: Loại bỏ ngay clients có confidence score cực cao
2. Soft Filtering: Loại bỏ thêm clients có confidence cao + reputation thấp

Author: Week 3 Implementation
"""

import numpy as np
from typing import Dict, List, Tuple, Set


class TwoStageFilter:
    """
    Two-Stage Filtering với adaptive thresholds.
    
    Philosophy:
    - Stage 1 (Hard): Loại bỏ ngay threats rõ ràng
    - Stage 2 (Soft): Loại bỏ thêm suspicious clients dựa trên reputation
    """
    
    def __init__(self,
                 hard_threshold_base: float = 0.85,
                 soft_threshold_base: float = 0.5,
                 rep_threshold_normal: float = 0.3,
                 rep_threshold_alert: float = 0.5,
                 rep_threshold_defense: float = 0.6):
        """
        Args:
            hard_threshold_base: Ngưỡng cơ bản cho hard filtering (default=0.85)
            soft_threshold_base: Ngưỡng cơ bản cho soft filtering (default=0.5)
            rep_threshold_normal: Reputation threshold ở mode NORMAL (default=0.2)
            rep_threshold_alert: Reputation threshold ở mode ALERT (default=0.4)
            rep_threshold_defense: Reputation threshold ở mode DEFENSE (default=0.6)
        """
        self.hard_threshold_base = hard_threshold_base
        self.soft_threshold_base = soft_threshold_base
        
        self.rep_thresholds = {
            'NORMAL': rep_threshold_normal,
            'ALERT': rep_threshold_alert,
            'DEFENSE': rep_threshold_defense
        }
        
        print(f"✅ TwoStageFilter initialized:")
        print(f"   - Hard threshold: {hard_threshold_base}")
        print(f"   - Soft threshold: {soft_threshold_base}")
        print(f"   - Rep thresholds: N={rep_threshold_normal}, A={rep_threshold_alert}, D={rep_threshold_defense}")
    
    def filter_clients(self,
                      client_ids: List[int],
                      confidence_scores: Dict[int, float],
                      reputations: Dict[int, float],
                      mode: str = 'NORMAL',
                      heterogeneity: float = 0.5) -> Tuple[Set[int], Set[int], Dict]:
        """
        Lọc clients qua 2 giai đoạn.
        
        Args:
            client_ids: List of client IDs
            confidence_scores: Dict mapping client_id -> confidence score [0,1]
            reputations: Dict mapping client_id -> reputation score [0,1]
            mode: Current mode ('NORMAL', 'ALERT', 'DEFENSE')
            heterogeneity: Heterogeneity score [0,1] để điều chỉnh threshold
        
        Returns:
            trusted_clients: Set of client IDs that passed both stages
            filtered_clients: Set of client IDs that were filtered
            filter_stats: Dictionary with filtering statistics
        """
        # Adjust thresholds based on heterogeneity
        hard_threshold = self._adjust_threshold(
            self.hard_threshold_base, 
            heterogeneity,
            increase=True  # Tăng threshold khi Non-IID cao
        )
        
        soft_threshold = self._adjust_threshold(
            self.soft_threshold_base,
            heterogeneity,
            increase=True
        )
        
        # Get reputation threshold cho mode hiện tại
        rep_threshold = self.rep_thresholds.get(mode, 0.4)
        
        # Stage 1: Hard Filtering
        hard_filtered = set()
        for cid in client_ids:
            conf = confidence_scores.get(cid, 0.0)
            if conf > hard_threshold:
                hard_filtered.add(cid)
        
        # Remaining clients after stage 1
        remaining = set(client_ids) - hard_filtered
        
        # Stage 2: Soft Filtering (reputation-based)
        soft_filtered = set()
        for cid in remaining:
            conf = confidence_scores.get(cid, 0.0)
            rep = reputations.get(cid, 0.8)  # Default high rep if missing
            
            # Filter nếu: confidence cao VÀ reputation thấp
            if conf > soft_threshold and rep < rep_threshold:
                soft_filtered.add(cid)
        
        # Final trusted clients
        trusted = remaining - soft_filtered
        filtered = hard_filtered | soft_filtered
        
        # Statistics
        stats = {
            'total_clients': len(client_ids),
            'hard_filtered': len(hard_filtered),
            'soft_filtered': len(soft_filtered),
            'total_filtered': len(filtered),
            'trusted': len(trusted),
            'hard_threshold': hard_threshold,
            'soft_threshold': soft_threshold,
            'rep_threshold': rep_threshold,
            'mode': mode
        }
        
        return trusted, filtered, stats
    
    def _adjust_threshold(self,
                         base: float,
                         heterogeneity: float,
                         increase: bool = True) -> float:
        """
        Điều chỉnh threshold dựa trên heterogeneity.
        
        Args:
            base: Base threshold
            heterogeneity: Heterogeneity score [0,1]
            increase: True = tăng threshold khi H cao, False = giảm
        
        Returns:
            Adjusted threshold
        """
        adjustment = (heterogeneity - 0.5) * 0.2
        
        if increase:
            adjusted = base + adjustment
        else:
            adjusted = base - adjustment
        
        return np.clip(adjusted, 0.3, 0.95)
    
    def get_stats(self) -> Dict:
        """Get filter configuration."""
        return {
            'hard_threshold_base': self.hard_threshold_base,
            'soft_threshold_base': self.soft_threshold_base,
            'rep_thresholds': self.rep_thresholds
        }