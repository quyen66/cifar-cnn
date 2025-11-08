"""
Two-Stage Filtering
====================
Hard filtering + Soft filtering với reputation scores.

ALL PARAMETERS ARE CONFIGURABLE VIA CONSTRUCTOR (loaded from pyproject.toml)
"""

import numpy as np
from typing import Dict, List, Tuple, Set


class TwoStageFilter:
    """Two-stage filter với configurable parameters."""
    
    def __init__(self,
                 hard_k_threshold: int = 3,
                 soft_reputation_threshold: float = 0.4,
                 soft_distance_multiplier: float = 2.0,
                 soft_enabled: bool = True):
        """
        Initialize Two-Stage Filter with configurable parameters.
        
        Args:
            hard_k_threshold: K threshold for hard filtering
            soft_reputation_threshold: Reputation threshold for soft filtering
            soft_distance_multiplier: Distance multiplier for soft filtering
            soft_enabled: Enable/disable soft filtering
        """
        self.hard_k_threshold = hard_k_threshold
        self.soft_reputation_threshold = soft_reputation_threshold
        self.soft_distance_multiplier = soft_distance_multiplier
        self.soft_enabled = soft_enabled
        
        print(f"✅ TwoStageFilter initialized with params:")
        print(f"   Hard k-threshold: {hard_k_threshold}")
        print(f"   Soft rep-threshold: {soft_reputation_threshold}")
        print(f"   Soft distance-mult: {soft_distance_multiplier}")
        print(f"   Soft enabled: {soft_enabled}")
    
    def filter_clients(self,
                      client_ids: List[int],
                      confidence_scores: Dict[int, float],
                      reputations: Dict[int, float],
                      mode: str,
                      heterogeneity: float) -> Tuple[Set[int], Set[int], Dict]:
        """
        Filter clients using two-stage approach.
        
        Returns:
            (trusted_clients, filtered_clients, stats)
        """
        # Stage 1: Hard filtering
        trusted_stage1, filtered_stage1 = self._hard_filtering(
            client_ids, confidence_scores
        )
        
        # Stage 2: Soft filtering (if enabled)
        if self.soft_enabled and mode in ['ALERT', 'DEFENSE']:
            trusted_stage2, filtered_stage2 = self._soft_filtering(
                trusted_stage1, reputations, heterogeneity
            )
        else:
            trusted_stage2 = trusted_stage1
            filtered_stage2 = set()
        
        # Combine
        trusted = trusted_stage2
        filtered = filtered_stage1 | filtered_stage2
        
        stats = {
            'hard_filtered': len(filtered_stage1),
            'soft_filtered': len(filtered_stage2),
            'total_filtered': len(filtered),
            'trusted': len(trusted)
        }
        
        return trusted, filtered, stats
    
    def _hard_filtering(self,
                       client_ids: List[int],
                       confidence_scores: Dict[int, float]) -> Tuple[Set[int], Set[int]]:
        """Hard filtering based on confidence scores."""
        trusted = set()
        filtered = set()
        
        for cid in client_ids:
            conf = confidence_scores.get(cid, 0.0)
            
            # Use configured k threshold
            if conf >= self.hard_k_threshold:
                filtered.add(cid)
            else:
                trusted.add(cid)
        
        return trusted, filtered
    
    def _soft_filtering(self,
                       trusted_clients: Set[int],
                       reputations: Dict[int, float],
                       heterogeneity: float) -> Tuple[Set[int], Set[int]]:
        """Soft filtering based on reputation."""
        trusted = set()
        filtered = set()
        
        for cid in trusted_clients:
            rep = reputations.get(cid, 0.8)
            
            # Use configured reputation threshold
            if rep < self.soft_reputation_threshold:
                filtered.add(cid)
            else:
                trusted.add(cid)
        
        return trusted, filtered
    
    def get_stats(self) -> Dict:
        """Get filter statistics."""
        return {
            'hard_k_threshold': self.hard_k_threshold,
            'soft_reputation_threshold': self.soft_reputation_threshold,
            'soft_distance_multiplier': self.soft_distance_multiplier,
            'soft_enabled': self.soft_enabled
        }