"""
Two-Stage Filtering (FIXED VERSION)
====================================
FIX: Hard filtering + Soft filtering với ADAPTIVE THRESHOLDS.

Thay đổi chính:
1. ✅ Nhận noniid_handler để tính adaptive thresholds
2. ✅ Hard filter dùng threshold thích ứng với H và mode
3. ✅ Soft filter dùng reputation threshold thích ứng

ALL PARAMETERS ARE CONFIGURABLE VIA CONSTRUCTOR (loaded from pyproject.toml)
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional


class TwoStageFilter:
    """Two-stage filter với ADAPTIVE thresholds (FIXED)."""
    
    def __init__(self,
                 hard_k_threshold: int = 3,
                 hard_threshold_min: float = 0.85,  # NEW
                 hard_threshold_max: float = 0.95,  # NEW
                 soft_reputation_threshold: float = 0.4,
                 soft_distance_multiplier: float = 2.0,
                 soft_enabled: bool = True):
        """
        Initialize Two-Stage Filter with configurable parameters.
        
        Args:
            hard_k_threshold: K threshold for hard filtering
            hard_threshold_min: Min value cho hard confidence threshold (từ PDF: 0.85)
            hard_threshold_max: Max value cho hard confidence threshold (từ PDF: 0.95)
            soft_reputation_threshold: Reputation threshold for soft filtering (BASE value)
            soft_distance_multiplier: Distance multiplier for soft filtering
            soft_enabled: Enable/disable soft filtering
        """
        self.hard_k_threshold = hard_k_threshold
        self.hard_threshold_min = hard_threshold_min  
        self.hard_threshold_max = hard_threshold_max 
        self.soft_reputation_threshold = soft_reputation_threshold
        self.soft_distance_multiplier = soft_distance_multiplier
        self.soft_enabled = soft_enabled

        
        print(f"✅ TwoStageFilter initialized (FIXED) with params:")
        print(f"   Hard k-threshold: {hard_k_threshold}")
        print(f"   Hard threshold range: [{hard_threshold_min}, {hard_threshold_max}]")  # NEW
        print(f"   Soft rep-threshold (base): {soft_reputation_threshold}")
        print(f"   Soft distance-mult: {soft_distance_multiplier}")
        print(f"   Soft enabled: {soft_enabled}")
        print(f"   ⚠️  Thresholds will be ADJUSTED based on H and mode")
    
    def filter_clients(self,
                      client_ids: List[int],
                      confidence_scores: Dict[int, float],
                      reputations: Dict[int, float],
                      mode: str,
                      heterogeneity: float,
                      noniid_handler: Optional[object] = None) -> Tuple[Set[int], Set[int], Dict]:
        """
        Filter clients using two-stage approach with ADAPTIVE thresholds.
        
        FIX: Sử dụng noniid_handler để tính adaptive thresholds
        
        Args:
            client_ids: List of client IDs
            confidence_scores: Dict of confidence scores (0-1)
            reputations: Dict of reputation scores (0-1)
            mode: Current mode (NORMAL/ALERT/DEFENSE)
            heterogeneity: H score (0-1)
            noniid_handler: NonIIDHandler instance để tính adaptive thresholds
        
        Returns:
            (trusted_clients, filtered_clients, stats)
        """
        # === FIX: Tính adaptive thresholds ===
        if noniid_handler is not None:
            # Hard filtering threshold
            # Calculate base từ midpoint của range [0.85, 0.95]
            base_hard_threshold = (self.hard_threshold_min + self.hard_threshold_max) / 2.0
            
            # Apply adaptive adjustment based on H
            adaptive_hard_threshold = noniid_handler.get_adaptive_threshold(
                H=heterogeneity,
                mode=mode,
                base_threshold=base_hard_threshold
            )
            
            # Clip to configured range [0.85, 0.95]
            adaptive_hard_threshold = np.clip(
                adaptive_hard_threshold,
                self.hard_threshold_min,
                self.hard_threshold_max
            )
            
            # Soft filtering threshold (reputation - cao hơn khi H cao)
            base_soft_threshold = self.soft_reputation_threshold
            adaptive_soft_threshold = noniid_handler.get_adaptive_threshold(
                H=heterogeneity,
                mode=mode,
                base_threshold=base_soft_threshold
            )
        else:
            # Fallback: dùng midpoint của range
            print("   ⚠️  Warning: noniid_handler not provided, using default thresholds")
            adaptive_hard_threshold = (self.hard_threshold_min + self.hard_threshold_max) / 2.0
            adaptive_soft_threshold = self.soft_reputation_threshold
        
        print(f"   Adaptive thresholds:")
        print(f"      Hard (confidence): {adaptive_hard_threshold:.3f}")
        print(f"      Soft (reputation): {adaptive_soft_threshold:.3f}")
        
        # Stage 1: Hard filtering
        trusted_stage1, filtered_stage1 = self._hard_filtering(
            client_ids, confidence_scores, adaptive_hard_threshold
        )
        
        # Stage 2: Soft filtering (if enabled)
        if self.soft_enabled and mode in ['ALERT', 'DEFENSE']:
            trusted_stage2, filtered_stage2 = self._soft_filtering(
                trusted_stage1, reputations, adaptive_soft_threshold
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
            'trusted': len(trusted),
            'adaptive_hard_threshold': adaptive_hard_threshold,
            'adaptive_soft_threshold': adaptive_soft_threshold
        }
        
        return trusted, filtered, stats
    
    def _hard_filtering(self,
                       client_ids: List[int],
                       confidence_scores: Dict[int, float],
                       threshold: float) -> Tuple[Set[int], Set[int]]:
        """
        Hard filtering based on confidence scores.
        
        FIX: Nhận threshold động từ ngoài
        
        Args:
            client_ids: List of client IDs
            confidence_scores: Dict of confidence scores
            threshold: ADAPTIVE threshold (từ NonIIDHandler)
        
        Returns:
            (trusted_set, filtered_set)
        """
        trusted = set()
        filtered = set()
        
        for cid in client_ids:
            conf = confidence_scores.get(cid, 0.0)
            
            # Dùng threshold thích ứng
            if conf >= threshold:
                filtered.add(cid)
            else:
                trusted.add(cid)
        
        return trusted, filtered
    
    def _soft_filtering(self,
                       trusted_clients: Set[int],
                       reputations: Dict[int, float],
                       threshold: float) -> Tuple[Set[int], Set[int]]:
        """
        Soft filtering based on reputation.
        
        FIX: Nhận threshold động từ ngoài
        
        Args:
            trusted_clients: Set of clients passed hard filter
            reputations: Dict of reputation scores
            threshold: ADAPTIVE reputation threshold
        
        Returns:
            (trusted_set, filtered_set)
        """
        trusted = set()
        filtered = set()
        
        for cid in trusted_clients:
            rep = reputations.get(cid, 0.8)
            
            # Dùng threshold thích ứng
            if rep < threshold:
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
            'soft_enabled': self.soft_enabled,
            'note': 'Base thresholds - actual thresholds are adaptive'
        }