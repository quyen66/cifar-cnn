"""
Mode Controller
================
Controls system mode (NORMAL/ALERT/DEFENSE) with hysteresis.

ALL PARAMETERS ARE CONFIGURABLE VIA CONSTRUCTOR (loaded from pyproject.toml)
"""
from typing import Dict, List, Optional, Tuple, Set
import numpy as np


class ModeController:
    """Mode controller với configurable parameters."""
    
    def __init__(self,
                 threshold_normal_to_alert: float = 0.20,
                 threshold_alert_to_defense: float = 0.30,
                 hysteresis_normal: float = 0.10,
                 hysteresis_defense: float = 0.15,
                 rep_gate_defense: float = 0.5,
                 initial_mode: str = "NORMAL"):
        """
        Initialize Mode Controller with configurable parameters.
        
        Args:
            threshold_normal_to_alert: ρ threshold to switch NORMAL → ALERT
            threshold_alert_to_defense: ρ threshold to switch ALERT → DEFENSE
            hysteresis_normal: Hysteresis for switching back to NORMAL
            hysteresis_defense: Hysteresis for switching back from DEFENSE
            rep_gate_defense: Reputation gate for DEFENSE mode
            initial_mode: Initial mode
        """
        self.threshold_normal_to_alert = threshold_normal_to_alert
        self.threshold_alert_to_defense = threshold_alert_to_defense
        self.hysteresis_normal = hysteresis_normal
        self.hysteresis_defense = hysteresis_defense
        self.rep_gate_defense = rep_gate_defense
        self.current_mode = initial_mode
        
        # History
        self.mode_history = []
        self.rep_mean_history = []
        
        print(f"✅ ModeController initialized with params:")
        print(f"   Thresholds: N→A={threshold_normal_to_alert}, A→D={threshold_alert_to_defense}")
        print(f"   Hysteresis: normal={hysteresis_normal}, defense={hysteresis_defense}")
        print(f"   Rep gate (DEFENSE): {rep_gate_defense}")
    
    def update_mode(self,
                   threat_ratio: float,
                   detected_clients: List[int],
                   reputations: Dict[int, float],
                   current_round: int) -> str:
        """
        Update system mode based on threat ratio and reputations.
        
        Args:
            threat_ratio: ρ = detected/total
            detected_clients: List of detected client IDs
            reputations: Dict of client reputations
            current_round: Current round
        
        Returns:
            New mode
        """
        # Compute mean reputation
        if reputations:
            rep_mean = np.mean(list(reputations.values()))
        else:
            rep_mean = 0.8
        
        # Mode transition logic
        if self.current_mode == 'NORMAL':
            if threat_ratio > self.threshold_normal_to_alert:
                new_mode = 'ALERT'
            else:
                new_mode = 'NORMAL'
        
        elif self.current_mode == 'ALERT':
            if threat_ratio > self.threshold_alert_to_defense and rep_mean < self.rep_gate_defense:
                new_mode = 'DEFENSE'
            elif threat_ratio < (self.threshold_normal_to_alert - self.hysteresis_normal):
                new_mode = 'NORMAL'
            else:
                new_mode = 'ALERT'
        
        else:  # DEFENSE
            if threat_ratio < (self.threshold_alert_to_defense - self.hysteresis_defense):
                new_mode = 'ALERT'
            else:
                new_mode = 'DEFENSE'
        
        # Update history
        self.current_mode = new_mode
        self.mode_history.append(new_mode)
        self.rep_mean_history.append(rep_mean)
        
        return new_mode
    
    def get_stats(self) -> Dict:
        """Get controller statistics."""
        mode_counts = {}
        for mode in ['NORMAL', 'ALERT', 'DEFENSE']:
            mode_counts[mode] = self.mode_history.count(mode)
        
        return {
            'current_mode': self.current_mode,
            'total_rounds': len(self.mode_history),
            'mode_counts': mode_counts,
            'mean_rep_current': self.rep_mean_history[-1] if self.rep_mean_history else 0,
            'threshold_normal_to_alert': self.threshold_normal_to_alert,
            'threshold_alert_to_defense': self.threshold_alert_to_defense
        }