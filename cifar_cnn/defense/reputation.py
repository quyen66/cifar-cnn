"""
Reputation System
==================
Asymmetric EMA updates với floor lifting.

ALL PARAMETERS ARE CONFIGURABLE VIA CONSTRUCTOR (loaded from pyproject.toml)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Set



class ReputationSystem:
    """Reputation system với configurable parameters."""
    
    def __init__(self,
                 ema_alpha_increase: float = 0.4,
                 ema_alpha_decrease: float = 0.2,
                 penalty_flagged: float = 0.2,
                 penalty_variance: float = 0.1,
                 reward_clean: float = 0.1,
                 floor_lift_threshold: float = 0.4,
                 floor_lift_amount: float = 0.2,
                 initial_reputation: float = 0.8):
        """
        Initialize Reputation System with configurable parameters.
        
        Args:
            ema_alpha_increase: EMA alpha for reputation increase (clean behavior)
            ema_alpha_decrease: EMA alpha for reputation decrease (malicious behavior)
            penalty_flagged: Penalty when client is flagged
            penalty_variance: Penalty for high gradient variance
            reward_clean: Reward for clean behavior
            floor_lift_threshold: Threshold to trigger floor lifting
            floor_lift_amount: Amount to lift floor
            initial_reputation: Initial reputation for new clients
        """
        self.ema_alpha_increase = ema_alpha_increase
        self.ema_alpha_decrease = ema_alpha_decrease
        self.penalty_flagged = penalty_flagged
        self.penalty_variance = penalty_variance
        self.reward_clean = reward_clean
        self.floor_lift_threshold = floor_lift_threshold
        self.floor_lift_amount = floor_lift_amount
        self.initial_reputation = initial_reputation
        
        # Client reputations
        self.reputations = {}
        
        print(f"✅ ReputationSystem initialized with params:")
        print(f"   EMA alphas: increase={ema_alpha_increase}, decrease={ema_alpha_decrease}")
        print(f"   Penalties: flagged={penalty_flagged}, variance={penalty_variance}")
        print(f"   Reward: {reward_clean}")
        print(f"   Floor lift: threshold={floor_lift_threshold}, amount={floor_lift_amount}")
    
    def initialize_client(self, client_id: int):
        """Initialize reputation for a new client."""
        if client_id not in self.reputations:
            self.reputations[client_id] = self.initial_reputation
    
    def update(self,
               client_id: int,
               gradient: np.ndarray,
               grad_median: np.ndarray,
               was_flagged: bool,
               current_round: int) -> float:
        """
        Update reputation using asymmetric EMA.
        
        Returns:
            Updated reputation
        """
        # Initialize if needed
        self.initialize_client(client_id)
        
        current_rep = self.reputations[client_id]
        
        # Compute penalties/rewards
        if was_flagged:
            # Penalty for being flagged
            delta = -self.penalty_flagged
            alpha = self.ema_alpha_decrease  # Faster decrease
        else:
            # Reward for clean behavior
            delta = self.reward_clean
            alpha = self.ema_alpha_increase  # Slower increase
        
        # Additional penalty for high variance
        variance_penalty = self._compute_variance_penalty(gradient, grad_median)
        delta -= variance_penalty
        
        # Asymmetric EMA update
        new_rep = current_rep + alpha * delta
        
        # Clip to [0, 1]
        new_rep = max(0.0, min(1.0, new_rep))
        
        # Floor lifting
        if new_rep < self.floor_lift_threshold:
            new_rep += self.floor_lift_amount
            new_rep = min(1.0, new_rep)
        
        self.reputations[client_id] = new_rep
        
        return new_rep
    
    def _compute_variance_penalty(self,
                                  gradient: np.ndarray,
                                  grad_median: np.ndarray) -> float:
        """Compute variance-based penalty."""
        # Compute distance to median
        dist = np.linalg.norm(gradient.flatten() - grad_median)
        median_norm = np.linalg.norm(grad_median)
        
        # Normalized distance
        normalized_dist = dist / (median_norm + 1e-10)
        
        # Penalty proportional to distance
        penalty = min(self.penalty_variance, normalized_dist * self.penalty_variance)
        
        return penalty
    
    def get_reputation(self, client_id: int) -> float:
        """Get reputation for a client."""
        return self.reputations.get(client_id, self.initial_reputation)
    
    def get_stats(self) -> Dict:
        """Get reputation statistics."""
        if not self.reputations:
            return {
                'num_clients': 0,
                'mean_reputation': 0.0,
                'min_reputation': 0.0,
                'max_reputation': 0.0
            }
        
        reps = list(self.reputations.values())
        return {
            'num_clients': len(reps),
            'mean_reputation': np.mean(reps),
            'min_reputation': np.min(reps),
            'max_reputation': np.max(reps),
            'ema_alpha_increase': self.ema_alpha_increase,
            'ema_alpha_decrease': self.ema_alpha_decrease
        }