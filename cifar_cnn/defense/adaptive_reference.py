"""
Adaptive Hybrid Reference Tracker (NEW FILE)
=============================================
Giải quyết vấn đề FPR cao khi attack corrupt current-round reference.

NGUYÊN LÝ:
- Khi H score thấp (ít attack) → dùng current-round median (đáng tin)
- Khi H score cao (nhiều attack) → dùng historical momentum (không bị corrupt)

CÔNG THỨC:
    α = clip(α_base + α_h_mult × H, α_min, α_max)
    reference = α × historical_momentum + (1-α) × current_median

VÍ DỤ:
    H = 0.1 (IID, ít attack)  → α ≈ 0.25 → 75% current, 25% historical
    H = 0.6 (high attack)     → α ≈ 0.55 → 45% current, 55% historical
    H = 0.9 (severe attack)   → α ≈ 0.70 → 30% current, 70% historical

TÍCH HỢP:
    1. Copy file này vào cifar_cnn/defense/
    2. Import vào __init__.py
    3. Sửa server_app.py theo hướng dẫn
"""

import numpy as np
from typing import Optional, Dict, List


class AdaptiveReferenceTracker:
    """
    Tracks historical gradient momentum and computes adaptive reference
    for Layer 2 detection.
    
    Giải quyết vấn đề: Khi >30% clients là attackers, current-round median
    bị corrupt → benign clients fail cosine check → FPR cao.
    
    Solution: Blend historical momentum (clean) với current median (potentially corrupt)
    dựa trên H score.
    """
    
    def __init__(
        self,
        # Adaptive blending parameters
        alpha_base: float = 0.2,      # Base weight cho historical
        alpha_h_mult: float = 0.5,    # H multiplier: α = α_base + α_h_mult × H
        alpha_min: float = 0.1,       # Min historical weight (luôn giữ 1 ít historical)
        alpha_max: float = 0.8,       # Max historical weight (không bỏ hết current)
        
        # Momentum parameters
        momentum_decay: float = 0.9,  # EMA decay cho momentum update
        
        # Safety parameters
        warmup_rounds: int = 10,      # Số round warmup (chỉ dùng current)
        min_history_rounds: int = 3,  # Số round tối thiểu để dùng historical
        
        # Round-based adaptation
        round_decay_start: int = 30,  # Round bắt đầu giảm historical weight
        round_decay_factor: float = 0.01,  # Mỗi round sau 30, giảm α thêm factor
    ):
        """
        Initialize Adaptive Reference Tracker.
        
        Args:
            alpha_base: Base weight cho historical reference
            alpha_h_mult: Multiplier - tăng historical weight khi H tăng
            alpha_min: Minimum historical weight
            alpha_max: Maximum historical weight
            momentum_decay: EMA decay cho momentum update
            warmup_rounds: Số round warmup (không dùng historical)
            min_history_rounds: Số round tối thiểu cần thiết để dùng historical
            round_decay_start: Round bắt đầu giảm historical weight
            round_decay_factor: Factor giảm α mỗi round sau round_decay_start
        """
        # Blending parameters
        self.alpha_base = alpha_base
        self.alpha_h_mult = alpha_h_mult
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Momentum parameters
        self.momentum_decay = momentum_decay
        
        # Safety parameters
        self.warmup_rounds = warmup_rounds
        self.min_history_rounds = min_history_rounds
        self.round_decay_start = round_decay_start
        self.round_decay_factor = round_decay_factor
        
        # State
        self.historical_momentum: Optional[np.ndarray] = None
        self.update_count: int = 0
        self.last_alpha_used: float = 0.0
        self.last_h_score: float = 0.0
        
        # Logging
        self._log_init()
    
    def _log_init(self):
        """Log initialization parameters."""
        print(f"\n{'='*60}")
        print(f"✅ AdaptiveReferenceTracker Initialized")
        print(f"{'='*60}")
        print(f"📊 Blending Formula: ref = α×historical + (1-α)×current")
        print(f"   α = clip(α_base + α_h_mult × H, α_min, α_max)")
        print(f"   α_base={self.alpha_base}, α_h_mult={self.alpha_h_mult}")
        print(f"   α_min={self.alpha_min}, α_max={self.alpha_max}")
        print(f"📊 Momentum decay: {self.momentum_decay}")
        print(f"📊 Safety: warmup={self.warmup_rounds}, min_history={self.min_history_rounds}")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # CORE METHODS
    # =========================================================================
    
    def update_momentum(
        self,
        aggregated_gradient: np.ndarray,
        current_round: int,
        is_clean_round: bool = True
    ) -> None:
        """
        Update historical momentum với aggregated gradient từ round vừa xong.
        
        GỌI SAU KHI AGGREGATION XONG (cuối aggregate_fit).
        
        Args:
            aggregated_gradient: Gradient đã aggregate (từ clean clients)
            current_round: Round hiện tại
            is_clean_round: True nếu round này có ít attack/FP thấp
                           (Warmup rounds luôn là clean)
        
        Note:
            - Warmup rounds (1-10): Momentum được update với weight cao hơn
            - Post-warmup: Dùng EMA decay bình thường
        """
        # Ensure flat array
        if isinstance(aggregated_gradient, list):
            flat_grad = np.concatenate([g.flatten() for g in aggregated_gradient])
        else:
            flat_grad = aggregated_gradient.flatten()
        
        # First update - initialize
        if self.historical_momentum is None:
            self.historical_momentum = flat_grad.copy()
            self.update_count = 1
            print(f"   📊 [AdaptiveRef] Initialized momentum (round {current_round})")
            return
        
        # Check dimension match
        if flat_grad.shape != self.historical_momentum.shape:
            print(f"   ⚠️ [AdaptiveRef] Shape mismatch! Reinitializing momentum.")
            self.historical_momentum = flat_grad.copy()
            self.update_count = 1
            return
        
        # EMA update
        # Trong warmup, dùng decay cao hơn (học nhanh từ clean data)
        if current_round <= self.warmup_rounds:
            decay = 0.7  # Higher weight cho new gradient trong warmup
        else:
            decay = self.momentum_decay
        
        # momentum = decay × old_momentum + (1-decay) × new_gradient
        self.historical_momentum = (
            decay * self.historical_momentum + 
            (1 - decay) * flat_grad
        )
        
        self.update_count += 1
        
        # Debug log mỗi 10 rounds
        if current_round % 10 == 0 or current_round <= 3:
            momentum_norm = np.linalg.norm(self.historical_momentum)
            print(f"   📊 [AdaptiveRef] Momentum updated (round {current_round}, "
                  f"count={self.update_count}, norm={momentum_norm:.2f})")
    
    def compute_adaptive_reference(
        self,
        current_gradients: List[np.ndarray],
        h_score: float,
        current_round: int
    ) -> np.ndarray:
        """
        Tính adaptive reference vector cho Layer 2.
        
        GỌI TRƯỚC LAYER 2 DETECTION.
        
        Args:
            current_gradients: List gradients từ round hiện tại
            h_score: Heterogeneity score từ NonIIDHandler
            current_round: Round hiện tại
            
        Returns:
            reference: Blended reference vector
        """
        # Stack and compute current-round median
        grad_matrix = np.vstack([g.flatten() for g in current_gradients])
        current_median = np.median(grad_matrix, axis=0)
        
        # === CASE 1: Warmup phase - chỉ dùng current ===
        if current_round <= self.warmup_rounds:
            self.last_alpha_used = 0.0
            self.last_h_score = h_score
            print(f"   🛡️ [AdaptiveRef] Warmup round {current_round}: Using CURRENT median only")
            return current_median
        
        # === CASE 2: Không đủ history - chỉ dùng current ===
        if self.historical_momentum is None or self.update_count < self.min_history_rounds:
            self.last_alpha_used = 0.0
            self.last_h_score = h_score
            print(f"   🛡️ [AdaptiveRef] Insufficient history ({self.update_count} rounds): "
                  f"Using CURRENT median only")
            return current_median
        
        # === CASE 3: Dimension mismatch - fallback to current ===
        if current_median.shape != self.historical_momentum.shape:
            self.last_alpha_used = 0.0
            self.last_h_score = h_score
            print(f"   ⚠️ [AdaptiveRef] Shape mismatch! Using CURRENT median only")
            return current_median
        
        # === CASE 4: Normal operation - adaptive blend ===
        
        # Compute adaptive alpha based on H score
        # α = α_base + α_h_mult × H
        # Khi H cao (attack nhiều) → α cao → dùng historical nhiều hơn
        alpha = self.alpha_base + self.alpha_h_mult * h_score
        
        # Round-based decay (optional: giảm historical weight ở late rounds)
        # Vì model đã converge, current median đáng tin hơn
        if current_round > self.round_decay_start:
            rounds_after_decay = current_round - self.round_decay_start
            alpha -= rounds_after_decay * self.round_decay_factor
        
        # Clip to valid range
        alpha = np.clip(alpha, self.alpha_min, self.alpha_max)
        
        # Blend: reference = α × historical + (1-α) × current
        reference = alpha * self.historical_momentum + (1 - alpha) * current_median
        
        # Store for logging
        self.last_alpha_used = alpha
        self.last_h_score = h_score
        
        # Log
        hist_weight_pct = alpha * 100
        curr_weight_pct = (1 - alpha) * 100
        print(f"   🛡️ [AdaptiveRef] Round {current_round}: "
              f"H={h_score:.3f} → α={alpha:.3f} "
              f"(Historical: {hist_weight_pct:.1f}%, Current: {curr_weight_pct:.1f}%)")
        
        return reference
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        momentum_norm = (np.linalg.norm(self.historical_momentum) 
                        if self.historical_momentum is not None else 0.0)
        return {
            'update_count': self.update_count,
            'momentum_norm': float(momentum_norm),
            'last_alpha_used': float(self.last_alpha_used),
            'last_h_score': float(self.last_h_score),
            'has_momentum': self.historical_momentum is not None,
            'config': {
                'alpha_base': self.alpha_base,
                'alpha_h_mult': self.alpha_h_mult,
                'alpha_min': self.alpha_min,
                'alpha_max': self.alpha_max,
                'momentum_decay': self.momentum_decay,
                'warmup_rounds': self.warmup_rounds,
                'min_history_rounds': self.min_history_rounds
            }
        }
    
    def reset(self):
        """Reset tracker state."""
        self.historical_momentum = None
        self.update_count = 0
        self.last_alpha_used = 0.0
        self.last_h_score = 0.0
        print("   🔄 [AdaptiveRef] Tracker reset")


# =============================================================================
# TESTING
# =============================================================================

def test_adaptive_reference_tracker():
    """Test AdaptiveReferenceTracker."""
    print("\n" + "="*70)
    print("🧪 TESTING ADAPTIVE REFERENCE TRACKER")
    print("="*70)
    
    np.random.seed(42)
    
    tracker = AdaptiveReferenceTracker(
        alpha_base=0.2,
        alpha_h_mult=0.5,
        alpha_min=0.1,
        alpha_max=0.8,
        warmup_rounds=3,
        min_history_rounds=2
    )
    
    # Simulate warmup
    print("\n--- Warmup Phase ---")
    for r in range(1, 4):
        grads = [np.random.randn(1000) for _ in range(10)]
        ref = tracker.compute_adaptive_reference(grads, h_score=0.1, current_round=r)
        agg = np.mean(np.vstack([g.flatten() for g in grads]), axis=0)
        tracker.update_momentum(agg, r)
        print(f"   Round {r}: α={tracker.last_alpha_used:.3f}")
    
    # Simulate post-warmup with varying H scores
    print("\n--- Post-Warmup Phase ---")
    test_cases = [
        (4, 0.1, "Low H (IID)"),
        (5, 0.3, "Medium H"),
        (6, 0.6, "High H (attack likely)"),
        (7, 0.9, "Very High H (severe attack)"),
    ]
    
    for r, h, desc in test_cases:
        grads = [np.random.randn(1000) for _ in range(10)]
        ref = tracker.compute_adaptive_reference(grads, h_score=h, current_round=r)
        agg = np.mean(np.vstack([g.flatten() for g in grads]), axis=0)
        tracker.update_momentum(agg, r)
        print(f"   Round {r} ({desc}): H={h:.1f} → α={tracker.last_alpha_used:.3f}")
    
    # Verify alpha increases with H
    print("\n--- Verification ---")
    alphas = []
    for h in [0.0, 0.25, 0.5, 0.75, 1.0]:
        grads = [np.random.randn(1000) for _ in range(10)]
        tracker.compute_adaptive_reference(grads, h_score=h, current_round=10)
        alphas.append((h, tracker.last_alpha_used))
    
    is_monotonic = all(alphas[i][1] <= alphas[i+1][1] for i in range(len(alphas)-1))
    if is_monotonic:
        print("   ✅ Alpha increases monotonically with H")
    else:
        print("   ❌ Alpha NOT monotonic with H")
        for h, a in alphas:
            print(f"      H={h:.2f} → α={a:.3f}")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_adaptive_reference_tracker()