# cifar_cnn/defense/reputation.py

from typing import Dict, List, Optional
from collections import defaultdict
from logging import INFO
from flwr.common.logger import log

class ReputationSystem:
    def __init__(
        self,
        ema_alpha_increase: float = 0.1,
        ema_alpha_decrease: float = 0.6,
        penalty_flagged: float = 0.2,
        penalty_variance: float = 0.1,
        reward_clean: float = 0.1,
        floor_warning_threshold: float = 0.2, # Key mới
        floor_target_value: float = 0.3,      # Key mới
        floor_probation_rounds: int = 5,      # Key mới
        initial_reputation: float = 0.5,
        # Giữ lại các tham số cũ (nhưng không dùng) để tránh lỗi nếu server truyền dư
        floor_lift_threshold: Optional[float] = None,
        floor_lift_amount: Optional[float] = None
    ):
        # Param Config
        self.alpha_up = ema_alpha_increase
        self.alpha_down = ema_alpha_decrease
        self.penalty_flagged = penalty_flagged
        self.reward_clean = reward_clean
        
        # Initial State
        self.initial_reputation = initial_reputation
        self.reputations: Dict[str, float] = defaultdict(lambda: self.initial_reputation)
        
        # Probation Logic Config
        # Ưu tiên dùng key mới, nếu không có thì fallback sang key cũ (để tương thích ngược)
        self.floor_warning_threshold = floor_warning_threshold if floor_lift_threshold is None else floor_lift_threshold
        self.floor_target_value = floor_target_value if floor_lift_amount is None else floor_lift_amount
        self.floor_probation_rounds = floor_probation_rounds
        
        # State counters for probation
        self.probation_counts: Dict[str, int] = defaultdict(int)
        
        log(INFO, f"🛡️ ReputationSystem initialized. Init={self.initial_reputation}, AlphaDown={self.alpha_down}")

    def initialize_client(self, client_id: int):
        """Khởi tạo danh tiếng cho client mới."""
        cid_str = str(client_id)
        if cid_str not in self.reputations:
            self.reputations[cid_str] = self.initial_reputation

    def update(
        self, 
        client_id: int, 
        gradient: any, # Unused direct gradient access, processing done via flags
        median_gradient: any, 
        is_flagged: bool, 
        server_round: int
    ) -> float:
        """
        Cập nhật danh tiếng sau mỗi vòng.
        Trả về giá trị danh tiếng mới.
        """
        cid = str(client_id)
        current_rep = self.reputations[cid]
        
        # --- BƯỚC 1: CẬP NHẬT EMA ---
        if is_flagged:
            # Phạt: Giảm nhanh
            new_rep = (1 - self.alpha_down) * current_rep
        else:
            # Thưởng: Tăng chậm
            # Performance score = 1.0 (vì đã clean)
            new_rep = (1 - self.alpha_up) * current_rep + self.alpha_up * 1.0

        # Kẹp giá trị [0, 1]
        new_rep = max(0.0, min(1.0, new_rep))
        
        # --- BƯỚC 2: CƠ CHẾ NÂNG SÀN (PROBATION) ---
        if new_rep < self.floor_warning_threshold:
            if not is_flagged:
                self.probation_counts[cid] += 1
                if self.probation_counts[cid] >= self.floor_probation_rounds:
                    log(INFO, f"🆙 Client {cid} passed probation ({self.floor_probation_rounds} rounds). Lift {new_rep:.2f} -> {self.floor_target_value}")
                    new_rep = self.floor_target_value
                    self.probation_counts[cid] = 0
            else:
                self.probation_counts[cid] = 0
        else:
            if cid in self.probation_counts:
                del self.probation_counts[cid]

        self.reputations[cid] = new_rep
        return new_rep

    def get_reputation(self, client_id: int) -> float:
        return self.reputations[str(client_id)]
        
    def get_stats(self) -> Dict:
        reps = list(self.reputations.values())
        return {
            "mean_reputation": sum(reps) / len(reps) if reps else 0.0,
            "min_reputation": min(reps) if reps else 0.0,
            "max_reputation": max(reps) if reps else 0.0,
            "clients_in_probation": len(self.probation_counts)
        }