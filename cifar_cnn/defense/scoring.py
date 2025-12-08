# cifar_cnn/defense/scoring.py
"""
Confidence Scoring (V2)
=======================
Tính điểm bất thường (ci) dựa trên kết quả từ các lớp trước.

Config loaded from: [tool.flwr.app.config.defense.confidence]

Formula:
    score = (w_flag * I_flag + w_euc * I_euc + w_base * δi) * factor
    ci = clip(score, 0, 1)

Logic Factor:
    Nếu δi > threshold VÀ status == SUSPICIOUS -> factor = 0.8 (Giảm nhẹ tội)
    Ngược lại -> factor = 1.0
"""

import numpy as np
from typing import Dict, List

class ConfidenceScorer:
    def __init__(
        self,
        weight_flagged: float = 0.4,
        weight_euclidean: float = 0.3,
        weight_baseline: float = 0.3,
        baseline_suspicious_threshold: float = 0.3,
        baseline_factor_suspicious: float = 0.8
    ):
        self.w_flag = weight_flagged
        self.w_euc = weight_euclidean
        self.w_base = weight_baseline
        self.delta_threshold = baseline_suspicious_threshold
        self.factor_suspicious = baseline_factor_suspicious
        
        print(f"✅ ConfidenceScorer Initialized:")
        print(f"   Weights: Flag={self.w_flag}, Euc={self.w_euc}, Base={self.w_base}")
        print(f"   Suspicious factor: {self.factor_suspicious} (if δ > {self.delta_threshold})")

    def calculate_scores(
        self,
        client_ids: List[int],
        layer1_results: Dict[int, str],      # "FLAGGED", "ACCEPTED"...
        suspicion_levels: Dict[int, str],    # "suspicious", "clean"... (từ Layer 2)
        baseline_deviations: Dict[int, float] # δi (từ NonIIDHandler)
    ) -> Dict[int, float]:
        
        scores = {}
        
        for cid in client_ids:
            # 1. Indicator Flagged (Lớp 1)
            # Nếu bị FLAGGED ở L1 -> 1.0, ngược lại 0.0
            is_flagged = 1.0 if layer1_results.get(cid) == "FLAGGED" else 0.0
            
            # 2. Indicator Fail Euclidean (Lớp 2)
            # Nếu status là SUSPICIOUS -> có nghĩa là fail Euclidean hoặc L1 Flagged
            # Để tách biệt fail Euclidean, ta cần check logic:
            # Trong Layer 2: Suspicious = (L1 Flagged) OR (Fail Distance)
            # Ở đây ta xấp xỉ: Nếu L2 Suspicious -> coi như có dấu hiệu bất thường
            # Tuy nhiên, để chính xác theo công thức, tốt nhất truyền thẳng kết quả check distance vào.
            # *Tạm thời dùng logic*: Nếu Suspicious -> tính là 1 phần rủi ro
            is_suspicious = 1.0 if suspicion_levels.get(cid) == "suspicious" else 0.0
            
            # 3. Baseline Deviation (δi)
            delta = baseline_deviations.get(cid, 0.0)
            
            # Tính điểm thô
            # Lưu ý: Logic mapping "is_suspicious" ở đây cần khớp với định nghĩa I_Fail_Euclidean
            # Nếu bạn muốn chính xác tuyệt đối, Layer 2 cần trả về flag `fail_distance` riêng.
            # Ở đây ta dùng công thức tổng quát:
            raw_score = (self.w_flag * is_flagged + 
                         self.w_euc * is_suspicious + 
                         self.w_base * delta)
            
            # 4. Áp dụng Factor (Ngữ cảnh)
            factor = 1.0
            # Nếu lệch baseline nhiều (δ > 0.3) NHƯNG đã được cứu vãn (Suspicious/Clean)
            # thì giảm nhẹ điểm phạt (chấp nhận drift)
            if delta > self.delta_threshold and suspicion_levels.get(cid) in ["suspicious", "clean"]:
                factor = self.factor_suspicious
            
            final_score = raw_score * factor
            scores[cid] = float(np.clip(final_score, 0.0, 1.0))
            
        return scores