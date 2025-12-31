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
        baseline_factor_suspicious: float = 0.8,
        rescued_penalty: float = 0.3,
        rescued_suspicious_penalty: float = 0.7,
        rescued_euclidean: float = 0.5,
    ):
        self.w_flag = weight_flagged
        self.w_euc = weight_euclidean
        self.w_base = weight_baseline
        self.delta_threshold = baseline_suspicious_threshold
        self.factor_suspicious = baseline_factor_suspicious
        self.rescued_penalty = rescued_penalty
        self.rescued_suspicious_penalty = rescued_suspicious_penalty
        self.rescued_euclidean = rescued_euclidean
        
        print(f"✅ ConfidenceScorer Initialized:")
        print(f"   Weights: Flag={self.w_flag}, Euc={self.w_euc}, Base={self.w_base}")
        print(f"   Rescued penalty: {rescued_penalty}")
        print(f"   Suspicious factor: {self.factor_suspicious} (if δ > {self.delta_threshold})")

    def calculate_scores(
        self,
        client_ids: List[int],
        layer1_results: Dict[int, str],      # "FLAGGED", "ACCEPTED"...
        layer2_status: Dict[int, str],
        suspicion_levels: Dict[int, str],    # "suspicious", "clean"... (từ Layer 2)
        baseline_deviations: Dict[int, float], # δi (từ NonIIDHandler)
    ) -> Dict[int, float]:
        
        scores = {}
        
        for cid in client_ids:
            # 1. Indicator Flagged (Lớp 1)
            l1_status = layer1_results.get(cid, "ACCEPTED")
            l2_status = layer2_status.get(cid, "ACCEPTED")
            
            if l1_status == "REJECTED" or l2_status == "REJECTED":
                scores[cid] = 1.0
                print(f"         Client {cid}: ci=1.000 (FORCED REJECT - {l1_status}/{l2_status})")
                continue
            
            delta = baseline_deviations.get(cid, 0.0)
            suspicion = suspicion_levels.get(cid, "clean")

            if l1_status == "REJECTED":
                # Layer 1 hard reject → always flagged
                I_flagged = 1.0
                I_euclidean = 1.0
            elif l1_status == "FLAGGED":
                if l2_status == "ACCEPTED":
                    # Layer 2 rescued → NOT flagged anymore!
                    if suspicion == "suspicious":
                        I_flagged = self.rescued_suspicious_penalty  # 0.7
                    else:
                        I_flagged = self.rescued_penalty  # 0.3
                else:
                    # Layer 2 confirmed → still flagged
                    I_flagged = 1.0
            else:
                # Layer 1 accepted → not flagged
                I_flagged = 0.0


            # 2. Indicator Fail Euclidean (Lớp 2)
            
            if l2_status == "REJECTED":
                I_euclidean = 1.0  # Failed L2 checks!
            elif suspicion == "suspicious":
                I_euclidean = self.rescued_euclidean  # Passed but with suspicion
            else:
                I_euclidean = 0.0  # Clean
                
                 
            # Baseline deviation factor
            if delta > self.delta_threshold:
                delta_factor = delta * self.factor_suspicious
            else:
                delta_factor = delta
            
            # Calculate CI
            ci = (self.w_flag * I_flagged + 
                self.w_euc * I_euclidean + 
                self.w_base * delta_factor)
            
            scores[cid] = min(max(ci, 0.0), 1.0)
            
            # Debug print
            print(f"         Client {cid}: ci={scores[cid]:.3f} (I_flag={I_flagged}, L1={l1_status}, L2={l2_status})")

            
        return scores