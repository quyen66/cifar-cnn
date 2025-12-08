"""
Layer 2: Distance + Direction Detection (V2 - main.pdf)
=========================================================
Phân tích sâu và "cứu vãn" clients từ Layer 1.

VAI TRÒ LAYER 2:
- Nhận kết quả từ Layer 1 (REJECTED/FLAGGED/ACCEPTED)
- REJECTED từ Layer 1 → Giữ nguyên, KHÔNG xử lý
- FLAGGED/ACCEPTED → Phân tích thêm bằng Distance + Direction

KIỂM TRA:
1. Euclidean Distance: di = ||gi - gmedian||
   - Vi phạm nếu: di > 1.5 × Median({dj})
   
2. Cosine Similarity: Simi = cos(gi, gmedian)
   - Vi phạm nếu: Simi ≤ 0.3 (hướng ngược/vuông góc)

MA TRẬN QUYẾT ĐỊNH (Giai đoạn 3 trong PDF):
┌─────────────┬──────────────┬─────────────┬──────────────────────┐
│ Layer1      │ Cosine       │ Euclidean   │ Kết quả Layer 2      │
├─────────────┼──────────────┼─────────────┼──────────────────────┤
│ REJECTED    │ -            │ -           │ REJECTED             │
│ FLAGGED     │ ≤ 0.3        │ -           │ REJECTED             │
│ FLAGGED     │ > 0.3        │ fail        │ ACCEPTED (suspicious)│
│ FLAGGED     │ > 0.3        │ pass        │ ACCEPTED (suspicious)│ ← Vẫn nghi ngờ vì L1 đã flag
│ ACCEPTED    │ ≤ 0.3        │ -           │ REJECTED             │
│ ACCEPTED    │ > 0.3        │ fail        │ ACCEPTED (suspicious)│
│ ACCEPTED    │ > 0.3        │ pass        │ ACCEPTED (clean)     │ ← Chỉ case này mới clean
└─────────────┴──────────────┴─────────────┴──────────────────────┘

NGUYÊN TẮC: Nếu L1 đã FLAGGED, dù L2 rescue cũng phải giữ SUSPICIOUS

OUTPUT:
- final_status: Dict[client_id, "REJECTED"/"ACCEPTED"]
- suspicion_level: Dict[client_id, "clean"/"suspicious"/None]
  (None cho REJECTED clients - không cần track suspicion)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum


class Layer2Result(Enum):
    """Kết quả cuối cùng sau Layer 2."""
    REJECTED = "REJECTED"
    ACCEPTED = "ACCEPTED"


class SuspicionLevel(Enum):
    """Mức độ nghi ngờ (chỉ áp dụng cho ACCEPTED clients)."""
    CLEAN = "clean"           # Hoàn toàn sạch (L1 ACCEPTED + L2 pass all)
    SUSPICIOUS = "suspicious"  # Chấp nhận nhưng nghi ngờ (dùng để tính ci)


class Layer2Detector:
    """
    Layer 2 Detector với ma trận cứu vãn (V2 - main.pdf).
    
    Nhận input từ Layer 1 và quyết định cuối cùng.
    """
    
    def __init__(
        self,
        distance_multiplier: float = 1.5,
        cosine_threshold: float = 0.3
    ):
        """
        Initialize Layer 2 Detector.
        
        Args:
            distance_multiplier: Multiplier for distance threshold (1.5)
            cosine_threshold: Threshold for cosine similarity (0.3)
        """
        self.distance_multiplier = distance_multiplier
        self.cosine_threshold = cosine_threshold
        
        # Stats for debugging
        self.last_stats = {}
        
        print(f"✅ Layer2Detector V2 initialized:")
        print(f"   Distance multiplier: {distance_multiplier}")
        print(f"   Cosine threshold: {cosine_threshold}")

    def detect(
        self,
        gradients: List[np.ndarray],
        client_ids: List[int],
        layer1_results: Dict[int, str],
        current_round: int = 0,
        is_malicious_ground_truth: Optional[List[bool]] = None
    ) -> Tuple[Dict[int, str], Dict[int, Optional[str]]]:
        """
        Phân tích Layer 2 dựa trên kết quả Layer 1.
        
        Args:
            gradients: List gradient arrays
            client_ids: List client IDs
            layer1_results: Dict[client_id, status] từ Layer 1
            current_round: Round hiện tại
            is_malicious_ground_truth: Ground truth (optional)
            
        Returns:
            final_status: Dict[client_id, "REJECTED"/"ACCEPTED"]
            suspicion_levels: Dict[client_id, "clean"/"suspicious"/None]
                - None cho REJECTED clients (không cần track)
        """
        num_clients = len(gradients)
        
        if num_clients < 2:
            return (
                {cid: Layer2Result.ACCEPTED.value for cid in client_ids},
                {cid: SuspicionLevel.CLEAN.value for cid in client_ids}
            )
        
        # =========================================================
        # STEP 1: Tính toán metrics (cho tất cả clients)
        # =========================================================
        distances, cosines, median_grad = self._compute_metrics(gradients)
        
        # Tính ngưỡng distance động
        median_distance = np.median(distances)
        distance_threshold = self.distance_multiplier * median_distance
        
        # =========================================================
        # STEP 2: Áp dụng ma trận quyết định
        # =========================================================
        final_status = {}
        suspicion_levels = {}
        
        # Track cho stats
        stats = {
            "median_distance": float(median_distance),
            "distance_threshold": float(distance_threshold),
            "cosine_threshold": self.cosine_threshold,
            "decisions": []
        }
        
        for i, cid in enumerate(client_ids):
            layer1_status = layer1_results.get(cid, "ACCEPTED")
            dist = distances[i]
            cos = cosines[i]
            
            fail_distance = dist > distance_threshold
            fail_cosine = cos <= self.cosine_threshold
            
            # Áp dụng ma trận quyết định
            result, suspicion = self._apply_decision_matrix(
                layer1_status, fail_cosine, fail_distance
            )
            
            final_status[cid] = result.value
            # suspicion có thể là None (cho REJECTED) hoặc SuspicionLevel enum
            suspicion_levels[cid] = suspicion.value if suspicion else None
            
            stats["decisions"].append({
                "client_id": cid,
                "layer1": layer1_status,
                "distance": float(dist),
                "cosine": float(cos),
                "fail_distance": fail_distance,
                "fail_cosine": fail_cosine,
                "final": result.value,
                "suspicion": suspicion.value if suspicion else None
            })
        
        self.last_stats = stats
        
        # Log results
        self._log_results(
            client_ids, final_status, suspicion_levels,
            layer1_results, is_malicious_ground_truth, current_round
        )
        
        return final_status, suspicion_levels

    def _compute_metrics(
        self, 
        gradients: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tính Euclidean distance và Cosine similarity với median gradient.
        
        Returns:
            distances: Array of Euclidean distances
            cosines: Array of Cosine similarities
            median_grad: Median gradient vector
        """
        # Stack gradients
        grad_matrix = np.vstack([g.flatten() for g in gradients])
        
        # Tính median gradient (theo từng chiều)
        median_grad = np.median(grad_matrix, axis=0)
        
        # Euclidean distances
        distances = np.array([
            np.linalg.norm(g - median_grad) for g in grad_matrix
        ])
        
        # Cosine similarities
        median_norm = np.linalg.norm(median_grad)
        cosines = []
        for g in grad_matrix:
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-9 or median_norm < 1e-9:
                cosines.append(0.0)
            else:
                cos = np.dot(g, median_grad) / (g_norm * median_norm)
                cosines.append(float(np.clip(cos, -1.0, 1.0)))
        
        return distances, np.array(cosines), median_grad

    def _apply_decision_matrix(
        self,
        layer1_status: str,
        fail_cosine: bool,
        fail_distance: bool
    ) -> Tuple[Layer2Result, Optional[SuspicionLevel]]:
        """
        Áp dụng ma trận quyết định theo main.pdf.
        
        Returns:
            result: REJECTED hoặc ACCEPTED
            suspicion: CLEAN/SUSPICIOUS cho ACCEPTED, None cho REJECTED
        """
        # REJECTED từ Layer 1 → Giữ nguyên
        if layer1_status == "REJECTED":
            return Layer2Result.REJECTED, None
        
        # Kiểm tra Cosine (quan trọng nhất - hướng gradient)
        if fail_cosine:
            # Hướng sai → REJECTED
            return Layer2Result.REJECTED, None
        
        # Cosine OK, xét theo Layer 1 status
        if layer1_status == "FLAGGED":
            # L1 đã FLAGGED → dù L2 thấy OK cũng phải SUSPICIOUS
            return Layer2Result.ACCEPTED, SuspicionLevel.SUSPICIOUS
        
        # L1 ACCEPTED + Cosine OK
        if fail_distance:
            # Distance lớn nhưng hướng đúng → nghi ngờ
            return Layer2Result.ACCEPTED, SuspicionLevel.SUSPICIOUS
        
        # L1 ACCEPTED + Cosine OK + Distance OK → Hoàn toàn sạch
        return Layer2Result.ACCEPTED, SuspicionLevel.CLEAN

    def _log_results(
        self,
        client_ids: List[int],
        final_status: Dict[int, str],
        suspicion_levels: Dict[int, Optional[str]],
        layer1_results: Dict[int, str],
        ground_truth: Optional[List[bool]],
        current_round: int
    ):
        """Log kết quả Layer 2."""
        # Count by status
        rejected_count = sum(1 for s in final_status.values() if s == "REJECTED")
        accepted_count = sum(1 for s in final_status.values() if s == "ACCEPTED")
        
        # Count by suspicion (chỉ cho ACCEPTED)
        clean_count = sum(1 for s in suspicion_levels.values() if s == "clean")
        suspicious_count = sum(1 for s in suspicion_levels.values() if s == "suspicious")
        
        # Count rescued (FLAGGED L1 → ACCEPTED L2)
        rescued = sum(
            1 for cid in client_ids 
            if layer1_results.get(cid) == "FLAGGED" and final_status.get(cid) == "ACCEPTED"
        )
        
        # Count confirmed (FLAGGED L1 → REJECTED L2)
        confirmed = sum(
            1 for cid in client_ids
            if layer1_results.get(cid) == "FLAGGED" and final_status.get(cid) == "REJECTED"
        )
        
        print(f"\n{'='*60}")
        print(f"📊 Layer 2 Results - Round {current_round}")
        print(f"{'='*60}")
        print(f"   Final Status:")
        print(f"      REJECTED: {rejected_count}")
        print(f"      ACCEPTED: {accepted_count}")
        print(f"   Suspicion Levels (ACCEPTED only):")
        print(f"      Clean: {clean_count}")
        print(f"      Suspicious: {suspicious_count}")
        print(f"   Layer 1 → Layer 2 Changes:")
        print(f"      Rescued (FLAGGED→ACCEPTED): {rescued}")
        print(f"      Confirmed (FLAGGED→REJECTED): {confirmed}")
        
        if ground_truth:
            # Tính metrics cuối cùng
            actual_malicious = [i for i, m in enumerate(ground_truth) if m]
            detected = [i for i, cid in enumerate(client_ids) 
                       if final_status.get(cid) == "REJECTED"]
            
            tp = len(set(actual_malicious) & set(detected))
            fp = len(set(detected) - set(actual_malicious))
            
            detection_rate = tp / len(actual_malicious) if actual_malicious else 0
            benign_count = len(ground_truth) - len(actual_malicious)
            fpr = fp / benign_count if benign_count > 0 else 0
            
            print(f"\n   📈 Final Metrics (after Layer 2):")
            print(f"      Detection Rate: {detection_rate:.1%} ({tp}/{len(actual_malicious)})")
            print(f"      False Positive Rate: {fpr:.1%} ({fp}/{benign_count})")
        
        print(f"{'='*60}\n")

    def get_stats(self) -> Dict:
        """Get last detection stats."""
        return self.last_stats