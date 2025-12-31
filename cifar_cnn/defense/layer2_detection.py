"""
Layer 2: Distance + Direction Detection (V2 - main.pdf)
=========================================================
Phân tích sâu và "cứu vãn" clients từ Layer 1.

VAI TRÒ LAYER 2:
- Nhận kết quả từ Layer 1 (REJECTED/FLAGGED/ACCEPTED)
- REJECTED từ Layer 1 → Giữ nguyên, KHÔNG xử lý
- FLAGGED/ACCEPTED → Phân tích thêm bằng Distance + Direction

KIỂM TRA (Tương thích Full Model & Last Layer):
1. Euclidean Distance: di = ||gi - gref||
   - Vi phạm nếu: di > 1.5 × Median({dj})
   - Với Full Model: di sẽ lớn hơn nhưng ngưỡng cũng tăng theo tỷ lệ, logic vẫn đúng.
   
2. Cosine Similarity: Simi = cos(gi, gref)
   - Vi phạm nếu: Simi ≤ 0.3 (hướng ngược/vuông góc)
   - Với Full Model: Cosine rất hiệu quả để phát hiện Sign Flip/Label Flip.

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
    Hỗ trợ tham chiếu ngoài (External Reference / Historical Momentum).
    """
    
    def __init__(
        self,
        distance_multiplier: float = 1.5,
        cosine_threshold: float = 0.3,
        enable_rescue: bool = False,
        rescue_cosine_threshold: float = 0.7,  # Ngưỡng cosine để rescue (cao hơn reject)
        require_distance_pass_for_rescue: bool = True,  # Phải pass distance để rescue
    ):
        """
        Initialize Layer 2 Detector.
        """
        self.distance_multiplier = distance_multiplier
        self.cosine_threshold = cosine_threshold
        self.enable_rescue = enable_rescue
        self.rescue_cosine_threshold = rescue_cosine_threshold
        self.require_distance_pass_for_rescue = require_distance_pass_for_rescue
        self.last_stats = {}
        
        print(f"✅ Layer2Detector V2 initialized:")
        print(f"   Distance multiplier: {distance_multiplier}")
        print(f"   Cosine threshold (reject): {cosine_threshold}")
        print(f"   Enable rescue: {enable_rescue}")
        if enable_rescue:
            print(f"   Rescue cosine threshold: {rescue_cosine_threshold}")
            print(f"   Require distance pass: {require_distance_pass_for_rescue}")

    def detect(
        self,
        gradients: List[np.ndarray],
        client_ids: List[int],
        layer1_results: Dict[int, str],
        current_round: int = 0,
        is_malicious_ground_truth: Optional[List[bool]] = None,
        external_reference: Optional[np.ndarray] = None  # <--- QUAN TRỌNG: Nhận Lịch sử từ Server
    ) -> Tuple[Dict[int, str], Dict[int, Optional[str]]]:
        """
        Phân tích Layer 2 dựa trên kết quả Layer 1.
        
        Args:
            gradients: List gradient arrays (Full Model hoặc Last Layer)
            client_ids: List client IDs
            layer1_results: Dict[client_id, status] từ Layer 1
            current_round: Round hiện tại
            is_malicious_ground_truth: Ground truth (optional)
            external_reference: Vector tham chiếu từ Server (Momentum)
            
        Returns:
            final_status: Dict[client_id, "REJECTED"/"ACCEPTED"]
            suspicion_levels: Dict[client_id, "clean"/"suspicious"/None]
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
        # Truyền external_reference xuống để tính khoảng cách chuẩn xác
        distances, cosines, median_grad = self._compute_metrics(gradients, reference_vector=external_reference)
        
        # Tính ngưỡng distance động (Adaptive Threshold)
        median_distance = np.median(distances)
        distance_threshold = self.distance_multiplier * median_distance
        
        self._log_per_client_metrics(
                client_ids, distances, cosines, 
                layer1_results, is_malicious_ground_truth,
                distance_threshold, current_round
            )
        
        # =========================================================
        # STEP 2: Áp dụng ma trận quyết định
        # =========================================================
        final_status = {}
        suspicion_levels = {}
        
        # Stats for debugging
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
            
            # Logic check vi phạm
            fail_distance = dist > distance_threshold
            fail_cosine = cos <= self.cosine_threshold
            
            # Check rescue condition
            can_rescue = (
                cos > self.rescue_cosine_threshold and 
                (not self.require_distance_pass_for_rescue or not fail_distance)
            )

            # Áp dụng ma trận quyết định
            result, suspicion = self._apply_decision_matrix(
                layer1_status, fail_cosine, fail_distance, can_rescue
            )
            
            final_status[cid] = result.value
            suspicion_levels[cid] = suspicion.value if suspicion else None
            
            stats["decisions"].append({
                "client_id": cid,
                "layer1": layer1_status,
                "distance": float(dist),
                "cosine": float(cos),
                "fail_distance": fail_distance,
                "fail_cosine": fail_cosine,
                "final": result.value,
                "can_rescue": can_rescue,
                "suspicion": suspicion.value if suspicion else None
            })
        
        self.last_stats = stats
        
        # Log results
        self._log_results(
            client_ids, final_status, suspicion_levels,
            layer1_results, is_malicious_ground_truth, current_round
        )
        
        return final_status, suspicion_levels

    def _log_per_client_metrics(
        self,
        client_ids: List[int],
        distances: np.ndarray,
        cosines: np.ndarray,
        layer1_results: Dict[int, str],
        ground_truth: Optional[List[bool]],
        distance_threshold: float,
        current_round: int
    ):
        """
        Log chi tiết cosine và distance cho từng client.
        """
        print(f"\n{'─'*70}")
        print(f"📐 Layer 2 Per-Client Metrics - Round {current_round}")
        print(f"{'─'*70}")
        print(f"   Distance threshold: {distance_threshold:.2f}")
        print(f"   Cosine reject threshold: {self.cosine_threshold}")
        print(f"   Cosine rescue threshold: {self.rescue_cosine_threshold}")
        print(f"{'─'*70}")
        
        # Header
        if ground_truth:
            print(f"   {'ID':>4} | {'L1 Status':>10} | {'Cosine':>8} | {'Distance':>10} | {'Dist OK':>8} | {'Cos OK':>7} | {'Rescue?':>7} | {'GT':>5}")
            print(f"   {'-'*4}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}")
        else:
            print(f"   {'ID':>4} | {'L1 Status':>10} | {'Cosine':>8} | {'Distance':>10} | {'Dist OK':>8} | {'Cos OK':>7} | {'Rescue?':>7}")
            print(f"   {'-'*4}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")
        
        # Sort by L1 status (FLAGGED first) then by cosine
        sorted_indices = sorted(
            range(len(client_ids)),
            key=lambda i: (
                0 if layer1_results.get(client_ids[i]) == "FLAGGED" else 1,
                -cosines[i]  # Higher cosine first
            )
        )
        
        for i in sorted_indices:
            cid = client_ids[i]
            l1_status = layer1_results.get(cid, "ACCEPTED")
            cos = cosines[i]
            dist = distances[i]
            
            dist_ok = "✓" if dist <= distance_threshold else "✗"
            cos_ok = "✓" if cos > self.cosine_threshold else "✗"
            
            # Check rescue eligibility
            can_rescue = (
                cos > self.rescue_cosine_threshold and 
                (not self.require_distance_pass_for_rescue or dist <= distance_threshold)
            )
            rescue_status = "✓" if can_rescue else "✗"
            
            # Color coding for cosine
            if cos > self.rescue_cosine_threshold:
                cos_indicator = "🟢"  # Good - can rescue
            elif cos > self.cosine_threshold:
                cos_indicator = "🟡"  # Medium - pass but can't rescue
            else:
                cos_indicator = "🔴"  # Bad - will reject
            
            if ground_truth:
                gt_label = "MAL" if ground_truth[i] else "BEN"
                gt_emoji = "⚠️" if ground_truth[i] else "✅"
                print(f"   {cid:>4} | {l1_status:>10} | {cos_indicator}{cos:>6.3f} | {dist:>10.2f} | {dist_ok:>8} | {cos_ok:>7} | {rescue_status:>7} | {gt_emoji}{gt_label}")
            else:
                print(f"   {cid:>4} | {l1_status:>10} | {cos_indicator}{cos:>6.3f} | {dist:>10.2f} | {dist_ok:>8} | {cos_ok:>7} | {rescue_status:>7}")
        
        print(f"{'─'*70}")
        
        # Summary statistics
        flagged_ids = [i for i, cid in enumerate(client_ids) if layer1_results.get(cid) == "FLAGGED"]
        if flagged_ids:
            flagged_cosines = cosines[flagged_ids]
            print(f"   📊 FLAGGED clients cosine stats:")
            print(f"      Min: {np.min(flagged_cosines):.3f}, Max: {np.max(flagged_cosines):.3f}, Mean: {np.mean(flagged_cosines):.3f}")
            
            eligible_rescue = sum(1 for i in flagged_ids if cosines[i] > self.rescue_cosine_threshold)
            print(f"      Eligible for rescue (cos > {self.rescue_cosine_threshold}): {eligible_rescue}/{len(flagged_ids)}")
            
            if ground_truth:
                flagged_malicious = sum(1 for i in flagged_ids if ground_truth[i])
                flagged_benign = len(flagged_ids) - flagged_malicious
                print(f"      Ground truth: {flagged_malicious} malicious, {flagged_benign} benign")
                
                # Analyze which would be rescued
                rescue_eligible_mal = sum(1 for i in flagged_ids if ground_truth[i] and cosines[i] > self.rescue_cosine_threshold)
                rescue_eligible_ben = sum(1 for i in flagged_ids if not ground_truth[i] and cosines[i] > self.rescue_cosine_threshold)
                print(f"      Would rescue: {rescue_eligible_ben} benign (correct), {rescue_eligible_mal} malicious (wrong)")
        
        print(f"{'─'*70}\n")

    def _compute_metrics(
        self, 
        gradients: List[np.ndarray],
        reference_vector: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tính Euclidean distance và Cosine similarity.
        Hỗ trợ External Reference (Momentum).
        """
        # Stack gradients thành ma trận (N x D)
        # Với Full Model: D ~ 3.2 triệu.
        # Lưu ý: Có thể tốn RAM, nhưng numpy xử lý tốt nếu RAM > 4GB.
        grad_matrix = np.vstack([g.flatten() for g in gradients])
        
        # Xác định Vector Tham Chiếu (Reference Vector)
        if reference_vector is not None:
            # Ưu tiên 1: Dùng Lịch sử từ Server (Chống >50% Attack)
            median_grad = reference_vector
        else:
            # Ưu tiên 2: Dùng Median của vòng hiện tại (Fallback)
            median_grad = np.median(grad_matrix, axis=0)
        
        # Euclidean distances: ||gi - gref||
        distances = np.array([
            np.linalg.norm(g - median_grad) for g in grad_matrix
        ])
        
        # Cosine similarities: dot(gi, gref) / (|gi|*|gref|)
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
        fail_distance: bool,
        can_rescue: bool
    ) -> Tuple[Layer2Result, Optional[SuspicionLevel]]:
        """
        Áp dụng ma trận quyết định
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
            if self.enable_rescue and can_rescue:
                return Layer2Result.ACCEPTED, SuspicionLevel.SUSPICIOUS
            else:
                return Layer2Result.REJECTED, None
        
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
        rejected_count = sum(1 for s in final_status.values() if s == "REJECTED")
        accepted_count = sum(1 for s in final_status.values() if s == "ACCEPTED")
        
        clean_count = sum(1 for s in suspicion_levels.values() if s == "clean")
        suspicious_count = sum(1 for s in suspicion_levels.values() if s == "suspicious")
        
        rescued = sum(
            1 for cid in client_ids 
            if layer1_results.get(cid) == "FLAGGED" and final_status.get(cid) == "ACCEPTED"
        )
        
        confirmed = sum(
            1 for cid in client_ids
            if layer1_results.get(cid) == "FLAGGED" and final_status.get(cid) == "REJECTED"
        )
        
        flagged_count = sum(1 for cid in client_ids if layer1_results.get(cid) == "FLAGGED")
        
        print(f"\n{'='*60}")
        print(f"📊 Layer 2 Summary - Round {current_round}")
        print(f"{'='*60}")
        print(f"   Final Status:")
        print(f"      REJECTED: {rejected_count}")
        print(f"      ACCEPTED: {accepted_count}")
        print(f"   Suspicion Levels (ACCEPTED only):")
        print(f"      Clean: {clean_count}")
        print(f"      Suspicious: {suspicious_count}")
        print(f"   Layer 1 → Layer 2 Changes:")
        print(f"      FLAGGED from L1: {flagged_count}")
        print(f"      Rescued (FLAGGED→ACCEPTED): {rescued}")
        print(f"      Confirmed (FLAGGED→REJECTED): {confirmed}")
        
        if self.enable_rescue:
            rescue_rate = rescued / flagged_count if flagged_count > 0 else 0
            print(f"      Rescue Rate: {rescue_rate:.1%} ({rescued}/{flagged_count})")
        
        if ground_truth:
            actual_malicious = [i for i, m in enumerate(ground_truth) if m]
            detected = [i for i, cid in enumerate(client_ids) 
                       if final_status.get(cid) == "REJECTED"]
            
            tp = len(set(actual_malicious) & set(detected))
            fp = len(set(detected) - set(actual_malicious))
            fn = len(set(actual_malicious) - set(detected))
            
            detection_rate = tp / len(actual_malicious) if actual_malicious else 0
            benign_count = len(ground_truth) - len(actual_malicious)
            fpr = fp / benign_count if benign_count > 0 else 0
            
            print(f"\n   📈 Final Metrics (after Layer 2):")
            print(f"      Detection Rate: {detection_rate:.1%} ({tp}/{len(actual_malicious)})")
            print(f"      False Positive Rate: {fpr:.1%} ({fp}/{benign_count})")
            print(f"      Missed (FN): {fn}")
            
            if rescued > 0:
                rescued_clients = [
                    cid for cid in client_ids 
                    if layer1_results.get(cid) == "FLAGGED" and final_status.get(cid) == "ACCEPTED"
                ]
                rescued_indices = [client_ids.index(cid) for cid in rescued_clients]
                rescued_malicious = sum(1 for idx in rescued_indices if ground_truth[idx])
                rescued_benign = rescued - rescued_malicious
                
                print(f"      🔍 Rescue Analysis:")
                print(f"         Rescued benign (correct): {rescued_benign}")
                print(f"         Rescued malicious (wrong): {rescued_malicious}")
        
        print(f"{'='*60}\n")

    def get_stats(self) -> Dict:
        """Get last detection stats."""
        return self.last_stats