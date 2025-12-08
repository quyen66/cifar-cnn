"""
Layer 1: Enhanced DBSCAN Detection (V2 - main.pdf)
====================================================
Giai đoạn 2: Hai lớp phát hiện tuần tự

Pipeline:
1. Magnitude Filter (không gian gốc) → 3 trạng thái
   - REJECTED: ||gi|| > Median + 15×MAD (loại ngay, bỏ qua DBSCAN)
   - FLAGGED: ||gi|| > Median + 4×MAD
   - ACCEPTED: ||gi|| ≤ Median + 4×MAD

2. DBSCAN Clustering (CHỈ cho clients KHÔNG bị REJECTED)
   - PCA giảm chiều: dpca = min(20, floor(0.5×n_remaining))
   - DBSCAN: ε = 0.5×median_dist, minPts = 3
   - Outliers (label=-1) → FLAGGED

Output: Dict[client_id, status] với status ∈ {"REJECTED", "FLAGGED", "ACCEPTED"}

LƯU Ý: Module này CHỈ CHẠY SAU WARMUP (vòng 11+)
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Layer1Status(Enum):
    """Trạng thái của client sau Layer 1."""
    REJECTED = "REJECTED"   # Loại ngay, không qua Layer 2
    FLAGGED = "FLAGGED"     # Nghi ngờ, cần Layer 2 kiểm tra
    ACCEPTED = "ACCEPTED"   # Tạm chấp nhận


@dataclass
class Layer1Result:
    """Kết quả chi tiết của Layer 1."""
    status: Dict[int, str]              # client_id -> status string
    magnitude_stats: Dict                # Thống kê magnitude filter
    dbscan_stats: Dict                   # Thống kê DBSCAN
    rejected_ids: List[int]             # IDs bị REJECTED
    flagged_ids: List[int]              # IDs bị FLAGGED
    accepted_ids: List[int]             # IDs được ACCEPTED


class Layer1Detector:
    """
    Layer 1 Detector với 3 trạng thái (V2 - main.pdf).
    
    Đặc điểm:
    - Magnitude Filter bảo vệ PCA khỏi outliers cực đoan
    - DBSCAN chỉ chạy trên clients không bị REJECTED (tiết kiệm tài nguyên)
    - PCA dims động theo số clients còn lại
    """
    
    def __init__(
        self,
        pca_dims: int = 20,
        dbscan_min_samples: int = 3,
        dbscan_eps_multiplier: float = 0.5,
        mad_k_reject: float = 15.0,
        mad_k_flag: float = 4.0,
        voting_threshold: int = 2
    ):
        """
        Initialize Layer 1 Detector.
        
        Args:
            pca_dims: Target max PCA dimensions
            dbscan_min_samples: minPts for DBSCAN (default=3)
            dbscan_eps_multiplier: eps = mult × median_dist (default=0.5)
            mad_k_reject: k for REJECTED threshold (default=15)
            mad_k_flag: k for FLAGGED threshold (default=4)
            voting_threshold: Unused, kept for compatibility
        """
        self.target_pca_dims = pca_dims
        self.min_samples = dbscan_min_samples
        self.eps_multiplier = dbscan_eps_multiplier
        self.mad_k_reject = mad_k_reject
        self.mad_k_flag = mad_k_flag
        
        # Stats
        self.last_result: Optional[Layer1Result] = None
        
        print(f"✅ Layer1Detector V2 initialized:")
        print(f"   Magnitude: k_reject={mad_k_reject}, k_flag={mad_k_flag}")
        print(f"   DBSCAN: minPts={dbscan_min_samples}, eps_mult={dbscan_eps_multiplier}")
        print(f"   PCA dims (target): {pca_dims}")

    def detect(
        self,
        gradients: List[np.ndarray],
        client_ids: List[int],
        current_round: int = 0,
        is_malicious_ground_truth: Optional[List[bool]] = None
    ) -> Dict[int, str]:
        """
        Phát hiện và phân loại clients thành 3 trạng thái.
        
        Pipeline theo main.pdf:
        1. Magnitude Filter → REJECTED/FLAGGED/ACCEPTED
        2. DBSCAN (chỉ cho non-REJECTED) → có thể nâng lên FLAGGED
        
        Args:
            gradients: List gradient arrays từ clients
            client_ids: List client IDs
            current_round: Round hiện tại (for logging)
            is_malicious_ground_truth: Ground truth (optional, for metrics)
            
        Returns:
            Dict[client_id, status] với status ∈ {"REJECTED", "FLAGGED", "ACCEPTED"}
        """
        num_clients = len(gradients)
        
        # Edge case
        if num_clients < 2:
            result = {cid: Layer1Status.ACCEPTED.value for cid in client_ids}
            self.last_result = Layer1Result(
                status=result,
                magnitude_stats={},
                dbscan_stats={},
                rejected_ids=[],
                flagged_ids=[],
                accepted_ids=list(client_ids)
            )
            return result
        
        # =========================================================
        # STEP 1: Magnitude Filter (3 ngưỡng)
        # =========================================================
        mag_status, mag_stats = self._magnitude_filter(gradients, client_ids)
        
        # Phân loại theo magnitude
        rejected_indices = [i for i, s in enumerate(mag_status) if s == Layer1Status.REJECTED]
        non_rejected_indices = [i for i, s in enumerate(mag_status) if s != Layer1Status.REJECTED]
        
        # =========================================================
        # STEP 2: DBSCAN (CHỈ cho clients KHÔNG bị REJECTED)
        # =========================================================
        # Theo PDF: "Kiểm tra mật độ DBSCAN: Áp dụng cho các máy khách 
        # chưa bị gắn cờ REJECTED"
        
        dbscan_stats = {"skipped": True, "reason": "No non-rejected clients"}
        
        if len(non_rejected_indices) >= 2:
            # Lấy gradients của clients không bị REJECTED
            non_rejected_grads = [gradients[i] for i in non_rejected_indices]
            non_rejected_cids = [client_ids[i] for i in non_rejected_indices]
            
            # Chạy DBSCAN
            dbscan_flags, dbscan_stats = self._dbscan_filter(
                non_rejected_grads, 
                non_rejected_cids,
                len(non_rejected_indices)
            )
            
            # Cập nhật status: Nếu DBSCAN flag → FLAGGED
            for idx, orig_idx in enumerate(non_rejected_indices):
                if dbscan_flags[idx]:
                    # DBSCAN thấy outlier → nâng lên FLAGGED
                    mag_status[orig_idx] = Layer1Status.FLAGGED
        
        # =========================================================
        # STEP 3: Build final result
        # =========================================================
        final_status = {
            client_ids[i]: mag_status[i].value 
            for i in range(num_clients)
        }
        
        # Categorize
        rejected_ids = [client_ids[i] for i in range(num_clients) 
                       if mag_status[i] == Layer1Status.REJECTED]
        flagged_ids = [client_ids[i] for i in range(num_clients) 
                      if mag_status[i] == Layer1Status.FLAGGED]
        accepted_ids = [client_ids[i] for i in range(num_clients) 
                       if mag_status[i] == Layer1Status.ACCEPTED]
        
        # Store result
        self.last_result = Layer1Result(
            status=final_status,
            magnitude_stats=mag_stats,
            dbscan_stats=dbscan_stats,
            rejected_ids=rejected_ids,
            flagged_ids=flagged_ids,
            accepted_ids=accepted_ids
        )
        
        # Log
        self._log_results(current_round, is_malicious_ground_truth, client_ids, mag_status)
        
        return final_status

    def _magnitude_filter(
        self, 
        gradients: List[np.ndarray],
        client_ids: List[int]
    ) -> Tuple[List[Layer1Status], Dict]:
        """
        Magnitude Filter với 3 ngưỡng.
        
        Theo PDF:
        - τ(k) = Median(||gj||) + k × MAD
        - REJECTED: ||gi|| > τ(15)
        - FLAGGED: ||gi|| > τ(4)
        - ACCEPTED: ||gi|| ≤ τ(4)
        
        Returns:
            status: List[Layer1Status]
            stats: Dict với thông tin debug
        """
        # Tính norms
        norms = np.array([np.linalg.norm(g) for g in gradients])
        
        # Tính Median và MAD
        median_norm = np.median(norms)
        mad = np.median(np.abs(norms - median_norm))
        
        # Tránh MAD = 0 (tất cả gradient giống nhau)
        effective_mad = mad if mad > 1e-9 else 1.0
        
        # Tính ngưỡng
        threshold_reject = median_norm + self.mad_k_reject * effective_mad  # k=15
        threshold_flag = median_norm + self.mad_k_flag * effective_mad      # k=4
        
        # Phân loại
        status = []
        for i, norm in enumerate(norms):
            if norm > threshold_reject:
                status.append(Layer1Status.REJECTED)
            elif norm > threshold_flag:
                status.append(Layer1Status.FLAGGED)
            else:
                status.append(Layer1Status.ACCEPTED)
        
        # Stats
        stats = {
            "median_norm": float(median_norm),
            "mad": float(mad),
            "effective_mad": float(effective_mad),
            "threshold_reject": float(threshold_reject),
            "threshold_flag": float(threshold_flag),
            "k_reject": self.mad_k_reject,
            "k_flag": self.mad_k_flag,
            "counts": {
                "rejected": sum(1 for s in status if s == Layer1Status.REJECTED),
                "flagged": sum(1 for s in status if s == Layer1Status.FLAGGED),
                "accepted": sum(1 for s in status if s == Layer1Status.ACCEPTED)
            },
            "norms": {client_ids[i]: float(norms[i]) for i in range(len(norms))}
        }
        
        return status, stats

    def _dbscan_filter(
        self, 
        gradients: List[np.ndarray],
        client_ids: List[int],
        num_clients: int
    ) -> Tuple[List[bool], Dict]:
        """
        DBSCAN clustering trên PCA space.
        
        Theo PDF:
        - dpca = min(20, floor(0.5×n))
        - ε = 0.5 × median_dist
        - minPts = 3
        - Outliers (label=-1) → FLAGGED
        
        Returns:
            flags: List[bool] - True nếu là outlier (cần FLAGGED)
            stats: Dict với thông tin debug
        """
        # Dynamic PCA dims theo PDF
        density_dims = max(2, int(num_clients * 0.5))
        actual_pca_dims = min(self.target_pca_dims, density_dims, num_clients)
        
        flags = [False] * num_clients
        stats = {
            "skipped": False,
            "num_clients_analyzed": num_clients,
            "pca_dims_target": self.target_pca_dims,
            "pca_dims_actual": actual_pca_dims,
            "eps": 0,
            "min_samples": self.min_samples,
            "outlier_count": 0,
            "cluster_count": 0
        }
        
        if actual_pca_dims < 2 or num_clients < self.min_samples:
            stats["skipped"] = True
            stats["reason"] = f"Not enough clients ({num_clients}) or dims ({actual_pca_dims})"
            return flags, stats
        
        try:
            # Stack gradients
            grad_matrix = np.vstack([g.flatten() for g in gradients])
            
            # PCA giảm chiều (Randomized cho hiệu quả)
            pca = PCA(n_components=actual_pca_dims, svd_solver='randomized', random_state=42)
            reduced = pca.fit_transform(grad_matrix)
            
            # Tính eps động: ε = 0.5 × median_dist
            dists = euclidean_distances(reduced)
            # Lấy upper triangle (không kể diagonal)
            upper_indices = np.triu_indices(num_clients, k=1)
            upper_dists = dists[upper_indices]
            
            if len(upper_dists) > 0:
                median_dist = np.median(upper_dists)
            else:
                median_dist = 1.0
                
            eps = max(self.eps_multiplier * median_dist, 1e-6)
            
            # DBSCAN
            clustering = DBSCAN(
                eps=eps, 
                min_samples=self.min_samples, 
                metric='euclidean'
            )
            labels = clustering.fit_predict(reduced)
            
            # Outliers (label=-1) → cần FLAGGED
            flags = [label == -1 for label in labels]
            
            # Count clusters (không kể noise)
            unique_labels = set(labels)
            cluster_count = len([l for l in unique_labels if l != -1])
            
            # Update stats
            stats["eps"] = float(eps)
            stats["median_dist"] = float(median_dist)
            stats["outlier_count"] = sum(flags)
            stats["cluster_count"] = cluster_count
            stats["labels"] = {client_ids[i]: int(labels[i]) for i in range(len(labels))}
            stats["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
            
        except Exception as e:
            stats["skipped"] = True
            stats["reason"] = f"Exception: {str(e)}"
            print(f"⚠️ DBSCAN failed: {e}")
            
        return flags, stats

    def _log_results(
        self,
        current_round: int,
        ground_truth: Optional[List[bool]],
        client_ids: List[int],
        status: List[Layer1Status]
    ):
        """Log kết quả detection."""
        if self.last_result is None:
            return
            
        r = self.last_result
        mag = r.magnitude_stats
        db = r.dbscan_stats
        
        print(f"\n{'='*65}")
        print(f"📊 LAYER 1 RESULTS - Round {current_round}")
        print(f"{'='*65}")
        
        # Magnitude stats
        print(f"\n🔍 Magnitude Filter:")
        print(f"   Median norm: {mag.get('median_norm', 0):.4f}")
        print(f"   MAD: {mag.get('mad', 0):.4f}")
        print(f"   Threshold REJECT (k={mag.get('k_reject')}): {mag.get('threshold_reject', 0):.4f}")
        print(f"   Threshold FLAG (k={mag.get('k_flag')}): {mag.get('threshold_flag', 0):.4f}")
        
        mag_counts = mag.get('counts', {})
        print(f"   Results: REJECTED={mag_counts.get('rejected', 0)}, "
              f"FLAGGED={mag_counts.get('flagged', 0)}, "
              f"ACCEPTED={mag_counts.get('accepted', 0)}")
        
        # DBSCAN stats
        print(f"\n🔍 DBSCAN Clustering:")
        if db.get('skipped'):
            print(f"   ⚠️ Skipped: {db.get('reason', 'Unknown')}")
        else:
            print(f"   Clients analyzed: {db.get('num_clients_analyzed', 0)}")
            print(f"   PCA dims: {db.get('pca_dims_actual', 0)} (target={db.get('pca_dims_target', 0)})")
            print(f"   eps: {db.get('eps', 0):.4f} (median_dist={db.get('median_dist', 0):.4f})")
            print(f"   Clusters found: {db.get('cluster_count', 0)}")
            print(f"   Outliers (→FLAGGED): {db.get('outlier_count', 0)}")
        
        # Final counts
        print(f"\n📋 Final Status:")
        print(f"   REJECTED: {len(r.rejected_ids)} {r.rejected_ids if len(r.rejected_ids) <= 10 else '...'}")
        print(f"   FLAGGED:  {len(r.flagged_ids)} {r.flagged_ids if len(r.flagged_ids) <= 10 else '...'}")
        print(f"   ACCEPTED: {len(r.accepted_ids)} {r.accepted_ids if len(r.accepted_ids) <= 10 else '...'}")
        
        # Metrics vs ground truth
        if ground_truth:
            num_clients = len(ground_truth)
            actual_malicious_idx = [i for i, m in enumerate(ground_truth) if m]
            detected_idx = [i for i, s in enumerate(status) if s != Layer1Status.ACCEPTED]
            
            tp = len(set(actual_malicious_idx) & set(detected_idx))
            fp = len(set(detected_idx) - set(actual_malicious_idx))
            fn = len(set(actual_malicious_idx) - set(detected_idx))
            tn = num_clients - tp - fp - fn
            
            detection_rate = tp / len(actual_malicious_idx) if actual_malicious_idx else 0
            benign_count = num_clients - len(actual_malicious_idx)
            fpr = fp / benign_count if benign_count > 0 else 0
            
            print(f"\n📈 Metrics (vs ground truth):")
            print(f"   True Positives: {tp}/{len(actual_malicious_idx)} malicious detected")
            print(f"   False Positives: {fp}/{benign_count} benign flagged")
            print(f"   Detection Rate: {detection_rate:.1%}")
            print(f"   False Positive Rate: {fpr:.1%}")
        
        print(f"{'='*65}\n")

    def get_result(self) -> Optional[Layer1Result]:
        """Get last detection result."""
        return self.last_result
    
    def get_stats(self) -> Dict:
        """Get stats as dict (for compatibility)."""
        if self.last_result is None:
            return {}
        return {
            "magnitude": self.last_result.magnitude_stats,
            "dbscan": self.last_result.dbscan_stats,
            "rejected_count": len(self.last_result.rejected_ids),
            "flagged_count": len(self.last_result.flagged_ids),
            "accepted_count": len(self.last_result.accepted_ids)
        }