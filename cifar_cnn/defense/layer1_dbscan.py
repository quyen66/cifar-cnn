# cifar_cnn/defense/layer1_dbscan.py

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from typing import List, Dict, Optional
from logging import INFO, DEBUG
from flwr.common.logger import log

class Layer1Detector:
    def __init__(
        self,
        pca_dims: int = 20, # Đây sẽ đóng vai trò là "Ngưỡng trần" (Target Max)
        dbscan_min_samples: int = 3,
        dbscan_eps_multiplier: float = 0.5,
        mad_k_normal: float = 4.0,
        mad_k_warmup: float = 6.0,
        voting_threshold_normal: int = 2,
        voting_threshold_warmup: int = 3,
        warmup_rounds: int = 10
    ):
        self.target_pca_dims = pca_dims
        self.min_samples = dbscan_min_samples
        self.eps_multiplier = dbscan_eps_multiplier
        self.mad_k_normal = mad_k_normal
        self.mad_k_warmup = mad_k_warmup
        self.voting_threshold_normal = voting_threshold_normal
        self.voting_threshold_warmup = voting_threshold_warmup
        self.warmup_rounds = warmup_rounds
        
        # Stats
        self.pca_fitted = False
        self.last_noise_count = 0

    def detect(
        self,
        gradients: List[np.ndarray],
        client_ids: List[int],
        is_malicious_ground_truth: Optional[List[bool]] = None,
        current_round: int = 0
    ) -> Dict[int, bool]:
        """
        Phát hiện tấn công bằng kết hợp Magnitude Filter và DBSCAN trên PCA.
        """
        num_clients = len(gradients)
        if num_clients < 2:
            return {cid: False for cid in client_ids}

        # 1. Magnitude Check (Bộ lọc độ lớn)
        # ---------------------------------------------------------
        norms = np.linalg.norm(gradients, axis=1)
        median_norm = np.median(norms)
        mad_norm = np.median(np.abs(norms - median_norm))
        
        # Chọn ngưỡng K tùy theo giai đoạn (Warmup nới lỏng hơn)
        k_mad = self.mad_k_warmup if current_round < self.warmup_rounds else self.mad_k_normal
        
        # Tránh chia cho 0 nếu tất cả gradient giống hệt nhau
        effective_mad = mad_norm if mad_norm > 1e-9 else 1.0
        threshold_mag = median_norm + k_mad * effective_mad
        
        mag_flags = [norm > threshold_mag for norm in norms]

        # 2. PCA + DBSCAN Clustering
        # ---------------------------------------------------------
        # FIX: Dynamic PCA Dims Calculation
        # Logic: 
        # - Target: self.target_pca_dims (20)
        # - Density constraint: 50% số lượng client (để không gian đủ đặc)
        # - Hard constraint: min(num_clients, n_features)
        
        density_constrained_dims = max(2, int(num_clients * 0.5))
        optimal_dims = min(self.target_pca_dims, density_constrained_dims)
        
        # Đảm bảo không vượt quá số lượng mẫu thực tế (Sklearn requirement)
        final_pca_dims = min(optimal_dims, num_clients)
        
        # Chỉ chạy PCA nếu có đủ ít nhất 2 client
        dbscan_flags = [False] * num_clients
        
        if final_pca_dims >= 2:
            try:
                # Dùng RandomizedPCA cho nhanh nếu dữ liệu lớn, hoặc mặc định
                pca = PCA(n_components=final_pca_dims, svd_solver='randomized', random_state=42)
                reduced_grads = pca.fit_transform(gradients)
                self.pca_fitted = True
                
                # Tính Epsilon động cho DBSCAN
                # Dùng khoảng cách k-nearest neighbors trung bình để ước lượng epsilon
                # Ở đây dùng heuristic đơn giản: 0.5 * median pairwise distance
                from sklearn.metrics.pairwise import euclidean_distances
                dists = euclidean_distances(reduced_grads)
                median_dist = np.median(dists)
                eps = median_dist * self.eps_multiplier
                
                # Nếu eps quá nhỏ (các điểm trùng nhau), set min threshold
                if eps < 1e-6: eps = 0.5 

                # Chạy DBSCAN
                clustering = DBSCAN(eps=eps, min_samples=self.min_samples, metric='euclidean')
                labels = clustering.fit_predict(reduced_grads)
                
                # Label -1 là nhiễu (outlier)
                dbscan_flags = [label == -1 for label in labels]
                self.last_noise_count = sum(dbscan_flags)
                
            except Exception as e:
                log(INFO, f"⚠️ PCA/DBSCAN failed: {e}. Fallback to Magnitude only.")
                dbscan_flags = [False] * num_clients
        
        # 3. Voting Mechanism (Kết hợp kết quả)
        # ---------------------------------------------------------
        # Normal mode: Cần cả 2 bộ lọc đồng ý (AND) hoặc 1 cái quá tệ?
        # Đề xuất: Union (OR) để an toàn cao nhất, HOẶC Voting có trọng số.
        # Ở đây dùng logic:
        # - Nếu Magnitude quá lớn -> Flag ngay.
        # - Nếu DBSCAN thấy tách biệt -> Flag.
        # -> Dùng OR (Union) là an toàn nhất cho Defense.
        
        final_flags = {}
        for i, cid in enumerate(client_ids):
            # Kết hợp: Bị flag nếu vi phạm Magnitude HOẶC bị DBSCAN cô lập
            is_attack = mag_flags[i] or dbscan_flags[i]
            final_flags[cid] = is_attack
            
        return final_flags

    def get_stats(self) -> Dict:
        return {
            "pca_fitted": self.pca_fitted,
            "last_noise_count": self.last_noise_count
        }