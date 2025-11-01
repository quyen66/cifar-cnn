"""
Layer 1: Enhanced DBSCAN Detection
===================================
Detects obvious Byzantine and Gaussian noise attacks using:
1. Magnitude filter with MAD (Median Absolute Deviation)
2. DBSCAN clustering in reduced 20D space
3. Voting mechanism to combine both detectors

"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple

class Layer1Detector:
    """
    Enhanced DBSCAN detector với magnitude filter và voting.
    
    Components:
    - Magnitude Filter: Catches extreme outliers using MAD
    - DBSCAN: Catches clusters of attackers in 20D PCA space
    - Voting: Combines both with weighted votes
    """
    
    def __init__(self, 
                 pca_dims: int = 20,
                 dbscan_min_samples: int = 2,
                 warmup_rounds: int = 15):
        """
        Args:
            pca_dims: Number of PCA dimensions (default=20)
            dbscan_min_samples: Min samples for DBSCAN core point (default=3)
            warmup_rounds: Number of warmup rounds with loose thresholds (default=10)
        """
        self.pca_dims = pca_dims
        self.dbscan_min_samples = dbscan_min_samples
        self.warmup_rounds = warmup_rounds
        
        # PCA projector (fitted on first call)
        self.pca = None
        
        print(f"✅ Layer1Detector initialized:")
        print(f"   - PCA dimensions: {pca_dims}")
        print(f"   - DBSCAN min_samples: {dbscan_min_samples}")
        print(f"   - Warmup rounds: {warmup_rounds}")
    
    def detect(self, 
               gradients: List[np.ndarray],
               is_malicious_ground_truth: List[bool], # Thêm ground truth để debug
               client_ids: List[int],
               current_round: int) -> Dict[int, bool]:
        """
        Detect malicious clients using Layer 1.
        
        Args:
            gradients: List of gradient arrays from clients
            client_ids: List of client IDs
            current_round: Current training round
            
        Returns:
            Dictionary mapping client_id -> is_malicious (True/False)
        """
        n = len(gradients)
        
        # Determine if in warmup phase
        is_warmup = current_round <= self.warmup_rounds
        
        # Step 1: Magnitude Filter
        magnitude_flags, magnitude_votes = self._magnitude_filter(
            gradients, is_warmup=is_warmup
        )
        
        # Step 2: DBSCAN Clustering
        dbscan_flags, dbscan_votes = self._dbscan_filter(
            gradients, is_warmup=is_warmup
        )
        
        # Step 3: Voting Mechanism
        layer1_flags = self._voting_mechanism(
            magnitude_votes, dbscan_votes, is_warmup=is_warmup
        )
        
        # Map to client IDs
        detection_results = {
            client_ids[i]: layer1_flags[i] 
            for i in range(n)
        }
        
        # Log detection results
        num_detected = sum(layer1_flags)
        if num_detected > 0:
            detected_ids = [client_ids[i] for i in range(n) if layer1_flags[i]]
            print(f"🚨 Layer 1 detected {num_detected}/{n} malicious clients: {detected_ids}")
        
            # SAU KHI TÍNH TOÁN XONG, THÊM ĐOẠN LOG NÀY
        print(f"\n===== DEBUG Round {current_round} =====")
        print(f"{'CID':<5} {'Malicious?':<12} {'Norm':<10} {'Mag Vote':<10} {'DBSCAN Vote':<12} {'Total Vote':<12} {'Detected?':<10}")
        
        norms = np.array([np.linalg.norm(g) for g in gradients])
        total_votes = [magnitude_votes[i] + dbscan_votes[i] for i in range(len(gradients))]
        
        for i, cid in enumerate(client_ids):
            detected_flag = layer1_flags[i]
            print(f"{cid:<5} {str(is_malicious_ground_truth[i]):<12} {norms[i]:<10.2f} {magnitude_votes[i]:<10} {dbscan_votes[i]:<12} {total_votes[i]:<12} {str(detected_flag):<10}")
        print("===================================\n")

        return detection_results
    
    def _magnitude_filter(self, 
                         gradients: List[np.ndarray],
                         is_warmup: bool) -> Tuple[List[bool], List[int]]:
        """
        Magnitude filter using MAD (Median Absolute Deviation).
        
        Byzantine attacks often have extremely large gradient norms.
        MAD is more robust than std to outliers.
        
        Returns:
            flags: List of boolean flags
            votes: List of votes (2 if flagged, 0 otherwise)
        """
        # Compute L2 norms
        norms = np.array([np.linalg.norm(g) for g in gradients])
        
        # Compute MAD
        median_norm = np.median(norms)
        mad = np.median(np.abs(norms - median_norm))
        
        # Threshold with adaptive k
        # Warmup: k=6.0 (looser), Normal: k=4.0 (stricter)
        k = 6.0 if is_warmup else 4.0
        threshold = median_norm + k * mad
        
        # Flag outliers
        flags = [norm > threshold for norm in norms]
        
        # Magnitude votes: weight=2 (high confidence)
        votes = [2 if flag else 0 for flag in flags]
        
        # Log if any flagged
        num_flagged = sum(flags)
        if num_flagged > 0:
            print(f"   [Magnitude] Flagged {num_flagged}/{len(norms)} clients "
                  f"(threshold={threshold:.2f}, k={k})")
        
        return flags, votes
    
    def _dbscan_filter(self,
                      gradients: List[np.ndarray],
                      is_warmup: bool) -> Tuple[List[bool], List[int]]:
        """
        DBSCAN clustering in 20D PCA space.
        
        Detects groups of attackers that form small clusters or outliers.
        Using PCA for efficiency (O(n²) in 20D vs O(n²) in 100k D).
        
        Returns:
            flags: List of boolean flags
            votes: List of votes (1 if flagged, 0 otherwise)
        """
        n = len(gradients)
        
        # Stack gradients into matrix
        grad_matrix = np.vstack([g.flatten() for g in gradients])
        
        # PCA to 20D
        if self.pca is None:
            self.pca = PCA(n_components=self.pca_dims)
            grad_20d = self.pca.fit_transform(grad_matrix)
        else:
            grad_20d = self.pca.transform(grad_matrix)
        
        # Compute pairwise distances for eps
        pairwise_dists = squareform(pdist(grad_20d, metric='euclidean'))
        
        # Set eps to 0.5 × median pairwise distance
        # This adapts to the spread of the data
        median_dist = np.median(pairwise_dists[np.triu_indices(n, k=1)])
        eps = 0.5 * median_dist
        
        # Run DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=self.dbscan_min_samples)
        labels = dbscan.fit_predict(grad_20d)
        
        # Flag outliers and small clusters
        flags = []
        for i in range(n):
            if labels[i] == -1:
                # Outlier (not in any cluster)
                flags.append(True)
            else:
                # Check cluster size
                cluster_size = np.sum(labels == labels[i])
                if cluster_size < 3:
                    # Small cluster (potential attacker group)
                    flags.append(True)
                else:
                    flags.append(False)
        
        # DBSCAN votes: weight=1 (moderate confidence)
        votes = [1 if flag else 0 for flag in flags]
        
        # Log results
        num_outliers = sum(labels == -1)
        num_small_clusters = sum([1 for i in range(n) 
                                  if labels[i] != -1 and 
                                  np.sum(labels == labels[i]) < 3])
        num_flagged = sum(flags)
        
        if num_flagged > 0:
            print(f"   [DBSCAN] Flagged {num_flagged}/{n} clients "
                  f"(outliers={num_outliers}, small_clusters={num_small_clusters}, eps={eps:.2f})")
        
        return flags, votes
    
    def _voting_mechanism(self,
                         magnitude_votes: List[int],
                         dbscan_votes: List[int],
                         is_warmup: bool) -> List[bool]:
        """
        Combine magnitude and DBSCAN votes.
        
        Voting threshold:
        - Warmup: 3 (need both to flag)
        - Normal: 2 (magnitude alone OR both detectors)
        
        Why weighted votes?
        - Magnitude (weight=2): Very reliable if norm is extreme
        - DBSCAN (weight=1): Can flag benign non-IID clients
        
        Returns:
            flags: Final boolean flags after voting
        """
        n = len(magnitude_votes)
        
        # Compute total votes
        total_votes = [magnitude_votes[i] + dbscan_votes[i] 
                      for i in range(n)]
        
        # Voting threshold
        # Warmup: threshold=3 (strict, avoid false positives)
        # Normal: threshold=2 (balanced)
        threshold = 3 if is_warmup else 1
        
        # Flag if votes >= threshold
        flags = [votes >= threshold for votes in total_votes]
        
        return flags
    
    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            'pca_dims': self.pca_dims,
            'dbscan_min_samples': self.dbscan_min_samples,
            'warmup_rounds': self.warmup_rounds,
            'pca_fitted': self.pca is not None
        }

