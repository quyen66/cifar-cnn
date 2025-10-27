"""
Comprehensive Hyperparameter Grid Search cho Layer 1 DBSCAN
============================================================

Test tất cả combinations của hyperparameters để tìm optimal settings.
Với config: 40 clients, fraction_fit=0.6 → ~24 clients/round

Hyperparameters cần test:
1. PCA dims: 20 (FIXED)
2. DBSCAN minPts: [2, 3, 4, 5]
3. DBSCAN eps multiplier: [0.3, 0.4, 0.5, 0.6, 0.7]
4. MAD multiplier k (normal): [3.0, 4.0, 5.0, 6.0]
5. MAD multiplier k (warmup): [5.0, 6.0, 7.0, 8.0]
6. Voting threshold (normal): [1, 2, 3]
7. Voting threshold (warmup): [2, 3, 4]

Total combinations: 4 × 5 × 4 × 4 × 3 × 3 = 2,880 tests
Running time: ~2-3 giờ với synthetic data
"""

import numpy as np
import itertools
from typing import Dict, List, Tuple
import json
from datetime import datetime
import os

# Import Layer1Detector (modify để accept params)
import sys
sys.path.insert(0, '/mnt/project')

class ConfigurableLayer1Detector:
    """Layer 1 Detector với configurable hyperparameters."""
    
    def __init__(self,
                 pca_dims: int = 20,
                 dbscan_min_samples: int = 3,
                 dbscan_eps_multiplier: float = 0.5,
                 mad_k_normal: float = 4.0,
                 mad_k_warmup: float = 6.0,
                 voting_threshold_normal: int = 2,
                 voting_threshold_warmup: int = 3,
                 warmup_rounds: int = 10):
        
        from sklearn.decomposition import PCA
        from sklearn.cluster import DBSCAN
        from scipy.spatial.distance import pdist, squareform
        
        self.pca_dims = pca_dims
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_eps_multiplier = dbscan_eps_multiplier
        self.mad_k_normal = mad_k_normal
        self.mad_k_warmup = mad_k_warmup
        self.voting_threshold_normal = voting_threshold_normal
        self.voting_threshold_warmup = voting_threshold_warmup
        self.warmup_rounds = warmup_rounds
        
        self.pca = None
        self.PCA = PCA
        self.DBSCAN = DBSCAN
        self.pdist = pdist
        self.squareform = squareform
    
    def detect(self, gradients: List[np.ndarray], current_round: int = 15) -> List[bool]:
        """Detect malicious clients."""
        n = len(gradients)
        is_warmup = current_round <= self.warmup_rounds
        
        # Magnitude filter
        mag_flags, mag_votes = self._magnitude_filter(gradients, is_warmup)
        
        # DBSCAN filter
        dbscan_flags, dbscan_votes = self._dbscan_filter(gradients)
        
        # Voting
        total_votes = [mag_votes[i] + dbscan_votes[i] for i in range(n)]
        threshold = self.voting_threshold_warmup if is_warmup else self.voting_threshold_normal
        flags = [votes >= threshold for votes in total_votes]
        
        return flags
    
    def _magnitude_filter(self, gradients: List[np.ndarray], is_warmup: bool) -> Tuple[List[bool], List[int]]:
        """Magnitude filter với MAD."""
        norms = np.array([np.linalg.norm(g) for g in gradients])
        median_norm = np.median(norms)
        mad = np.median(np.abs(norms - median_norm))
        
        k = self.mad_k_warmup if is_warmup else self.mad_k_normal
        threshold = median_norm + k * mad
        
        flags = [norm > threshold for norm in norms]
        votes = [2 if flag else 0 for flag in flags]
        
        return flags, votes
    
    def _dbscan_filter(self, gradients: List[np.ndarray]) -> Tuple[List[bool], List[int]]:
        """DBSCAN clustering."""
        n = len(gradients)
        grad_matrix = np.vstack([g.flatten() for g in gradients])
        
        # PCA
        if self.pca is None:
            # Dynamic PCA dims
            actual_dims = min(self.pca_dims, n - 1, grad_matrix.shape[1])
            self.pca = self.PCA(n_components=actual_dims)
            grad_20d = self.pca.fit_transform(grad_matrix)
        else:
            grad_20d = self.pca.transform(grad_matrix)
        
        # Compute eps
        pairwise_dists = self.squareform(self.pdist(grad_20d, metric='euclidean'))
        median_dist = np.median(pairwise_dists[np.triu_indices(n, k=1)])
        eps = self.dbscan_eps_multiplier * median_dist
        
        # DBSCAN
        dbscan = self.DBSCAN(eps=eps, min_samples=self.dbscan_min_samples)
        labels = dbscan.fit_predict(grad_20d)
        
        # Flag outliers and small clusters
        flags = []
        for i in range(n):
            if labels[i] == -1:
                flags.append(True)
            else:
                cluster_size = np.sum(labels == labels[i])
                if cluster_size < 3:
                    flags.append(True)
                else:
                    flags.append(False)
        
        votes = [1 if flag else 0 for flag in flags]
        return flags, votes


def generate_test_data(n_benign: int, n_malicious: int, dim: int, 
                       attack_type: str = 'byzantine') -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate synthetic test data.
    
    Args:
        n_benign: Number of benign clients
        n_malicious: Number of malicious clients
        dim: Gradient dimension
        attack_type: 'byzantine', 'gaussian', or 'label_flip'
    
    Returns:
        gradients: List of gradient arrays
        malicious_ids: List of malicious client indices
    """
    # Benign gradients: small magnitude, similar direction
    benign_grads = [np.random.randn(dim) * 0.1 for _ in range(n_benign)]
    
    # Malicious gradients
    if attack_type == 'byzantine':
        # Byzantine: very large magnitude, random direction
        malicious_grads = [np.random.randn(dim) * 10.0 for _ in range(n_malicious)]
    elif attack_type == 'gaussian':
        # Gaussian noise: moderate magnitude, random direction
        malicious_grads = [np.random.randn(dim) * 2.0 for _ in range(n_malicious)]
    elif attack_type == 'label_flip':
        # Label flip: similar magnitude, opposite direction
        base_grad = np.random.randn(dim) * 0.1
        malicious_grads = [-base_grad + np.random.randn(dim) * 0.05 
                          for _ in range(n_malicious)]
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    # Combine
    all_grads = benign_grads + malicious_grads
    malicious_ids = list(range(n_benign, n_benign + n_malicious))
    
    return all_grads, malicious_ids


def compute_metrics(detected: List[bool], 
                    malicious_ids: List[int],
                    n_total: int) -> Dict[str, float]:
    """Compute detection metrics."""
    detected_set = set([i for i, flag in enumerate(detected) if flag])
    malicious_set = set(malicious_ids)
    benign_set = set(range(n_total)) - malicious_set
    
    tp = len(detected_set & malicious_set)
    fp = len(detected_set & benign_set)
    fn = len(malicious_set - detected_set)
    tn = len(benign_set - detected_set)
    
    detection_rate = tp / len(malicious_set) if malicious_set else 0
    fpr = fp / len(benign_set) if benign_set else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = detection_rate
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'detection_rate': detection_rate,
        'fpr': fpr,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def run_comprehensive_grid_search():
    """Run comprehensive grid search."""
    
    print("="*70)
    print("COMPREHENSIVE HYPERPARAMETER GRID SEARCH")
    print("="*70)
    
    # Define hyperparameter grid
    param_grid = {
        'dbscan_min_samples': [2, 3, 4, 5],
        'dbscan_eps_multiplier': [0.3, 0.4, 0.5, 0.6, 0.7],
        'mad_k_normal': [3.0, 4.0, 5.0, 6.0],
        'mad_k_warmup': [5.0, 6.0, 7.0, 8.0],
        'voting_threshold_normal': [1, 2, 3],
        'voting_threshold_warmup': [2, 3, 4]
    }
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))
    
    print(f"\n📊 Grid Search Configuration:")
    print(f"   Total combinations: {len(combinations)}")
    for name, values in param_grid.items():
        print(f"   - {name}: {values}")
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Byzantine 30%',
            'n_benign': 18,
            'n_malicious': 6,
            'attack_type': 'byzantine',
            'dim': 10000
        },
        {
            'name': 'Gaussian 30%',
            'n_benign': 18,
            'n_malicious': 6,
            'attack_type': 'gaussian',
            'dim': 10000
        },
        {
            'name': 'Label Flip 30%',
            'n_benign': 18,
            'n_malicious': 6,
            'attack_type': 'label_flip',
            'dim': 10000
        }
    ]
    
    print(f"\n🎯 Test Scenarios: {len(test_scenarios)}")
    for scenario in test_scenarios:
        print(f"   - {scenario['name']}: {scenario['n_benign']} benign + "
              f"{scenario['n_malicious']} malicious")
    
    # Run grid search
    results = []
    total_tests = len(combinations) * len(test_scenarios)
    
    print(f"\n🔄 Running {total_tests} tests...")
    print(f"   Estimated time: {total_tests * 0.5 / 60:.1f} minutes")
    
    for idx, combo in enumerate(combinations):
        # Create param dict
        params = dict(zip(param_names, combo))
        params['pca_dims'] = 20  # Fixed
        params['warmup_rounds'] = 10  # Fixed
        
        # Test on each scenario
        scenario_results = {}
        for scenario in test_scenarios:
            # Generate data
            np.random.seed(42)  # For reproducibility
            gradients, malicious_ids = generate_test_data(
                scenario['n_benign'],
                scenario['n_malicious'],
                scenario['dim'],
                scenario['attack_type']
            )
            
            # Test warmup and normal
            for phase in ['warmup', 'normal']:
                current_round = 5 if phase == 'warmup' else 15
                
                # Create detector with these params
                detector = ConfigurableLayer1Detector(**params)
                
                # Detect
                detected = detector.detect(gradients, current_round)
                
                # Compute metrics
                metrics = compute_metrics(
                    detected,
                    malicious_ids,
                    len(gradients)
                )
                
                scenario_results[f"{scenario['name']}_{phase}"] = metrics
        
        # Store result
        result = {
            'params': params,
            'scenarios': scenario_results
        }
        results.append(result)
        
        # Progress
        if (idx + 1) % 100 == 0:
            print(f"   Progress: {idx + 1}/{len(combinations)} combinations tested")
    
    # Analyze results
    print("\n\n" + "="*70)
    print("ANALYSIS: TOP 10 CONFIGURATIONS")
    print("="*70)
    
    # Score each configuration
    for result in results:
        # Average F1 score across all scenarios
        f1_scores = []
        for scenario_name, metrics in result['scenarios'].items():
            f1_scores.append(metrics['f1'])
        
        result['avg_f1'] = np.mean(f1_scores)
        result['avg_detection'] = np.mean([m['detection_rate'] 
                                          for m in result['scenarios'].values()])
        result['avg_fpr'] = np.mean([m['fpr'] 
                                     for m in result['scenarios'].values()])
    
    # Sort by F1 score
    results_sorted = sorted(results, key=lambda x: x['avg_f1'], reverse=True)
    
    # Print top 10
    print("\n🏆 TOP 10 CONFIGURATIONS (by F1 score):\n")
    for rank, result in enumerate(results_sorted[:10], 1):
        params = result['params']
        print(f"{rank}. F1={result['avg_f1']:.3f} | "
              f"Detection={result['avg_detection']:.1%} | "
              f"FPR={result['avg_fpr']:.1%}")
        print(f"   Parameters:")
        for key, value in params.items():
            if key not in ['pca_dims', 'warmup_rounds']:
                print(f"      {key}: {value}")
        print()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"layer1_grid_search_comprehensive_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'param_grid': param_grid,
            'test_scenarios': test_scenarios,
            'results': results_sorted[:100],  # Save top 100
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"   Top 100 configurations saved")
    
    # Print recommendation
    best = results_sorted[0]
    print("\n\n" + "="*70)
    print("🎯 RECOMMENDED CONFIGURATION")
    print("="*70)
    print(f"\nF1 Score: {best['avg_f1']:.3f}")
    print(f"Detection Rate: {best['avg_detection']:.1%}")
    print(f"False Positive Rate: {best['avg_fpr']:.1%}")
    print("\nParameters:")
    for key, value in best['params'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE GRID SEARCH COMPLETE!")
    print("="*70)
    
    return results_sorted


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "LAYER 1 HYPERPARAMETER TUNING" + " "*24 + "║")
    print("║" + " "*20 + "Comprehensive Grid Search" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    # Run grid search
    results = run_comprehensive_grid_search()
    
    print("\n🎉 All done! Check the saved JSON file for detailed results.\n")