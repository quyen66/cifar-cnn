"""
Quick Hyperparameter Test cho Layer 1 DBSCAN
=============================================

Test với reasonable ranges để tìm nhanh hyperparameters tốt.
Runtime: ~5-10 phút

Config: 40 clients, fraction_fit=0.6 → ~24 clients/round

Reasonable ranges (dựa trên literature và initial testing):
1. PCA dims: 20 (FIXED)
2. DBSCAN minPts: [3, 4, 5]  (với n=24, nên test 3-5)
3. DBSCAN eps multiplier: [0.4, 0.5, 0.6]  (centered around default 0.5)
4. MAD multiplier k (normal): [3.5, 4.0, 4.5]
5. Voting threshold (normal): [2]  (keep default)

Total combinations: 3 × 3 × 3 = 27 tests
"""

import numpy as np
import itertools
from typing import Dict, List, Tuple
import json
from datetime import datetime
from tabulate import tabulate

# Reuse ConfigurableLayer1Detector from comprehensive script
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
    """Generate synthetic test data."""
    benign_grads = [np.random.randn(dim) * 0.1 for _ in range(n_benign)]
    
    if attack_type == 'byzantine':
        malicious_grads = [np.random.randn(dim) * 10.0 for _ in range(n_malicious)]
    elif attack_type == 'gaussian':
        malicious_grads = [np.random.randn(dim) * 2.0 for _ in range(n_malicious)]
    elif attack_type == 'label_flip':
        base_grad = np.random.randn(dim) * 0.1
        malicious_grads = [-base_grad + np.random.randn(dim) * 0.05 
                          for _ in range(n_malicious)]
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
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


def run_quick_test():
    """Run quick hyperparameter test."""
    
    print("="*70)
    print("QUICK HYPERPARAMETER TEST")
    print("="*70)
    
    # Reasonable parameter ranges
    param_grid = {
        'dbscan_min_samples': [3, 4, 5],
        'dbscan_eps_multiplier': [0.4, 0.5, 0.6],
        'mad_k_normal': [3.5, 4.0, 4.5]
    }
    
    # Fixed parameters
    fixed_params = {
        'pca_dims': 20,
        'mad_k_warmup': 6.0,
        'voting_threshold_normal': 2,
        'voting_threshold_warmup': 3,
        'warmup_rounds': 10
    }
    
    # Generate combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))
    
    print(f"\n📊 Test Configuration:")
    print(f"   Total combinations: {len(combinations)}")
    for name, values in param_grid.items():
        print(f"   - {name}: {values}")
    print("\n   Fixed parameters:")
    for name, value in fixed_params.items():
        print(f"   - {name}: {value}")
    
    # Test scenario (focused on Byzantine 30%)
    print(f"\n🎯 Test Scenario:")
    print(f"   - 24 total clients (18 benign + 6 malicious)")
    print(f"   - Attack types: Byzantine, Gaussian, Label Flip")
    print(f"   - Gradient dimension: 10,000")
    
    # Run tests
    results = []
    
    print(f"\n🔄 Running {len(combinations) * 3} tests...")
    
    for idx, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        params.update(fixed_params)
        
        # Test on each attack type
        attack_results = {}
        for attack_type in ['byzantine', 'gaussian', 'label_flip']:
            # Generate data
            np.random.seed(42)
            gradients, malicious_ids = generate_test_data(
                n_benign=18,
                n_malicious=6,
                dim=10000,
                attack_type=attack_type
            )
            
            # Test normal phase (round 15)
            detector = ConfigurableLayer1Detector(**params)
            detected = detector.detect(gradients, current_round=15)
            
            # Compute metrics
            metrics = compute_metrics(detected, malicious_ids, len(gradients))
            attack_results[attack_type] = metrics
        
        # Average metrics
        avg_metrics = {
            'detection_rate': np.mean([m['detection_rate'] for m in attack_results.values()]),
            'fpr': np.mean([m['fpr'] for m in attack_results.values()]),
            'f1': np.mean([m['f1'] for m in attack_results.values()])
        }
        
        result = {
            'params': {k: v for k, v in params.items() if k in param_names},
            'by_attack': attack_results,
            'average': avg_metrics
        }
        results.append(result)
        
        if (idx + 1) % 5 == 0:
            print(f"   Progress: {idx + 1}/{len(combinations)} combinations tested")
    
    # Sort by F1 score
    results_sorted = sorted(results, key=lambda x: x['average']['f1'], reverse=True)
    
    # Print results table
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Prepare table data
    table_data = []
    for rank, result in enumerate(results_sorted, 1):
        params = result['params']
        avg = result['average']
        
        row = [
            rank,
            params['dbscan_min_samples'],
            params['dbscan_eps_multiplier'],
            params['mad_k_normal'],
            f"{avg['detection_rate']:.1%}",
            f"{avg['fpr']:.1%}",
            f"{avg['f1']:.3f}"
        ]
        table_data.append(row)
    
    headers = ['Rank', 'minPts', 'eps_mult', 'MAD_k', 'Detection', 'FPR', 'F1']
    print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Detailed breakdown of top 3
    print("\n\n" + "="*70)
    print("TOP 3 CONFIGURATIONS - DETAILED BREAKDOWN")
    print("="*70)
    
    for rank, result in enumerate(results_sorted[:3], 1):
        print(f"\n{'─'*70}")
        print(f"RANK #{rank}")
        print(f"{'─'*70}")
        
        params = result['params']
        print("\nParameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        print("\nResults by Attack Type:")
        for attack_type, metrics in result['by_attack'].items():
            print(f"\n  {attack_type.upper()}:")
            print(f"    Detection Rate: {metrics['detection_rate']:.1%} "
                  f"({metrics['tp']}/{metrics['tp'] + metrics['fn']})")
            print(f"    False Positive Rate: {metrics['fpr']:.1%} "
                  f"({metrics['fp']}/{metrics['fp'] + metrics['tn']})")
            print(f"    Precision: {metrics['precision']:.1%}")
            print(f"    Recall: {metrics['recall']:.1%}")
            print(f"    F1 Score: {metrics['f1']:.3f}")
        
        avg = result['average']
        print(f"\n  AVERAGE:")
        print(f"    Detection Rate: {avg['detection_rate']:.1%}")
        print(f"    False Positive Rate: {avg['fpr']:.1%}")
        print(f"    F1 Score: {avg['f1']:.3f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"layer1_quick_test_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'param_grid': param_grid,
            'fixed_params': fixed_params,
            'results': results_sorted,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n\n💾 Results saved to: {output_file}")
    
    # Recommendation
    best = results_sorted[0]
    print("\n\n" + "="*70)
    print("🎯 RECOMMENDED CONFIGURATION")
    print("="*70)
    print(f"\nAverage F1 Score: {best['average']['f1']:.3f}")
    print(f"Average Detection Rate: {best['average']['detection_rate']:.1%}")
    print(f"Average FPR: {best['average']['fpr']:.1%}")
    print("\nOptimal Parameters:")
    for key, value in best['params'].items():
        print(f"  {key}: {value}")
    print("\nFixed Parameters (keep as is):")
    for key, value in fixed_params.items():
        print(f"  {key}: {value}")
    
    # Comparison with defaults
    print("\n📊 COMPARISON WITH DEFAULT:")
    default_params = {
        'dbscan_min_samples': 3,
        'dbscan_eps_multiplier': 0.5,
        'mad_k_normal': 4.0
    }
    default_result = [r for r in results if r['params'] == default_params][0]
    
    print(f"\nDefault Configuration:")
    print(f"  F1: {default_result['average']['f1']:.3f}")
    print(f"  Detection: {default_result['average']['detection_rate']:.1%}")
    print(f"  FPR: {default_result['average']['fpr']:.1%}")
    
    print(f"\nRecommended Configuration:")
    print(f"  F1: {best['average']['f1']:.3f} "
          f"({(best['average']['f1'] - default_result['average']['f1']) / default_result['average']['f1'] * 100:+.1f}%)")
    print(f"  Detection: {best['average']['detection_rate']:.1%} "
          f"({(best['average']['detection_rate'] - default_result['average']['detection_rate']) * 100:+.1f} pp)")
    print(f"  FPR: {best['average']['fpr']:.1%} "
          f"({(best['average']['fpr'] - default_result['average']['fpr']) * 100:+.1f} pp)")
    
    print("\n" + "="*70)
    print("✅ QUICK TEST COMPLETE!")
    print("="*70)
    
    return results_sorted


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*18 + "LAYER 1 QUICK HYPERPARAMETER TEST" + " "*17 + "║")
    print("║" + " "*25 + "Fast & Focused" + " "*30 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    try:
        results = run_quick_test()
        print("\n🎉 Success! Check the saved JSON file for detailed results.\n")
    except ImportError as e:
        print(f"\n⚠️  Missing dependency: {e}")
        print("Install with: pip install tabulate --break-system-packages")
        print("\nRunning without table formatting...\n")
        # Could still run but without pretty tables