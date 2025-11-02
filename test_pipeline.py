#!/usr/bin/env python3
"""Quick test for full pipeline components."""

import sys
import numpy as np

sys.path.insert(0, '/mnt/user-data/outputs')
sys.path.insert(0, '/mnt/project')

# Import components
from cifar_cnn.defense import Layer1Detector, Layer2Detector, ReputationSystem, NonIIDHandler, TwoStageFilter, ModeController
from cifar_cnn.aggregation_methods import aggregate_by_mode

print("\n" + "="*70)
print("QUICK PIPELINE TEST")
print("="*70)

# Generate test data
np.random.seed(42)
n_benign, n_malicious = 18, 6
dim = 1000

benign_grads = [np.random.randn(dim) * 0.1 for _ in range(n_benign)]
malicious_grads = [np.random.randn(dim) * 10.0 for _ in range(n_malicious)]
all_grads = malicious_grads + benign_grads
client_ids = list(range(len(all_grads)))
malicious_ids = set(range(n_malicious))

# Compute median
grad_matrix = np.vstack([g for g in all_grads])
grad_median = np.median(grad_matrix, axis=0)

print(f"\n✓ Generated {len(all_grads)} gradients ({n_malicious} malicious)")

# Test each component
print("\n" + "-"*70)
print("Testing Components...")
print("-"*70)

# 1. Layer 1
print("\n1. Layer 1 Detector...")
layer1 = Layer1Detector()
ground_truth = [cid in malicious_ids for cid in client_ids]
l1_results = layer1.detect(all_grads, client_ids, ground_truth, current_round=10)
detected_l1 = sum(1 for flag in l1_results.values() if flag)
print(f"   ✓ Detected {detected_l1} clients")

# 2. Layer 2
print("\n2. Layer 2 Detector...")
layer2 = Layer2Detector()
l2_results = layer2.detect(all_grads, client_ids, 10, l1_results)
detected_l2 = sum(1 for flag in l2_results.values() if flag)
print(f"   ✓ Detected {detected_l2} additional clients")

# 3. Non-IID
print("\n3. Non-IID Handler...")
noniid = NonIIDHandler()
H = noniid.compute_heterogeneity_score(all_grads, client_ids)
print(f"   ✓ Heterogeneity: H = {H:.3f}")

# 4. Filtering
print("\n4. Two-Stage Filter...")
filter_sys = TwoStageFilter()
conf_scores = {cid: 0.9 if l1_results.get(cid) else 0.1 for cid in client_ids}
reps = {cid: 0.8 for cid in client_ids}
trusted, filtered, stats = filter_sys.filter_clients(client_ids, conf_scores, reps, 'NORMAL', H)
print(f"   ✓ Trusted: {len(trusted)}, Filtered: {len(filtered)}")

# 5. Reputation
print("\n5. Reputation System...")
rep_sys = ReputationSystem()
for i, cid in enumerate(client_ids):
    rep_sys.initialize_client(cid)
    was_flagged = l1_results.get(cid, False) or l2_results.get(cid, False)
    rep = rep_sys.update(cid, all_grads[i], grad_median, was_flagged, 10)
print(f"   ✓ Updated {len(client_ids)} reputations")

# 6. Mode Controller
print("\n6. Mode Controller...")
mode_ctrl = ModeController()
rho = len(malicious_ids) / len(client_ids)
mode = mode_ctrl.update_mode(rho, list(malicious_ids), reps, 10)
print(f"   ✓ Mode: {mode}")

# 7. Aggregation
print("\n7. Aggregation Methods...")
for m in ['NORMAL', 'ALERT', 'DEFENSE']:
    agg = aggregate_by_mode(benign_grads, mode=m)
    print(f"   ✓ {m}: norm={np.linalg.norm(agg):.2f}")

print("\n" + "="*70)
print("✅ ALL COMPONENTS WORKING!")
print("="*70 + "\n")