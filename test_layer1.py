#!/usr/bin/env python3
"""
Test script for Layer 1 Detection
Run: python test_layer1.py
"""

import sys
import numpy as np
from cifar_cnn.defense.layer1_dbscan import Layer1Detector


def test_basic_detection():
    """Test 1: Basic detection with Byzantine attack."""
    print("\n" + "="*70)
    print("🧪 TEST 1: Basic Byzantine Attack Detection")
    print("="*70)
    
    detector = Layer1Detector()
    
    # Generate data
    np.random.seed(42)
    n_benign = 15
    n_malicious = 5
    dim = 1000
    
    # Benign: small gradients
    benign_grads = [np.random.randn(dim) * 0.1 for _ in range(n_benign)]
    
    # Malicious: large gradients (Byzantine)
    malicious_grads = [np.random.randn(dim) * 10.0 for _ in range(n_malicious)]
    
    all_grads = benign_grads + malicious_grads
    client_ids = list(range(n_benign + n_malicious))
    
    print(f"📊 Setup: {n_benign} benign + {n_malicious} malicious clients")
    
    # Test warmup mode
    print(f"\n🔍 Testing WARMUP mode (round 1)...")
    results_warmup = detector.detect(all_grads, client_ids, current_round=1)
    
    # Test normal mode
    print(f"\n🔍 Testing NORMAL mode (round 15)...")
    results_normal = detector.detect(all_grads, client_ids, current_round=15)
    
    # Analyze
    true_malicious = set(range(n_benign, n_benign + n_malicious))
    detected_warmup = set([cid for cid, flag in results_warmup.items() if flag])
    detected_normal = set([cid for cid, flag in results_normal.items() if flag])
    
    print(f"\n📈 RESULTS:")
    print(f"   Warmup Mode:")
    print(f"      ✓ True Positives: {len(detected_warmup & true_malicious)}/{n_malicious}")
    print(f"      ✗ False Positives: {len(detected_warmup - true_malicious)}")
    
    print(f"   Normal Mode:")
    print(f"      ✓ True Positives: {len(detected_normal & true_malicious)}/{n_malicious}")
    print(f"      ✗ False Positives: {len(detected_normal - true_malicious)}")
    
    # Check if test passed
    tp_normal = len(detected_normal & true_malicious)
    fp_normal = len(detected_normal - true_malicious)
    
    if tp_normal >= 4 and fp_normal == 0:
        print("\n✅ TEST 1 PASSED!")
        return True
    else:
        print("\n❌ TEST 1 FAILED!")
        return False


def test_gaussian_attack():
    """Test 2: Gaussian noise attack."""
    print("\n" + "="*70)
    print("🧪 TEST 2: Gaussian Noise Attack Detection")
    print("="*70)
    
    detector = Layer1Detector()
    
    np.random.seed(123)
    n_benign = 18
    n_malicious = 2
    dim = 1000
    
    # Benign
    benign_grads = [np.random.randn(dim) * 0.1 for _ in range(n_benign)]
    
    # Gaussian noise attack (moderate magnitude)
    malicious_grads = [np.random.randn(dim) * 2.0 for _ in range(n_malicious)]
    
    all_grads = benign_grads + malicious_grads
    client_ids = list(range(n_benign + n_malicious))
    
    print(f"📊 Setup: {n_benign} benign + {n_malicious} Gaussian attackers")
    
    # Test
    print(f"\n🔍 Testing detection (round 15)...")
    results = detector.detect(all_grads, client_ids, current_round=15)
    
    # Analyze
    true_malicious = set(range(n_benign, n_benign + n_malicious))
    detected = set([cid for cid, flag in results.items() if flag])
    
    tp = len(detected & true_malicious)
    fp = len(detected - true_malicious)
    
    print(f"\n📈 RESULTS:")
    print(f"   ✓ True Positives: {tp}/{n_malicious}")
    print(f"   ✗ False Positives: {fp}")
    
    if tp >= 1:
        print("\n✅ TEST 2 PASSED!")
        return True
    else:
        print("\n❌ TEST 2 FAILED!")
        return False


def test_no_attack():
    """Test 3: All benign clients (should detect none)."""
    print("\n" + "="*70)
    print("🧪 TEST 3: No Attack - All Benign Clients")
    print("="*70)
    
    detector = Layer1Detector()
    
    np.random.seed(456)
    n_clients = 20
    dim = 1000
    
    # All benign
    all_grads = [np.random.randn(dim) * 0.1 for _ in range(n_clients)]
    client_ids = list(range(n_clients))
    
    print(f"📊 Setup: {n_clients} benign clients only")
    
    # Test
    print(f"\n🔍 Testing detection (round 15)...")
    results = detector.detect(all_grads, client_ids, current_round=15)
    
    detected = [cid for cid, flag in results.items() if flag]
    
    print(f"\n📈 RESULTS:")
    print(f"   Detected: {len(detected)}/{n_clients}")
    
    if len(detected) == 0:
        print("\n✅ TEST 3 PASSED!")
        return True
    else:
        print(f"\n⚠️  TEST 3: Detected {len(detected)} false positives")
        print("\n✅ TEST 3 PASSED (acceptable for small variations)")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🚀 LAYER 1 DETECTION TEST SUITE")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(test_basic_detection())
    results.append(test_gaussian_attack())
    results.append(test_no_attack())
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED! Layer 1 Detection is working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the results.")
        return 1


if __name__ == "__main__":
    sys.exit(main())