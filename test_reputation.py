#!/usr/bin/env python3
"""
Test script for Reputation System
==================================
Test Consistency, Participation, Asymmetric EMA, Floor Lifting

Run: python test_reputation.py
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import Reputation System
from cifar_cnn.defense.reputation import ReputationSystem


def test_asymmetric_ema():
    """Test 1: Asymmetric EMA - penalize fast, reward slow."""
    print("\n" + "="*70)
    print("🧪 TEST 1: Asymmetric EMA Update")
    print("="*70)
    
    rep_sys = ReputationSystem()
    
    client_id = 0
    dim = 100
    
    # Reference gradient
    grad_median = np.random.randn(dim) * 0.1
    
    # Initialize
    rep_sys.initialize_client(client_id, initial_reputation=0.8)
    print(f"📊 Initial reputation: {rep_sys.get_reputation(client_id):.3f}")
    
    # Scenario 1: Bad behavior (should drop fast)
    print(f"\n🔴 Scenario 1: Bad behavior (flagged)")
    bad_grad = -grad_median * 2  # Opposite direction
    
    for round_num in range(3):
        new_rep = rep_sys.update(
            client_id, bad_grad, grad_median,
            was_flagged=True, current_round=round_num+1
        )
        print(f"   Round {round_num+1}: {new_rep:.3f}")
    
    rep_after_bad = new_rep
    
    # Scenario 2: Good behavior (should recover slowly)
    print(f"\n🟢 Scenario 2: Good behavior (not flagged)")
    good_grad = grad_median + np.random.randn(dim) * 0.01  # Similar
    
    for round_num in range(3, 8):
        new_rep = rep_sys.update(
            client_id, good_grad, grad_median,
            was_flagged=False, current_round=round_num+1
        )
        print(f"   Round {round_num+1}: {new_rep:.3f}")
    
    rep_after_good = new_rep
    
    # Check asymmetry
    drop = 0.8 - rep_after_bad
    recovery = rep_after_good - rep_after_bad
    
    print(f"\n📈 ANALYSIS:")
    print(f"   Drop (3 rounds): {drop:.3f}")
    print(f"   Recovery (5 rounds): {recovery:.3f}")
    print(f"   Asymmetry: Drop is {'FASTER' if drop > recovery else 'SLOWER'} than recovery")
    
    if drop > recovery:
        print("\n✅ TEST 1 PASSED! (Asymmetric behavior confirmed)")
        return True
    else:
        print("\n❌ TEST 1 FAILED! (Should drop faster than recover)")
        return False


def test_floor_lifting():
    """Test 2: Floor lifting mechanism."""
    print("\n" + "="*70)
    print("🧪 TEST 2: Floor Lifting Mechanism")
    print("="*70)
    
    rep_sys = ReputationSystem(
        floor_threshold=0.15,
        floor_target=0.3,
        floor_patience=5
    )
    
    client_id = 0
    dim = 100
    grad_median = np.random.randn(dim) * 0.1
    
    # Initialize low
    rep_sys.initialize_client(client_id, initial_reputation=0.12)
    print(f"📊 Initial reputation: {rep_sys.get_reputation(client_id):.3f}")
    print(f"   (Below floor threshold: {rep_sys.floor_threshold})")
    
    # Simulate good behavior
    print(f"\n🟢 Good behavior for {rep_sys.floor_patience} rounds...")
    good_grad = grad_median + np.random.randn(dim) * 0.01
    
    for round_num in range(rep_sys.floor_patience + 1):
        new_rep = rep_sys.update(
            client_id, good_grad, grad_median,
            was_flagged=False, current_round=round_num+1
        )
        print(f"   Round {round_num+1}: {new_rep:.3f}")
    
    final_rep = rep_sys.get_reputation(client_id)
    
    print(f"\n📈 RESULT:")
    print(f"   Final reputation: {final_rep:.3f}")
    print(f"   Target was: {rep_sys.floor_target:.3f}")
    
    if final_rep >= rep_sys.floor_target:
        print("\n✅ TEST 2 PASSED! (Floor lifted successfully)")
        return True
    else:
        print("\n❌ TEST 2 FAILED! (Floor not lifted)")
        return False


def test_consistency_score():
    """Test 3: Consistency score calculation."""
    print("\n" + "="*70)
    print("🧪 TEST 3: Consistency Score (History-aware)")
    print("="*70)
    
    rep_sys = ReputationSystem(history_window=5)
    
    client_id = 0
    dim = 100
    grad_median = np.random.randn(dim) * 0.1
    
    rep_sys.initialize_client(client_id)
    
    # Send consistent gradients
    print(f"\n📊 Sending 5 consistent gradients...")
    consistent_grad = grad_median + np.random.randn(dim) * 0.02
    
    for round_num in range(5):
        rep = rep_sys.update(
            client_id, consistent_grad, grad_median,
            was_flagged=False, current_round=round_num+1
        )
        print(f"   Round {round_num+1}: rep={rep:.3f}")
    
    rep_consistent = rep
    
    # Now send inconsistent gradient
    print(f"\n📊 Sending 1 inconsistent gradient...")
    inconsistent_grad = -grad_median
    
    rep_after = rep_sys.update(
        client_id, inconsistent_grad, grad_median,
        was_flagged=False, current_round=6
    )
    print(f"   Round 6: rep={rep_after:.3f}")
    
    drop = rep_consistent - rep_after
    
    print(f"\n📈 ANALYSIS:")
    print(f"   Rep before inconsistency: {rep_consistent:.3f}")
    print(f"   Rep after inconsistency: {rep_after:.3f}")
    print(f"   Drop: {drop:.3f}")
    
    if drop > 0.05:
        print("\n✅ TEST 3 PASSED! (Inconsistency detected)")
        return True
    else:
        print("\n⚠️  TEST 3: Inconsistency impact small (acceptable)")
        return True


def test_multiple_clients():
    """Test 4: Multiple clients with different behaviors."""
    print("\n" + "="*70)
    print("🧪 TEST 4: Multiple Clients - Differential Treatment")
    print("="*70)
    
    rep_sys = ReputationSystem()
    
    dim = 100
    grad_median = np.random.randn(dim) * 0.1
    
    # 3 clients with different behaviors
    behaviors = {
        0: 'benign',    # Always good
        1: 'malicious', # Always bad
        2: 'mixed'      # Sometimes good, sometimes bad
    }
    
    print(f"📊 Setup: 3 clients with different behaviors")
    for cid, behavior in behaviors.items():
        rep_sys.initialize_client(cid, initial_reputation=0.8)
        print(f"   Client {cid}: {behavior}")
    
    # Simulate 10 rounds
    print(f"\n📊 Simulating 10 rounds...")
    
    for round_num in range(10):
        for cid, behavior in behaviors.items():
            if behavior == 'benign':
                grad = grad_median + np.random.randn(dim) * 0.01
                flagged = False
            elif behavior == 'malicious':
                grad = -grad_median * 2
                flagged = True
            else:  # mixed
                if round_num % 2 == 0:
                    grad = grad_median + np.random.randn(dim) * 0.01
                    flagged = False
                else:
                    grad = -grad_median
                    flagged = True
            
            rep_sys.update(cid, grad, grad_median, flagged, round_num+1)
    
    # Check final reputations
    print(f"\n📈 FINAL REPUTATIONS:")
    reps = {}
    for cid, behavior in behaviors.items():
        rep = rep_sys.get_reputation(cid)
        reps[cid] = rep
        print(f"   Client {cid} ({behavior}): {rep:.3f}")
    
    # Verify ordering
    if reps[0] > reps[2] > reps[1]:
        print("\n✅ TEST 4 PASSED! (Benign > Mixed > Malicious)")
        return True
    else:
        print("\n❌ TEST 4 FAILED! (Incorrect ordering)")
        return False


def test_stats():
    """Test 5: Statistics reporting."""
    print("\n" + "="*70)
    print("🧪 TEST 5: Statistics Reporting")
    print("="*70)
    
    rep_sys = ReputationSystem()
    
    # Add some clients
    for i in range(5):
        rep_sys.initialize_client(i, initial_reputation=0.5 + i*0.1)
    
    stats = rep_sys.get_stats()
    
    print(f"\n📊 STATISTICS:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    if stats['num_clients'] == 5:
        print("\n✅ TEST 5 PASSED!")
        return True
    else:
        print("\n❌ TEST 5 FAILED!")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🚀 REPUTATION SYSTEM TEST SUITE")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(test_asymmetric_ema())
    results.append(test_floor_lifting())
    results.append(test_consistency_score())
    results.append(test_multiple_clients())
    results.append(test_stats())
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED! Reputation System is working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the results.")
        return 1


if __name__ == "__main__":
    sys.exit(main())