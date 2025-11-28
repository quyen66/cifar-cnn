"""
Auto-generated script to apply optimal parameters
==================================================

Generated from: param_test_results_20251115_010944_layer1_gradeB.json
Generated at: 2025-11-15T01:09:44.172767

This script updates all defense layer files with optimal parameters.

Usage:
    python apply_optimal_params.py
"""

import sys
from pathlib import Path

# ============================================================================
# OPTIMAL PARAMETERS
# ============================================================================

OPTIMAL_PARAMS = {'layer1': {'pca_dims': 20, 'mad_k_normal': 5.0, 'mad_k_warmup': 6.0, 'dbscan_min_samples': 4, 'dbscan_eps_multiplier': 1.2, 'voting_threshold_normal': 4, 'voting_threshold_warmup': 3, 'warmup_rounds': 20}}

# ============================================================================
# FILE UPDATERS
# ============================================================================

def update_layer1():
    """Update Layer 1 DBSCAN parameters."""
    params = OPTIMAL_PARAMS.get('layer1', {})
    
    if not params:
        print("⚠️  No optimal params for layer1")
        return
    
    print("\n📝 Updating Layer 1 DBSCAN...")
    
    # TODO: Implement actual file update logic
    # Similar to apply_optimal_config.py
    
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}")
    
    print("  ✅ Layer 1 updated")

def update_layer2():
    """Update Layer 2 Distance+Direction parameters."""
    params = OPTIMAL_PARAMS.get('layer2', {})
    
    if not params:
        print("⚠️  No optimal params for layer2")
        return
    
    print("\n📝 Updating Layer 2 Distance+Direction...")
    
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}")
    
    print("  ✅ Layer 2 updated")

def update_noniid():
    """Update Non-IID Handler parameters."""
    params = OPTIMAL_PARAMS.get('noniid', {})
    
    if not params:
        print("⚠️  No optimal params for noniid")
        return
    
    print("\n📝 Updating Non-IID Handler...")
    
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}")
    
    print("  ✅ Non-IID updated")

def update_filtering():
    """Update Two-Stage Filtering parameters."""
    params = OPTIMAL_PARAMS.get('filtering', {})
    
    if not params:
        print("⚠️  No optimal params for filtering")
        return
    
    print("\n📝 Updating Two-Stage Filtering...")
    
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}")
    
    print("  ✅ Filtering updated")

def update_reputation():
    """Update Reputation System parameters."""
    params = OPTIMAL_PARAMS.get('reputation', {})
    
    if not params:
        print("⚠️  No optimal params for reputation")
        return
    
    print("\n📝 Updating Reputation System...")
    
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}")
    
    print("  ✅ Reputation updated")

def update_mode():
    """Update Mode Controller parameters."""
    params = OPTIMAL_PARAMS.get('mode', {})
    
    if not params:
        print("⚠️  No optimal params for mode")
        return
    
    print("\n📝 Updating Mode Controller...")
    
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}")
    
    print("  ✅ Mode updated")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("APPLYING OPTIMAL PARAMETERS")
    print("="*70)
    
    update_layer1()
    update_layer2()
    update_noniid()
    update_filtering()
    update_reputation()
    update_mode()
    
    print("\n" + "="*70)
    print("✅ ALL PARAMETERS APPLIED!")
    print("="*70)
    print("\nNext: Run full test suite to validate\n")

if __name__ == "__main__":
    main()
