#!/usr/bin/env python3
"""
Debug: Phase 1 H Weights Suite Validation
==========================================

Verify H weight parameter suite for Phase 1 optimization.

Usage:
    python3 debug_h_weights_suite.py h_weights_suite_*.json
"""

import json
import sys
from pathlib import Path


def convert_to_kebab_case(snake_case: str) -> str:
    """Convert snake_case to kebab-case"""
    return snake_case.replace('_', '-')


def validate_h_weights_suite(suite_file: str):
    """Validate H weights suite structure and parameters."""
    
    print("=" * 80)
    print("PHASE 1: H WEIGHTS SUITE VALIDATION")
    print("=" * 80)
    print()
    
    # Load suite
    with open(suite_file, 'r') as f:
        data = json.load(f)
    
    print(f"📦 Suite file: {suite_file}")
    print()
    
    # Check structure
    print("🔍 STRUCTURE CHECK")
    print("-" * 80)
    
    if 'metadata' not in data:
        print("   ⚠️  No 'metadata' key found")
    else:
        metadata = data['metadata']
        print(f"   ✅ Metadata found:")
        print(f"      Generator: {metadata.get('generator', 'unknown')}")
        print(f"      Phase: {metadata.get('optimization_phase', 'unknown')}")
        print(f"      Total configs: {metadata.get('total_configs', 0)}")
    
    print()
    
    if 'suite' not in data:
        print("   ❌ No 'suite' key found")
        return
    
    suite = data['suite']
    
    # Check for h_weights key (Phase 1)
    if 'h_weights' not in suite:
        print("   ❌ No 'h_weights' key found in suite")
        print("   Expected structure: suite['h_weights'] for Phase 1")
        print()
        print("   Available keys in suite:", list(suite.keys()))
        return
    
    configs = suite['h_weights']
    print(f"   ✅ H weights configs found: {len(configs)}")
    print()
    
    # Validate fixed parameters
    if 'metadata' in data and 'fixed_params' in data['metadata']:
        print("🔒 FIXED PARAMETERS (Phase 1)")
        print("-" * 80)
        fixed = data['metadata']['fixed_params']
        for param, value in fixed.items():
            print(f"   {param:30s} = {value}")
        print()
    
    # Validate H weight configurations
    print("✅ H WEIGHT CONFIGURATIONS")
    print("-" * 80)
    print()
    
    required_keys = ['weight_cv', 'weight_sim', 'weight_cluster']
    invalid_configs = []
    
    for idx, config in enumerate(configs, 1):
        # Check required keys
        missing_keys = [k for k in required_keys if k not in config]
        if missing_keys:
            invalid_configs.append((idx, f"Missing keys: {missing_keys}"))
            continue
        
        # Check weight sum
        cv = config['weight_cv']
        sim = config['weight_sim']
        cluster = config['weight_cluster']
        total = cv + sim + cluster
        
        if abs(total - 1.0) > 1e-6:
            invalid_configs.append((idx, f"Weight sum = {total:.6f} (should be 1.0)"))
    
    if invalid_configs:
        print("❌ INVALID CONFIGURATIONS FOUND:")
        for idx, reason in invalid_configs:
            print(f"   Config #{idx}: {reason}")
        print()
    else:
        print(f"✅ All {len(configs)} configurations valid")
        print()
    
    # Show sample configurations
    print("📋 SAMPLE CONFIGURATIONS")
    print("-" * 80)
    print(f"{'ID':<5} {'CV':<8} {'Sim':<8} {'Cluster':<10} {'Sum':<8} {'Profile':<15}")
    print("-" * 80)
    
    for i in [0, 4, 9, 14]:  # Show first, middle, and last
        if i < len(configs):
            config = configs[i]
            cv = config.get('weight_cv', 0)
            sim = config.get('weight_sim', 0)
            cluster = config.get('weight_cluster', 0)
            total = cv + sim + cluster
            profile = config.get('profile', 'unknown')
            config_id = config.get('config_id', i+1)
            
            print(f"{config_id:<5} {cv:<8.2f} {sim:<8.2f} {cluster:<10.2f} {total:<8.3f} {profile:<15}")
    
    print()
    
    # Show TOML format examples
    print("📝 TOML FORMAT EXAMPLES")
    print("-" * 80)
    print()
    
    if len(configs) > 0:
        example = configs[0]
        print("Example flwr run-config for Config #1:")
        print()
        print("flwr run . --run-config \"\\")
        print("  num-server-rounds=20 \\")
        print("  enable-defense=true \\")
        
        for key in required_keys:
            if key in example:
                kebab_key = convert_to_kebab_case(key)
                value = example[key]
                print(f"  defense.noniid.{kebab_key}={value} \\")
        
        print("  ... other params ...\"")
        print()
    
    # Profile breakdown
    print("📊 PROFILE BREAKDOWN")
    print("-" * 80)
    
    profiles = {}
    for config in configs:
        profile = config.get('profile', 'unknown')
        profiles[profile] = profiles.get(profile, 0) + 1
    
    for profile, count in sorted(profiles.items()):
        print(f"   {profile:20s}: {count:2d} configs")
    
    print()
    
    # Final summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()
    
    total_issues = len(invalid_configs)
    
    if total_issues == 0:
        print("✅ ALL CHECKS PASSED")
        print(f"   {len(configs)} configurations ready for testing")
        print()
        print("Next step: Run tests")
        print(f"  python3 run_h_weight_tests.py {suite_file}")
    else:
        print(f"⚠️  FOUND {total_issues} ISSUES")
        print("   Review issues above before running tests")
    
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 debug_h_weights_suite.py <h_weights_suite.json>")
        sys.exit(1)
    
    suite_file = sys.argv[1]
    
    if not Path(suite_file).exists():
        print(f"❌ File not found: {suite_file}")
        
        # Try to find in current directory
        h_weight_files = list(Path('.').glob('h_weights_suite_*.json'))
        if h_weight_files:
            print()
            print("Found H weight suite files in current directory:")
            for f in h_weight_files:
                print(f"  {f}")
            print()
            print("Try:")
            print(f"  python3 debug_h_weights_suite.py {h_weight_files[0]}")
        
        sys.exit(1)
    
    validate_h_weights_suite(suite_file)


if __name__ == "__main__":
    main()