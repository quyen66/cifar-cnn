#!/usr/bin/env python3
"""
Generate Relaxed Parameter Suite
=================================

Creates new param suite with MORE LENIENT ranges based on Phase 1 analysis.

Usage:
    python generate_relaxed_suite.py --mode sampled  # ~300 configs
    python generate_relaxed_suite.py --mode full     # ~2500 configs
"""

import json
import itertools
import random
from datetime import datetime
from pathlib import Path
import argparse


# ============================================================================
# RELAXED PARAMETER RANGES
# ============================================================================

LAYER1_PARAMS = {
    'pca_dims': [20],
    'dbscan_min_samples': [3, 4, 5],
    'dbscan_eps_multiplier': [0.6, 0.8, 1.0],
    'mad_k_normal': [5.0, 6.0, 7.0],
    'mad_k_warmup': [8.0, 9.0, 10.0],
    'voting_threshold_normal': [2, 3, 4],
    'voting_threshold_warmup': [3, 4, 5],
    'warmup_rounds': [15, 20],
}

LAYER2_PARAMS = {
    'distance_multiplier': [2.5, 3.0, 3.5],
    'cosine_threshold': [0.1, 0.2, 0.3],
    'warmup_rounds': [20, 25, 30],
}

NONIID_PARAMS = {
    'h_threshold_normal': [0.6, 0.7, 0.8],
    'h_threshold_alert': [0.5, 0.6, 0.7],
    'adaptive_multiplier': [1.2, 1.3, 1.4],
    'baseline_percentile': [75, 80, 85],
}

FILTERING_PARAMS = {
    'hard_k_threshold': [4, 5, 6],
    'soft_reputation_threshold': [0.2, 0.3, 0.4],
    'soft_distance_multiplier': [3.0, 3.5, 4.0],
}

REPUTATION_PARAMS = {
    'ema_alpha_increase': [0.2, 0.3, 0.4],
    'ema_alpha_decrease': [0.05, 0.1, 0.15],
    'penalty_flagged': [0.05, 0.1, 0.15],
    'penalty_variance': [0.02, 0.05, 0.08],
    'reward_clean': [0.15, 0.2, 0.25],
    'floor_lift_threshold': [0.2, 0.3, 0.4],
}

MODE_PARAMS = {
    'threshold_normal_to_alert': [0.25, 0.3, 0.35],
    'threshold_alert_to_defense': [0.35, 0.4, 0.45],
    'hysteresis_normal': [0.15, 0.2, 0.25],
    'hysteresis_defense': [0.2, 0.25, 0.3],
    'rep_gate_defense': [0.3, 0.4, 0.5],
}


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_all_combinations(params):
    """Generate all combinations for a param dict."""
    keys = list(params.keys())
    values = [params[k] for k in keys]
    
    configs = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)
    
    return configs


def sample_combinations(params, n_samples):
    """Sample n combinations from param space."""
    keys = list(params.keys())
    values = [params[k] for k in keys]
    
    # Generate all possible
    all_configs = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        all_configs.append(config)
    
    # Sample
    if n_samples >= len(all_configs):
        return all_configs
    
    random.seed(42)  # Reproducible
    return random.sample(all_configs, n_samples)


def generate_suite(mode='sampled'):
    """
    Generate param suite.
    
    Args:
        mode: 'sampled' (~300 configs) or 'full' (~2500 configs)
    """
    
    if mode == 'sampled':
        # Sample mode - ~300 configs total
        layer1 = sample_combinations(LAYER1_PARAMS, 80)
        layer2 = sample_combinations(LAYER2_PARAMS, 20)
        noniid = sample_combinations(NONIID_PARAMS, 40)
        filtering = sample_combinations(FILTERING_PARAMS, 20)
        reputation = sample_combinations(REPUTATION_PARAMS, 80)
        mode_configs = sample_combinations(MODE_PARAMS, 60)
        
    elif mode == 'full':
        # Full mode - all combinations (~2500 configs)
        layer1 = generate_all_combinations(LAYER1_PARAMS)
        layer2 = generate_all_combinations(LAYER2_PARAMS)
        noniid = generate_all_combinations(NONIID_PARAMS)
        filtering = generate_all_combinations(FILTERING_PARAMS)
        reputation = generate_all_combinations(REPUTATION_PARAMS)
        mode_configs = generate_all_combinations(MODE_PARAMS)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    suite = {
        'layer1': layer1,
        'layer2': layer2,
        'noniid': noniid,
        'filtering': filtering,
        'reputation': reputation,
        'mode': mode_configs
    }
    
    # Calculate default config (middle values)
    default_config = {
        'pca_dims': 20,
        'dbscan_min_samples': 4,
        'dbscan_eps_multiplier': 0.8,
        'mad_k_normal': 6.0,
        'mad_k_warmup': 9.0,
        'voting_threshold_normal': 3,
        'voting_threshold_warmup': 4,
        'warmup_rounds': 15,
        'distance_multiplier': 3.0,
        'cosine_threshold': 0.2,
        'layer2_warmup_rounds': 25,
        'h_threshold_normal': 0.7,
        'h_threshold_alert': 0.6,
        'adaptive_multiplier': 1.3,
        'baseline_percentile': 80,
        'hard_k_threshold': 5,
        'soft_reputation_threshold': 0.3,
        'soft_distance_multiplier': 3.5,
        'ema_alpha_increase': 0.3,
        'ema_alpha_decrease': 0.1,
        'penalty_flagged': 0.1,
        'penalty_variance': 0.05,
        'reward_clean': 0.2,
        'floor_lift_threshold': 0.3,
        'threshold_normal_to_alert': 0.3,
        'threshold_alert_to_defense': 0.4,
        'hysteresis_normal': 0.2,
        'hysteresis_defense': 0.25,
        'rep_gate_defense': 0.4
    }
    
    # Create output
    total_configs = sum(len(configs) for configs in suite.values())
    
    output = {
        'generated_at': datetime.now().isoformat(),
        'mode': mode,
        'total_configs': total_configs,
        'default_config': default_config,
        'suite': suite,
        'notes': {
            'philosophy': 'Innocent until proven guilty - MUCH more lenient than original',
            'changes': 'Based on Phase 1 results showing defense was too aggressive',
            'expected': 'Lower false positives, higher accuracy, may miss subtle attacks'
        }
    }
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Generate relaxed param suite')
    parser.add_argument('--mode', choices=['sampled', 'full'], default='sampled',
                       help='Generation mode: sampled (~300) or full (~2500)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"GENERATING RELAXED PARAMETER SUITE - {args.mode.upper()} MODE")
    print(f"{'='*80}\n")
    
    # Generate
    suite = generate_suite(args.mode)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"param_suite_relaxed_{args.mode}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(suite, f, indent=2)
    
    # Print summary
    print(f"✅ Generated suite saved to: {filename}\n")
    print(f"📊 SUMMARY:")
    print(f"  Mode: {args.mode}")
    print(f"  Total configs: {suite['total_configs']}")
    print(f"\n  Breakdown:")
    for layer, configs in suite['suite'].items():
        print(f"    {layer}: {len(configs)} configs")
    
    print(f"\n⏱️  ESTIMATED TIME:")
    est_hours = suite['total_configs'] * 10.6 / 60
    print(f"    Total: {est_hours:.1f} hours ({est_hours/24:.1f} days)")
    
    print(f"\n💡 PHILOSOPHY:")
    print(f"    {suite['notes']['philosophy']}")
    print(f"    {suite['notes']['expected']}")
    
    print(f"\n{'='*80}")
    print(f"✅ DONE - Ready to run!")
    print(f"{'='*80}\n")
    
    print(f"Next steps:")
    print(f"  1. Review: cat {filename} | jq '.notes'")
    print(f"  2. Run: python -u run_param_tests.py {filename}")
    print(f"  3. Monitor: tail -f test_logs/test_*.log")
    print()


if __name__ == "__main__":
    main()