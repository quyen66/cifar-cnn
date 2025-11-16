#!/usr/bin/env python3
"""
Generate Focused Aggressive Parameter Suite
===========================================

Focused search targeting Grade A with ~200 configs.
Based on Rank #1 and #3 patterns with more aggressive tuning.

Target: Detection 75-82%, FPR 5-8%, F1 75-80%

Usage:
    python generate_focused_aggressive.py
"""

import json
import itertools
from datetime import datetime


# ============================================================================
# FOCUSED AGGRESSIVE PARAMETERS
# ============================================================================
# Based on successful patterns from current Grade B configs
# Rank #1: mad_k=3.0, vote=1, dbscan_min=6, eps=1.2, Detection 58%
# Rank #3: mad_k=4.0, vote=3, dbscan_min=3, eps=1.0, Detection 58%

# Strategy: Even MORE aggressive than Grade B winners

LAYER1_CONFIGS = {
    'pca_dims': [20, 25],
    
    # MAD - Even lower than Grade B winners (3.0, 4.0)
    'mad_k_normal': [2.5, 3.0, 3.5],
    'mad_k_warmup': [4.5, 5.0],
    
    # Voting - Very low (Grade B had 1, 3)
    'voting_threshold_normal': [1, 2],
    'voting_threshold_warmup': [3, 4],
    
    # DBSCAN - Balance between Grade B winners
    'dbscan_min_samples': [3, 4, 5],
    'dbscan_eps_multiplier': [1.0, 1.1, 1.2],
    
    'warmup_rounds': [12, 15],
}
# Total: 2 * 3 * 2 * 2 * 2 * 3 * 3 * 2 = 864 configs

# To reduce to ~200, sample strategically
LAYER1_SAMPLED = [
    # Group 1: Very aggressive (mad_k low, vote low)
    {'pca_dims': 20, 'mad_k_normal': 2.5, 'mad_k_warmup': 4.5, 
     'voting_threshold_normal': 1, 'voting_threshold_warmup': 3,
     'dbscan_min_samples': 3, 'dbscan_eps_multiplier': 1.0, 'warmup_rounds': 12},
    
    {'pca_dims': 20, 'mad_k_normal': 2.5, 'mad_k_warmup': 4.5,
     'voting_threshold_normal': 1, 'voting_threshold_warmup': 3,
     'dbscan_min_samples': 3, 'dbscan_eps_multiplier': 1.2, 'warmup_rounds': 12},
    
    {'pca_dims': 20, 'mad_k_normal': 2.5, 'mad_k_warmup': 5.0,
     'voting_threshold_normal': 1, 'voting_threshold_warmup': 4,
     'dbscan_min_samples': 4, 'dbscan_eps_multiplier': 1.1, 'warmup_rounds': 15},
    
    # Group 2: Based on Rank #1 pattern (mad=3.0, vote=1) but tweaked
    {'pca_dims': 20, 'mad_k_normal': 3.0, 'mad_k_warmup': 4.5,
     'voting_threshold_normal': 1, 'voting_threshold_warmup': 3,
     'dbscan_min_samples': 4, 'dbscan_eps_multiplier': 1.2, 'warmup_rounds': 12},
    
    {'pca_dims': 20, 'mad_k_normal': 3.0, 'mad_k_warmup': 5.0,
     'voting_threshold_normal': 1, 'voting_threshold_warmup': 4,
     'dbscan_min_samples': 5, 'dbscan_eps_multiplier': 1.1, 'warmup_rounds': 15},
    
    {'pca_dims': 25, 'mad_k_normal': 3.0, 'mad_k_warmup': 4.5,
     'voting_threshold_normal': 1, 'voting_threshold_warmup': 3,
     'dbscan_min_samples': 3, 'dbscan_eps_multiplier': 1.0, 'warmup_rounds': 12},
    
    # Group 3: Based on Rank #3 pattern (mad=4.0, vote=3) but lower
    {'pca_dims': 20, 'mad_k_normal': 3.5, 'mad_k_warmup': 5.0,
     'voting_threshold_normal': 2, 'voting_threshold_warmup': 3,
     'dbscan_min_samples': 3, 'dbscan_eps_multiplier': 1.0, 'warmup_rounds': 12},
    
    {'pca_dims': 20, 'mad_k_normal': 3.5, 'mad_k_warmup': 5.0,
     'voting_threshold_normal': 2, 'voting_threshold_warmup': 4,
     'dbscan_min_samples': 4, 'dbscan_eps_multiplier': 1.1, 'warmup_rounds': 15},
    
    {'pca_dims': 25, 'mad_k_normal': 3.5, 'mad_k_warmup': 4.5,
     'voting_threshold_normal': 2, 'voting_threshold_warmup': 3,
     'dbscan_min_samples': 5, 'dbscan_eps_multiplier': 1.2, 'warmup_rounds': 12},
]


def generate_full_grid():
    """Generate all 864 combinations."""
    keys = list(LAYER1_CONFIGS.keys())
    values = [LAYER1_CONFIGS[k] for k in keys]
    
    configs = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)
    
    return configs


def generate_strategic_sample(n=216):
    """Generate full systematic grid of ~216 configs."""
    
    configs = []
    
    # Full grid on key params
    for pca in [20, 25]:
        for mad_k in [2.5, 3.0, 3.5]:
            for mad_warmup in [4.5, 5.0, 5.5]:
                for vote in [1, 2]:
                    for vote_warmup in [3, 4]:
                        for dbscan_min in [3, 4, 5]:
                            for eps in [1.0, 1.2]:
                                for warmup in [12, 15]:
                                    config = {
                                        'pca_dims': pca,
                                        'mad_k_normal': mad_k,
                                        'mad_k_warmup': mad_warmup,
                                        'voting_threshold_normal': vote,
                                        'voting_threshold_warmup': vote_warmup,
                                        'dbscan_min_samples': dbscan_min,
                                        'dbscan_eps_multiplier': eps,
                                        'warmup_rounds': warmup,
                                    }
                                    configs.append(config)
    
    return configs


def main():
    print(f"\n{'='*80}")
    print(f"FOCUSED AGGRESSIVE SUITE - TARGET GRADE A")
    print(f"{'='*80}\n")
    
    print(f"🎯 TARGETS:")
    print(f"   Detection Rate: 75-82%")
    print(f"   FPR: 5-8%")
    print(f"   F1 Score: 75-80%\n")
    
    print(f"📊 STRATEGY:")
    print(f"   - Based on Grade B winners (Rank #1, #3)")
    print(f"   - Make even more aggressive")
    print(f"   - Focus on proven parameter ranges")
    print(f"   - Strategic sampling (~200 configs)\n")
    
    # Generate
    print(f"Generating full parameter grid...")
    layer1_configs = generate_strategic_sample()  # Will generate 2*3*3*2*2*3*2*2 = 864
    
    # Standard configs for other layers
    suite = {
        'layer1': layer1_configs,
        'layer2': [{
            'distance_multiplier': 1.5,
            'cosine_threshold': 0.3,
            'warmup_rounds': 15,
        }],
        'noniid': [{
            'h_threshold_normal': 0.6,
            'h_threshold_alert': 0.5,
            'adaptive_multiplier': 1.5,
            'baseline_percentile': 60,
        }],
        'filtering': [{
            'hard_k_threshold': 3,
            'soft_reputation_threshold': 0.4,
            'soft_distance_multiplier': 2.0,
        }],
        'reputation': [{
            'ema_alpha_increase': 0.4,
            'ema_alpha_decrease': 0.2,
            'penalty_flagged': 0.2,
            'penalty_variance': 0.1,
            'reward_clean': 0.1,
            'floor_lift_threshold': 0.4,
        }],
        'mode': [{
            'threshold_normal_to_alert': 0.2,
            'threshold_alert_to_defense': 0.3,
            'hysteresis_normal': 0.1,
            'hysteresis_defense': 0.15,
            'rep_gate_defense': 0.5,
        }],
    }
    
    # Create output
    output = {
        'generated_at': datetime.now().isoformat(),
        'mode': 'focused_aggressive',
        'total_configs': len(layer1_configs),
        'default_config': {
            # Layer 1 - Default aggressive
            'pca_dims': 20,
            'dbscan_min_samples': 3,
            'dbscan_eps_multiplier': 1.0,
            'mad_k_normal': 3.0,
            'mad_k_warmup': 5.0,
            'voting_threshold_normal': 1,
            'voting_threshold_warmup': 3,
            'warmup_rounds': 12,
            
            # Layer 2 - Fixed
            'distance_multiplier': 1.5,
            'cosine_threshold': 0.3,
            'layer2_warmup_rounds': 15,
            
            # Non-IID - Fixed
            'h_threshold_normal': 0.6,
            'h_threshold_alert': 0.5,
            'adaptive_multiplier': 1.5,
            'baseline_percentile': 60,
            
            # Filtering - Fixed
            'hard_k_threshold': 3,
            'soft_reputation_threshold': 0.4,
            'soft_distance_multiplier': 2.0,
            
            # Reputation - Fixed
            'ema_alpha_increase': 0.4,
            'ema_alpha_decrease': 0.2,
            'penalty_flagged': 0.2,
            'penalty_variance': 0.1,
            'reward_clean': 0.1,
            'floor_lift_threshold': 0.4,
            
            # Mode - Fixed
            'threshold_normal_to_alert': 0.2,
            'threshold_alert_to_defense': 0.3,
            'hysteresis_normal': 0.1,
            'hysteresis_defense': 0.15,
            'rep_gate_defense': 0.5
        },
        'suite': suite,
        'targets': {
            'detection_rate': '75-82%',
            'fpr': '5-8%',
            'f1_score': '75-80%',
            'grade': 'A'
        },
        'strategy': {
            'basis': 'Current Grade B configs (Rank #1, #3)',
            'approach': 'More aggressive thresholds',
            'key_changes': [
                'mad_k_normal: 2.5-3.5 (vs Grade B: 3.0-4.0)',
                'voting: 1-2 (vs Grade B: 1-3)',
                'Strategic sampling covering parameter space'
            ]
        }
    }
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"param_suite_focused_aggressive_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Generated: {filename}\n")
    print(f"📦 SUMMARY:")
    print(f"   Total configs: {len(layer1_configs)}")
    print(f"   Focus: Layer 1 optimization only")
    print(f"   Other layers: Fixed defaults\n")
    
    est_hours = len(layer1_configs) * 6 / 60
    print(f"⏱️  ESTIMATED TIME:")
    print(f"   Per config: ~6 minutes")
    print(f"   Total: {est_hours:.1f} hours ({est_hours/24:.1f} days)\n")
    
    print(f"🎯 EXPECTED:")
    print(f"   Grade A: 5-10 configs")
    print(f"   Grade B: 30-50 configs")
    print(f"   Detection: 70-85%")
    print(f"   FPR: 5-10%\n")
    
    print(f"{'='*80}")
    print(f"✅ READY TO TEST")
    print(f"{'='*80}\n")
    
    print(f"Next steps:")
    print(f"  python create_param_configs.py {filename}")
    print(f"  python -u run_param_tests.py param_suite_focused_aggressive_*.json")
    print(f"  python analyze_layer1_results_v2.py param_test_results_intermediate.json\n")


if __name__ == "__main__":
    main()