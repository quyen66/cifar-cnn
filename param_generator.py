#!/usr/bin/env python3
"""
H Weights Dense Grid Generator
===============================

Generate dense grid of H weight combinations for thorough optimization.

Options:
    --grid-step: Grid spacing (default: 0.05 for ~80 configs)
    --min-value: Minimum value for any weight (default: 0.1)
    
Usage:
    python3 generate_h_weights_dense.py --grid-step 0.05
    python3 generate_h_weights_dense.py --grid-step 0.1    # Faster, ~30 configs
"""

import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple


class DenseHWeightGenerator:
    """Generate dense grid of H weight combinations."""
    
    def __init__(self, grid_step: float = 0.05, min_value: float = 0.1):
        """
        Initialize generator.
        
        Args:
            grid_step: Step size for grid (0.05 or 0.1)
            min_value: Minimum allowed value for any weight
        """
        self.grid_step = grid_step
        self.min_value = min_value
        
        # Generate all valid combinations
        self.combinations = self._generate_combinations()
        
        print(f"✅ Generated {len(self.combinations)} valid combinations")
        print(f"   Grid step: {grid_step}")
        print(f"   Min value: {min_value}")
        print()
    
    def _generate_combinations(self) -> List[Tuple[float, float, float]]:
        """
        Generate all valid (cv, sim, cluster) combinations.
        
        Constraints:
        1. cv + sim + cluster = 1.0
        2. All values >= min_value
        3. All values are multiples of grid_step
        """
        combinations = []
        
        # Generate grid points
        max_value = 1.0 - 2 * self.min_value
        grid_points = []
        value = self.min_value
        while value <= max_value:
            grid_points.append(round(value, 2))
            value += self.grid_step
        
        # Try all cv, sim combinations
        for cv in grid_points:
            for sim in grid_points:
                # Calculate cluster
                cluster = round(1.0 - cv - sim, 2)
                
                # Check constraints
                if cluster >= self.min_value and cluster <= max_value:
                    # Verify sum = 1.0 (handle floating point)
                    total = cv + sim + cluster
                    if abs(total - 1.0) < 1e-6:
                        combinations.append((cv, sim, cluster))
        
        return combinations
    
    def _get_profile_name(self, cv: float, sim: float, cluster: float) -> str:
        """Get descriptive profile name."""
        
        # Check if PDF default
        if abs(cv - 0.4) < 0.01 and abs(sim - 0.4) < 0.01 and abs(cluster - 0.2) < 0.01:
            return "pdf_default"
        
        # Find dominant component
        max_weight = max(cv, sim, cluster)
        
        if max_weight == cv and cv >= 0.45:
            return "cv_emphasis"
        elif max_weight == sim and sim >= 0.45:
            return "sim_emphasis"
        elif max_weight == cluster and cluster >= 0.35:
            return "cluster_emphasis"
        elif abs(cv - sim) < 0.1 and abs(cv - cluster) < 0.1:
            return "balanced"
        else:
            return "mixed"
    
    def generate_suite(self) -> Dict:
        """Generate complete parameter suite."""
        
        configs = []
        
        for idx, (cv, sim, cluster) in enumerate(self.combinations, 1):
            config = {
                'config_id': idx,
                'weight_cv': cv,
                'weight_sim': sim,
                'weight_cluster': cluster,
                'profile': self._get_profile_name(cv, sim, cluster)
            }
            configs.append(config)
        
        suite = {
            'metadata': {
                'generator': 'generate_h_weights_dense.py',
                'version': '2.0.0',
                'timestamp': datetime.now().isoformat(),
                'total_configs': len(configs),
                'optimization_phase': 'Phase 1 - H Weights Dense Grid',
                'grid_step': self.grid_step,
                'min_value': self.min_value,
                'fixed_params': {
                    'h_threshold_normal': 0.6,
                    'h_threshold_alert': 0.5,
                    'adjustment_factor': 0.4,
                    'baseline_percentile': 60,
                    'baseline_window_size': 10,
                    'delta_norm_weight': 0.5,
                    'delta_direction_weight': 0.5
                }
            },
            'suite': {
                'h_weights': configs
            }
        }
        
        return suite
    
    def save_suite(self, suite: Dict, filename: str = None) -> str:
        """Save suite to JSON file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            step_str = str(self.grid_step).replace('.', '')
            filename = f"h_weights_dense_{step_str}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(suite, f, indent=2)
        
        return filename


def main():
    parser = argparse.ArgumentParser(description='Generate dense H weights grid')
    parser.add_argument('--grid-step', type=float, default=0.05,
                       help='Grid step size (default: 0.05)')
    parser.add_argument('--min-value', type=float, default=0.1,
                       help='Minimum value for any weight (default: 0.1)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.grid_step <= 0 or args.grid_step >= 1.0:
        print(f"❌ Invalid grid-step: {args.grid_step} (must be 0 < step < 1.0)")
        return
    
    if args.min_value < 0 or args.min_value >= 0.5:
        print(f"❌ Invalid min-value: {args.min_value} (must be 0 <= min < 0.5)")
        return
    
    print("=" * 80)
    print("H WEIGHTS DENSE GRID GENERATOR")
    print("=" * 80)
    print()
    
    # Generate suite
    generator = DenseHWeightGenerator(
        grid_step=args.grid_step,
        min_value=args.min_value
    )
    suite = generator.generate_suite()
    
    # Statistics
    print(f"📊 Suite Statistics:")
    print(f"   Total configurations: {suite['metadata']['total_configs']}")
    print(f"   Grid step: {args.grid_step}")
    print(f"   Min value: {args.min_value}")
    print()
    
    # Estimate time
    configs = len(suite['suite']['h_weights'])
    time_per_config = 40  # minutes (conservative)
    total_hours = (configs * time_per_config) / 60
    
    print(f"⏱️  Estimated Time:")
    print(f"   Per config: ~{time_per_config} minutes")
    print(f"   Total: ~{total_hours:.1f} hours ({total_hours/24:.1f} days)")
    print()
    
    # Profile breakdown
    profiles = {}
    for config in suite['suite']['h_weights']:
        profile = config['profile']
        profiles[profile] = profiles.get(profile, 0) + 1
    
    print(f"📈 Profile Breakdown:")
    for profile, count in sorted(profiles.items(), key=lambda x: -x[1]):
        print(f"   {profile:20s}: {count:3d} configs ({count/configs*100:.1f}%)")
    print()
    
    # Sample configs
    print(f"📋 Sample Configurations:")
    samples = [0, configs//4, configs//2, 3*configs//4, configs-1]
    for i in samples:
        if i < len(suite['suite']['h_weights']):
            config = suite['suite']['h_weights'][i]
            print(f"   Config #{config['config_id']:3d} ({config['profile']:15s}): "
                  f"CV={config['weight_cv']:.2f}, "
                  f"sim={config['weight_sim']:.2f}, "
                  f"cluster={config['weight_cluster']:.2f}")
    print()
    
    # Save
    filename = generator.save_suite(suite)
    print(f"💾 Suite saved to: {filename}")
    print()
    
    # Usage instructions
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("Run tests:")
    print(f"  python3 run_h_weight_tests_chdir.py {filename}")
    print()
    print(f"⚠️  This will take ~{total_hours:.0f} hours!")
    print()
    print("Consider:")
    print("  - Run overnight/weekend")
    print("  - Use --grid-step 0.1 for faster testing (~30 configs, ~20 hours)")
    print("  - Resume if interrupted with --resume flag")
    print()


if __name__ == "__main__":
    main()