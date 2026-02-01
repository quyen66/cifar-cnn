#!/usr/bin/env python3
"""
Hybrid H-Score Dense Grid Generator
====================================

Generate dense grid of H weight combinations for HYBRID H-SCORE:
  H = weight_h_grad × H_grad + weight_h_loss × H_loss + weight_h_acc × H_acc

Options:
    --grid-step: Grid spacing (default: 0.1)
    --min-value: Minimum value for any weight (default: 0.1)
    
Usage:
    python3 param_generator.py --grid-step 0.1
    python3 param_generator.py --grid-step 0.05  # Slower, more configs
"""

import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple


class HybridHWeightGenerator:
    """Generate dense grid of hybrid H weight combinations."""
    
    def __init__(self, grid_step: float = 0.1, min_value: float = 0.1):
        """
        Initialize generator.
        
        Args:
            grid_step: Step size for grid (0.1 or 0.05)
            min_value: Minimum allowed value for any weight (NO ZEROS!)
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
        Generate all valid (grad, loss, acc) combinations.
        
        Constraints:
        1. grad + loss + acc = 1.0
        2. All values >= min_value (NO ZEROS!)
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
        
        # Try all grad, loss combinations
        for grad in grid_points:
            for loss in grid_points:
                # Calculate acc
                acc = round(1.0 - grad - loss, 2)
                
                # Check constraints
                if acc >= self.min_value and acc <= max_value:
                    # Verify sum = 1.0 (handle floating point)
                    total = grad + loss + acc
                    if abs(total - 1.0) < 1e-6:
                        combinations.append((grad, loss, acc))
        
        return combinations
    
    def _get_profile_name(self, grad: float, loss: float, acc: float) -> str:
        """Get descriptive profile name based on weight distribution."""
        
        # Check if default recommended
        if abs(grad - 0.2) < 0.01 and abs(loss - 0.4) < 0.01 and abs(acc - 0.4) < 0.01:
            return "golden_ratio"
        
        # Find dominant component
        max_weight = max(grad, loss, acc)
        
        # Strong emphasis cases (>= 0.5)
        if max_weight == grad and grad >= 0.5:
            return "grad_dominant"
        elif max_weight == loss and loss >= 0.5:
            return "loss_dominant"
        elif max_weight == acc and acc >= 0.5:
            return "acc_dominant"
        
        # Moderate emphasis cases (>= 0.4)
        elif max_weight == grad and grad >= 0.4:
            return "grad_emphasis"
        elif max_weight == loss and loss >= 0.4:
            return "loss_emphasis"
        elif max_weight == acc and acc >= 0.4:
            return "acc_emphasis"
        
        # Balanced case
        elif abs(grad - loss) < 0.15 and abs(grad - acc) < 0.15:
            return "balanced"
        
        # Mixed cases
        else:
            return "mixed"
    
    def generate_suite(self) -> Dict:
        """Generate complete parameter suite."""
        
        configs = []
        
        for idx, (grad, loss, acc) in enumerate(self.combinations, 1):
            config = {
                'config_id': idx,
                'weight_h_grad': grad,
                'weight_h_loss': loss,
                'weight_h_acc': acc,
                'profile': self._get_profile_name(grad, loss, acc)
            }
            configs.append(config)
        
        suite = {
            'metadata': {
                'generator': 'param_generator.py',
                'version': '3.0.0-hybrid',
                'timestamp': datetime.now().isoformat(),
                'total_configs': len(configs),
                'optimization_phase': 'Phase 1 - Hybrid H Weights Dense Grid',
                'h_score_formula': 'H = w_grad*H_grad + w_loss*H_loss + w_acc*H_acc',
                'grid_step': self.grid_step,
                'min_value': self.min_value,
                'fixed_params': {
                    'adjustment_factor': 0.1,
                    'theta_adj_clip_min': 0.3,
                    'theta_adj_clip_max': 0.8,
                    'baseline_window_size': 10,
                    'delta_norm_weight': 0.5,
                    'delta_direction_weight': 0.5,
                    'grad_ema_decay': 0.3
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
            filename = f"hybrid_h_weights_grid{step_str}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(suite, f, indent=2)
        
        return filename


def main():
    parser = argparse.ArgumentParser(
        description='Generate dense grid of hybrid H-score weights'
    )
    parser.add_argument(
        '--grid-step', 
        type=float, 
        default=0.1,
        help='Grid step size (default: 0.1)'
    )
    parser.add_argument(
        '--min-value', 
        type=float, 
        default=0.1,
        help='Minimum value for any weight (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.grid_step <= 0 or args.grid_step >= 1.0:
        print(f"❌ Invalid grid-step: {args.grid_step} (must be 0 < step < 1.0)")
        return
    
    if args.min_value < 0 or args.min_value >= 0.5:
        print(f"❌ Invalid min-value: {args.min_value} (must be 0 <= min < 0.5)")
        return
    
    print("=" * 80)
    print("HYBRID H-SCORE DENSE GRID GENERATOR")
    print("Formula: H = w_grad×H_grad + w_loss×H_loss + w_acc×H_acc")
    print("=" * 80)
    print()
    
    # Generate suite
    generator = HybridHWeightGenerator(
        grid_step=args.grid_step,
        min_value=args.min_value
    )
    suite = generator.generate_suite()
    
    # Statistics
    print(f"📊 Suite Statistics:")
    print(f"   Total configurations: {suite['metadata']['total_configs']}")
    print(f"   Grid step: {args.grid_step}")
    print(f"   Min value: {args.min_value}")
    print(f"   Constraint: All weights >= {args.min_value} (NO ZEROS)")
    print()
    
    # Estimate time
    configs = len(suite['suite']['h_weights'])
    time_per_config = 40  # minutes (conservative, 30 rounds)
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
        pct = count / configs * 100
        print(f"   {profile:20s}: {count:3d} configs ({pct:5.1f}%)")
    print()
    
    # Sample configs
    print(f"📋 Sample Configurations:")
    print(f"   {'ID':<5} {'Profile':<20} {'Grad':<6} {'Loss':<6} {'Acc':<6} {'Sum':<6}")
    print(f"   {'-'*5} {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    
    samples = [0, configs//4, configs//2, 3*configs//4, configs-1]
    for i in samples:
        if i < len(suite['suite']['h_weights']):
            cfg = suite['suite']['h_weights'][i]
            total = cfg['weight_h_grad'] + cfg['weight_h_loss'] + cfg['weight_h_acc']
            print(f"   {cfg['config_id']:<5d} {cfg['profile']:<20s} "
                  f"{cfg['weight_h_grad']:<6.2f} "
                  f"{cfg['weight_h_loss']:<6.2f} "
                  f"{cfg['weight_h_acc']:<6.2f} "
                  f"{total:<6.2f}")
    print()
    
    # Check for recommended baseline
    has_golden = any(
        abs(c['weight_h_grad'] - 0.2) < 0.01 and 
        abs(c['weight_h_loss'] - 0.4) < 0.01 and 
        abs(c['weight_h_acc'] - 0.4) < 0.01
        for c in suite['suite']['h_weights']
    )
    
    if has_golden:
        print(f"✅ Golden Ratio baseline (0.2, 0.4, 0.4) included!")
    else:
        print(f"⚠️  Golden Ratio baseline (0.2, 0.4, 0.4) NOT in grid")
        print(f"   (Grid step {args.grid_step} may skip it)")
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
    print("1. Run tests:")
    print(f"   python3 run_param_tests.py {filename}")
    print()
    print("2. With resume support:")
    print(f"   python3 run_param_tests.py {filename} --resume")
    print()
    print(f"⚠️  This will take ~{total_hours:.0f} hours!")
    print()
    print("💡 Recommendations:")
    print("  - Run overnight/weekend")
    print("  - Use --grid-step 0.1 for faster testing (recommended)")
    print("  - Use --grid-step 0.05 for thorough search (~4x configs)")
    print("  - Resume if interrupted with --resume flag")
    print()
    print("📊 Output: benchmark_results.csv with all metrics")
    print()


if __name__ == "__main__":
    main()