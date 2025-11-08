"""
Defense Parameter Generator
============================

Tự động generate các combinations của parameters cho TẤT CẢ defense layers:
- Layer 1: DBSCAN Detection
- Layer 2: Distance + Direction Detection  
- Non-IID Handler
- Two-Stage Filtering
- Reputation System
- Mode Controller

Output: JSON file với tất cả param combinations để test
"""

import json
import itertools
from typing import Dict, List
from datetime import datetime
from pathlib import Path


class DefenseParamGenerator:
    """Generate parameter combinations cho defense system."""
    
    def __init__(self):
        """Initialize param grids cho tất cả layers."""
        
        # ===== LAYER 1: DBSCAN Detection =====
        self.layer1_grid = {
            'pca_dims': [20],  # Fixed
            'dbscan_min_samples': [2, 3, 4],
            'dbscan_eps_multiplier': [0.4, 0.5, 0.6],
            'mad_k_normal': [3.0, 4.0, 5.0],
            'mad_k_warmup': [5.0, 6.0, 7.0],
            'voting_threshold_normal': [1, 2],
            'voting_threshold_warmup': [2, 3],
            'warmup_rounds': [10, 15]
        }
        
        # ===== LAYER 2: Distance + Direction =====
        self.layer2_grid = {
            'distance_multiplier': [1.2, 1.5, 1.8, 2.0],
            'cosine_threshold': [0.2, 0.3, 0.4],
            'warmup_rounds': [10, 15, 20]
        }
        
        # ===== NON-IID HANDLER =====
        self.noniid_grid = {
            'h_threshold_normal': [0.5, 0.6, 0.7],
            'h_threshold_alert': [0.4, 0.5, 0.6],
            'adaptive_multiplier': [1.2, 1.5, 1.8],
            'baseline_percentile': [50, 60, 70]
        }
        
        # ===== TWO-STAGE FILTERING =====
        self.filtering_grid = {
            'hard_k_threshold': [2, 3, 4],
            'soft_reputation_threshold': [0.3, 0.4, 0.5],
            'soft_distance_multiplier': [1.5, 2.0, 2.5]
        }
        
        # ===== REPUTATION SYSTEM =====
        self.reputation_grid = {
            'ema_alpha_increase': [0.3, 0.4, 0.5],
            'ema_alpha_decrease': [0.1, 0.2, 0.3],
            'penalty_flagged': [0.1, 0.2, 0.3],
            'penalty_variance': [0.05, 0.1, 0.15],
            'reward_clean': [0.05, 0.1, 0.15],
            'floor_lift_threshold': [0.3, 0.4, 0.5]
        }
        
        # ===== MODE CONTROLLER =====
        self.mode_grid = {
            'threshold_normal_to_alert': [0.15, 0.20, 0.25],
            'threshold_alert_to_defense': [0.25, 0.30, 0.35],
            'hysteresis_normal': [0.05, 0.10],
            'hysteresis_defense': [0.10, 0.15],
            'rep_gate_defense': [0.4, 0.5, 0.6]
        }
    
    def generate_layer_configs(self, layer: str, max_configs: int = None) -> List[Dict]:
        """
        Generate param combinations cho 1 layer cụ thể.
        
        Args:
            layer: 'layer1', 'layer2', 'noniid', 'filtering', 'reputation', 'mode'
            max_configs: Giới hạn số configs (None = generate ALL)
        
        Returns:
            List of param dicts
        """
        grids = {
            'layer1': self.layer1_grid,
            'layer2': self.layer2_grid,
            'noniid': self.noniid_grid,
            'filtering': self.filtering_grid,
            'reputation': self.reputation_grid,
            'mode': self.mode_grid
        }
        
        if layer not in grids:
            raise ValueError(f"Unknown layer: {layer}")
        
        grid = grids[layer]
        
        # Generate ALL combinations
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        
        all_combos = list(itertools.product(*values))
        
        print(f"\n📊 {layer.upper()} - Total combinations: {len(all_combos)}")
        
        # Sample only if max_configs is specified AND exceeded
        if max_configs is not None and len(all_combos) > max_configs:
            import random
            random.seed(42)
            all_combos = random.sample(all_combos, max_configs)
            print(f"   ⚠️  Sampled: {max_configs} configs (random sampling)")
        else:
            print(f"   ✅ Using ALL {len(all_combos)} configs")
        
        # Convert to list of dicts
        configs = []
        for combo in all_combos:
            config = {keys[i]: combo[i] for i in range(len(keys))}
            configs.append(config)
        
        return configs
    
    def generate_quick_test_suite(self) -> Dict:
        """
        Generate quick test suite - vẫn ít hơn comprehensive nhưng KHÔNG sample.
        
        Strategy: Test 1 layer tại 1 thời điểm, giữ các layers khác ở default.
        Generate ALL combinations cho mỗi layer (không giới hạn).
        
        Returns:
            Dict với structure:
            {
                'layer1': [config1, config2, ...],
                'layer2': [...],
                ...
            }
        """
        print("\n" + "="*70)
        print("GENERATING QUICK TEST SUITE - ALL COMBINATIONS")
        print("="*70)
        print("Strategy: Test each layer independently, NO SAMPLING")
        
        suite = {}
        
        # Layer 1: ALL configs (không giới hạn)
        suite['layer1'] = self.generate_layer_configs('layer1', max_configs=None)
        
        # Layer 2: ALL configs
        suite['layer2'] = self.generate_layer_configs('layer2', max_configs=None)
        
        # Non-IID: ALL configs
        suite['noniid'] = self.generate_layer_configs('noniid', max_configs=None)
        
        # Filtering: ALL configs
        suite['filtering'] = self.generate_layer_configs('filtering', max_configs=None)
        
        # Reputation: ALL configs
        suite['reputation'] = self.generate_layer_configs('reputation', max_configs=None)
        
        # Mode: ALL configs
        suite['mode'] = self.generate_layer_configs('mode', max_configs=None)
        
        total = sum(len(v) for v in suite.values())
        print(f"\n✅ Total configs to test: {total}")
        print("   (ALL possible combinations, NO sampling)")
        print("="*70)
        
        return suite
    
    def generate_comprehensive_suite(self) -> Dict:
        """
        Generate comprehensive test - TẤT CẢ combinations, NO LIMITS.
        
        Returns:
            Dict tương tự quick suite nhưng với TẤT CẢ configs
        """
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE TEST SUITE - ALL COMBINATIONS")
        print("="*70)
        print("Strategy: Generate ALL possible parameter combinations")
        print("⚠️  WARNING: This may generate THOUSANDS of configs!")
        
        suite = {}
        
        # Generate ALL combinations cho TẤT CẢ layers (no limits)
        suite['layer1'] = self.generate_layer_configs('layer1', max_configs=None)
        suite['layer2'] = self.generate_layer_configs('layer2', max_configs=None)
        suite['noniid'] = self.generate_layer_configs('noniid', max_configs=None)
        suite['filtering'] = self.generate_layer_configs('filtering', max_configs=None)
        suite['reputation'] = self.generate_layer_configs('reputation', max_configs=None)
        suite['mode'] = self.generate_layer_configs('mode', max_configs=None)
        
        total = sum(len(v) for v in suite.values())
        print(f"\n✅ Total configs to test: {total}")
        print("   (COMPLETE grid search - ALL combinations)")
        print("="*70)
        
        return suite
    
    def generate_sampled_suite(self, 
                              layer1_max: int = 50,
                              layer2_max: int = 30,
                              noniid_max: int = 30,
                              filtering_max: int = 30,
                              reputation_max: int = 50,
                              mode_max: int = 30) -> Dict:
        """
        Generate SAMPLED test suite - fast version với giới hạn configs.
        
        Use this khi muốn chạy nhanh hơn với representative sample.
        
        Returns:
            Dict với sampled configs
        """
        print("\n" + "="*70)
        print("GENERATING SAMPLED TEST SUITE")
        print("="*70)
        print("Strategy: Random sampling from full grid")
        print("Use this for faster experiments")
        
        suite = {}
        
        # Sample với limits
        suite['layer1'] = self.generate_layer_configs('layer1', max_configs=layer1_max)
        suite['layer2'] = self.generate_layer_configs('layer2', max_configs=layer2_max)
        suite['noniid'] = self.generate_layer_configs('noniid', max_configs=noniid_max)
        suite['filtering'] = self.generate_layer_configs('filtering', max_configs=filtering_max)
        suite['reputation'] = self.generate_layer_configs('reputation', max_configs=reputation_max)
        suite['mode'] = self.generate_layer_configs('mode', max_configs=mode_max)
        
        total = sum(len(v) for v in suite.values())
        print(f"\n✅ Total configs to test: {total}")
        print("   (Sampled for faster experiments)")
        print("="*70)
        
        return suite
    
    def get_default_config(self) -> Dict:
        """Get default config cho TẤT CẢ layers."""
        return {
            # Layer 1
            'pca_dims': 20,
            'dbscan_min_samples': 3,
            'dbscan_eps_multiplier': 0.5,
            'mad_k_normal': 4.0,
            'mad_k_warmup': 6.0,
            'voting_threshold_normal': 2,
            'voting_threshold_warmup': 3,
            'warmup_rounds': 10,
            
            # Layer 2
            'distance_multiplier': 1.5,
            'cosine_threshold': 0.3,
            'layer2_warmup_rounds': 15,
            
            # Non-IID
            'h_threshold_normal': 0.6,
            'h_threshold_alert': 0.5,
            'adaptive_multiplier': 1.5,
            'baseline_percentile': 60,
            
            # Filtering
            'hard_k_threshold': 3,
            'soft_reputation_threshold': 0.4,
            'soft_distance_multiplier': 2.0,
            
            # Reputation
            'ema_alpha_increase': 0.4,
            'ema_alpha_decrease': 0.2,
            'penalty_flagged': 0.2,
            'penalty_variance': 0.1,
            'reward_clean': 0.1,
            'floor_lift_threshold': 0.4,
            
            # Mode
            'threshold_normal_to_alert': 0.20,
            'threshold_alert_to_defense': 0.30,
            'hysteresis_normal': 0.10,
            'hysteresis_defense': 0.15,
            'rep_gate_defense': 0.5
        }
    
    def save_suite(self, suite: Dict, output_file: str = "param_suite.json"):
        """Save test suite to JSON."""
        
        # Add metadata
        data = {
            'generated_at': datetime.now().isoformat(),
            'total_configs': sum(len(v) for v in suite.values()),
            'default_config': self.get_default_config(),
            'suite': suite
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Saved to: {output_file}")
        print(f"   Total configs: {data['total_configs']}")
        
        return output_file


def main():
    """Main function."""
    
    print("\n" + "="*70)
    print("DEFENSE PARAMETER GENERATOR")
    print("="*70)
    
    generator = DefenseParamGenerator()
    
    # Calculate total possible combinations
    total_layer1 = 1
    for v in generator.layer1_grid.values():
        total_layer1 *= len(v)
    
    total_layer2 = 1
    for v in generator.layer2_grid.values():
        total_layer2 *= len(v)
    
    total_noniid = 1
    for v in generator.noniid_grid.values():
        total_noniid *= len(v)
    
    total_filtering = 1
    for v in generator.filtering_grid.values():
        total_filtering *= len(v)
    
    total_reputation = 1
    for v in generator.reputation_grid.values():
        total_reputation *= len(v)
    
    total_mode = 1
    for v in generator.mode_grid.values():
        total_mode *= len(v)
    
    grand_total = (total_layer1 + total_layer2 + total_noniid + 
                   total_filtering + total_reputation + total_mode)
    
    print(f"\n📊 Maximum Possible Combinations:")
    print(f"   Layer1: {total_layer1}")
    print(f"   Layer2: {total_layer2}")
    print(f"   Non-IID: {total_noniid}")
    print(f"   Filtering: {total_filtering}")
    print(f"   Reputation: {total_reputation}")
    print(f"   Mode: {total_mode}")
    print(f"   ─────────────────")
    print(f"   TOTAL: {grand_total} configs")
    print(f"\n⚠️  Estimated time @ 3min/config: {grand_total * 3 / 60:.1f} hours")
    
    # Generate quick suite (ALL combinations, per-layer)
    print("\n1️⃣  Generating QUICK test suite (ALL per-layer combinations)...")
    quick_suite = generator.generate_quick_test_suite()
    generator.save_suite(quick_suite, "param_suite_quick.json")
    
    # Generate comprehensive suite (ALL combinations)
    print("\n2️⃣  Generating COMPREHENSIVE test suite (ALL combinations)...")
    comp_suite = generator.generate_comprehensive_suite()
    generator.save_suite(comp_suite, "param_suite_comprehensive.json")
    
    # Generate sampled suite (for fast experiments)
    print("\n3️⃣  Generating SAMPLED test suite (fast experiments)...")
    sampled_suite = generator.generate_sampled_suite(
        layer1_max=50,
        layer2_max=30,
        noniid_max=30,
        filtering_max=30,
        reputation_max=50,
        mode_max=30
    )
    generator.save_suite(sampled_suite, "param_suite_sampled.json")
    
    print("\n" + "="*70)
    print("✅ GENERATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. param_suite_quick.json        - ALL per-layer configs (recommended)")
    print("  2. param_suite_comprehensive.json - ALL possible configs (very large)")
    print("  3. param_suite_sampled.json      - Sampled subset (fast testing)")
    print("\nRecommended workflow:")
    print("  • Start: param_suite_sampled.json (fast validation)")
    print("  • Normal: param_suite_quick.json (thorough per-layer)")
    print("  • Final: param_suite_comprehensive.json (complete grid search)")
    print("\nNext steps:")
    print("  python run_param_tests.py param_suite_quick.json")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()