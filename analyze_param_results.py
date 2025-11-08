"""
Parameter Test Results Analyzer
================================

Phân tích results từ run_param_tests.py và tìm optimal params cho mỗi layer.

Output:
1. Top configs cho mỗi layer
2. Parameter sensitivity analysis
3. Recommendations
4. Code để apply optimal configs

Usage:
    python analyze_param_results.py param_test_results_*.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np


class ParamResultsAnalyzer:
    """Analyzer cho parameter test results."""
    
    def __init__(self, results_file: str):
        """
        Args:
            results_file: Path to results JSON
        """
        self.results_file = Path(results_file)
        
        # Load results
        with open(self.results_file, 'r') as f:
            self.data = json.load(f)
        
        self.results = self.data['results']
        self.failures = self.data.get('failures', [])
        
        # Organize by layer
        self.layer_results = self._organize_by_layer()
    
    def _organize_by_layer(self) -> Dict[str, List[Dict]]:
        """Organize results by layer."""
        organized = {}
        
        for result in self.results:
            layer = result['layer']
            if layer not in organized:
                organized[layer] = []
            organized[layer].append(result)
        
        return organized
    
    def analyze_layer(self, layer: str) -> Dict:
        """
        Analyze results cho 1 layer.
        
        Returns:
            Dict với:
            - top_configs: Top N configs
            - param_sensitivity: Sensitivity analysis
            - optimal_params: Best param values
        """
        if layer not in self.layer_results:
            return None
        
        results = self.layer_results[layer]
        
        # Sort by accuracy (descending)
        sorted_results = sorted(results, key=lambda x: x.get('accuracy', 0), reverse=True)
        
        # Top configs
        top_n = min(5, len(sorted_results))
        top_configs = sorted_results[:top_n]
        
        # Parameter sensitivity
        param_sensitivity = self._analyze_param_sensitivity(results)
        
        # Optimal params (from best config)
        optimal_params = top_configs[0]['config'] if top_configs else {}
        
        # Statistics
        accuracies = [r['accuracy'] for r in results]
        
        analysis = {
            'layer': layer,
            'total_tests': len(results),
            'top_configs': top_configs,
            'param_sensitivity': param_sensitivity,
            'optimal_params': optimal_params,
            'statistics': {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'median_accuracy': np.median(accuracies)
            }
        }
        
        return analysis
    
    def _analyze_param_sensitivity(self, results: List[Dict]) -> Dict:
        """
        Analyze parameter sensitivity.
        
        For each parameter, tính mean accuracy cho mỗi value.
        """
        sensitivity = {}
        
        # Get all param names
        if not results:
            return sensitivity
        
        param_names = results[0]['config'].keys()
        
        for param_name in param_names:
            # Group by param value
            value_groups = {}
            
            for result in results:
                value = result['config'][param_name]
                
                # Convert to hashable type
                if isinstance(value, (list, dict)):
                    continue
                
                if value not in value_groups:
                    value_groups[value] = []
                
                value_groups[value].append(result['accuracy'])
            
            # Compute stats cho mỗi value
            value_stats = {}
            for value, accuracies in value_groups.items():
                value_stats[str(value)] = {
                    'mean': float(np.mean(accuracies)),
                    'std': float(np.std(accuracies)),
                    'count': len(accuracies)
                }
            
            # Find best value
            best_value = max(value_stats.keys(), key=lambda v: value_stats[v]['mean'])
            
            sensitivity[param_name] = {
                'values': value_stats,
                'best_value': best_value,
                'best_mean': value_stats[best_value]['mean']
            }
        
        return sensitivity
    
    def print_summary(self):
        """Print analysis summary."""
        
        print("\n" + "="*80)
        print("PARAMETER TEST RESULTS ANALYSIS")
        print("="*80)
        print(f"Results file: {self.results_file.name}")
        print(f"Total tests: {len(self.results)}")
        print(f"Failed tests: {len(self.failures)}")
        print("="*80 + "\n")
        
        # Analyze each layer
        for layer in sorted(self.layer_results.keys()):
            self._print_layer_analysis(layer)
        
        # Overall recommendations
        self._print_recommendations()
    
    def _print_layer_analysis(self, layer: str):
        """Print analysis cho 1 layer."""
        
        analysis = self.analyze_layer(layer)
        
        if not analysis:
            return
        
        print(f"\n{'='*80}")
        print(f"📊 {layer.upper()} ANALYSIS")
        print(f"{'='*80}")
        
        stats = analysis['statistics']
        print(f"\nStatistics (n={analysis['total_tests']}):")
        print(f"  Mean accuracy: {stats['mean_accuracy']:.3f} (±{stats['std_accuracy']:.3f})")
        print(f"  Range: [{stats['min_accuracy']:.3f}, {stats['max_accuracy']:.3f}]")
        print(f"  Median: {stats['median_accuracy']:.3f}")
        
        # Top configs
        print(f"\n🏆 Top {len(analysis['top_configs'])} Configurations:")
        print("─"*80)
        
        for rank, config in enumerate(analysis['top_configs'], 1):
            print(f"\n  Rank #{rank}  (Accuracy: {config['accuracy']:.3f})")
            print(f"  Parameters:")
            for key, value in sorted(config['config'].items()):
                print(f"    {key}: {value}")
        
        # Parameter sensitivity
        print(f"\n📈 Parameter Sensitivity:")
        print("─"*80)
        
        sensitivity = analysis['param_sensitivity']
        
        for param_name, param_data in sorted(sensitivity.items()):
            print(f"\n  {param_name}:")
            
            # Sort values by mean
            sorted_values = sorted(
                param_data['values'].items(),
                key=lambda x: x[1]['mean'],
                reverse=True
            )
            
            for value, stats in sorted_values[:3]:  # Top 3 values
                print(f"    {value}: {stats['mean']:.3f} (±{stats['std']:.3f}, n={stats['count']})")
            
            best = param_data['best_value']
            print(f"    ✅ Best: {best}")
        
        # Optimal params summary
        print(f"\n✨ Optimal Parameters:")
        print("─"*80)
        for key, value in sorted(analysis['optimal_params'].items()):
            print(f"  {key}: {value}")
    
    def _print_recommendations(self):
        """Print overall recommendations."""
        
        print(f"\n{'='*80}")
        print("💡 RECOMMENDATIONS")
        print("="*80)
        
        print("\nNext steps:")
        print("  1. Review optimal parameters for each layer above")
        print("  2. Apply optimal configs using generated code (see below)")
        print("  3. Run full test suite with optimal configs")
        print("  4. Compare with baseline performance")
        
        print("\n⚠️  Important notes:")
        print("  • Results based on single attack scenario")
        print("  • May need adjustment for different attacks")
        print("  • Consider trade-offs between layers")
        print("  • Validate with comprehensive tests")
        
        print("="*80 + "\n")
    
    def generate_apply_script(self, output_file: str = "apply_optimal_params.py"):
        """Generate Python script để apply optimal params."""
        
        # Collect optimal params từ tất cả layers
        all_optimal = {}
        
        for layer in self.layer_results.keys():
            analysis = self.analyze_layer(layer)
            if analysis and analysis['optimal_params']:
                all_optimal[layer] = analysis['optimal_params']
        
        # Generate script
        script = '''"""
Auto-generated script to apply optimal parameters
==================================================

Generated from: {}
Generated at: {}

This script updates all defense layer files with optimal parameters.

Usage:
    python apply_optimal_params.py
"""

import sys
from pathlib import Path

# ============================================================================
# OPTIMAL PARAMETERS
# ============================================================================

OPTIMAL_PARAMS = {}

# ============================================================================
# FILE UPDATERS
# ============================================================================

def update_layer1():
    """Update Layer 1 DBSCAN parameters."""
    params = OPTIMAL_PARAMS.get('layer1', {{}})
    
    if not params:
        print("⚠️  No optimal params for layer1")
        return
    
    print("\\n📝 Updating Layer 1 DBSCAN...")
    
    # TODO: Implement actual file update logic
    # Similar to apply_optimal_config.py
    
    for key, value in sorted(params.items()):
        print(f"  {{key}}: {{value}}")
    
    print("  ✅ Layer 1 updated")

def update_layer2():
    """Update Layer 2 Distance+Direction parameters."""
    params = OPTIMAL_PARAMS.get('layer2', {{}})
    
    if not params:
        print("⚠️  No optimal params for layer2")
        return
    
    print("\\n📝 Updating Layer 2 Distance+Direction...")
    
    for key, value in sorted(params.items()):
        print(f"  {{key}}: {{value}}")
    
    print("  ✅ Layer 2 updated")

def update_noniid():
    """Update Non-IID Handler parameters."""
    params = OPTIMAL_PARAMS.get('noniid', {{}})
    
    if not params:
        print("⚠️  No optimal params for noniid")
        return
    
    print("\\n📝 Updating Non-IID Handler...")
    
    for key, value in sorted(params.items()):
        print(f"  {{key}}: {{value}}")
    
    print("  ✅ Non-IID updated")

def update_filtering():
    """Update Two-Stage Filtering parameters."""
    params = OPTIMAL_PARAMS.get('filtering', {{}})
    
    if not params:
        print("⚠️  No optimal params for filtering")
        return
    
    print("\\n📝 Updating Two-Stage Filtering...")
    
    for key, value in sorted(params.items()):
        print(f"  {{key}}: {{value}}")
    
    print("  ✅ Filtering updated")

def update_reputation():
    """Update Reputation System parameters."""
    params = OPTIMAL_PARAMS.get('reputation', {{}})
    
    if not params:
        print("⚠️  No optimal params for reputation")
        return
    
    print("\\n📝 Updating Reputation System...")
    
    for key, value in sorted(params.items()):
        print(f"  {{key}}: {{value}}")
    
    print("  ✅ Reputation updated")

def update_mode():
    """Update Mode Controller parameters."""
    params = OPTIMAL_PARAMS.get('mode', {{}})
    
    if not params:
        print("⚠️  No optimal params for mode")
        return
    
    print("\\n📝 Updating Mode Controller...")
    
    for key, value in sorted(params.items()):
        print(f"  {{key}}: {{value}}")
    
    print("  ✅ Mode updated")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\\n" + "="*70)
    print("APPLYING OPTIMAL PARAMETERS")
    print("="*70)
    
    update_layer1()
    update_layer2()
    update_noniid()
    update_filtering()
    update_reputation()
    update_mode()
    
    print("\\n" + "="*70)
    print("✅ ALL PARAMETERS APPLIED!")
    print("="*70)
    print("\\nNext: Run full test suite to validate\\n")

if __name__ == "__main__":
    main()
'''.format(self.results_file.name, self.data.get('end_time', 'unknown'), all_optimal)
        
        # Save script
        with open(output_file, 'w') as f:
            f.write(script)
        
        print(f"📄 Generated apply script: {output_file}")
        
        return output_file


def main():
    """Main function."""
    
    if len(sys.argv) < 2:
        # Find latest results file
        results_files = sorted(Path('.').glob('param_test_results_*.json'), 
                             key=lambda x: x.stat().st_mtime, 
                             reverse=True)
        
        if not results_files:
            print("❌ No results files found")
            print("Usage: python analyze_param_results.py <results_json>")
            sys.exit(1)
        
        results_file = results_files[0]
        print(f"📂 Using latest results: {results_file.name}\n")
    else:
        results_file = sys.argv[1]
    
    # Create analyzer
    analyzer = ParamResultsAnalyzer(results_file)
    
    # Print summary
    analyzer.print_summary()
    
    # Generate apply script
    analyzer.generate_apply_script()


if __name__ == "__main__":
    main()