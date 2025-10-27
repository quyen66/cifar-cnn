"""
Analysis Script cho Layer 1 Hyperparameter Test Results
=========================================================

Script này analyze và visualize kết quả từ:
- test_layer1_hyperparam_comprehensive.py
- test_layer1_hyperparam_quick.py

Features:
1. Load và parse JSON results
2. Statistical analysis
3. Visualizations (if matplotlib available)
4. Generate recommendation report
5. Export optimal config to Python dict
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import sys

def load_results(json_file: str) -> Dict:
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def analyze_param_sensitivity(results: List[Dict]) -> Dict:
    """
    Analyze how each parameter affects performance.
    
    Returns dict with sensitivity analysis for each parameter.
    """
    analysis = {}
    
    # Get all unique param names
    if not results:
        return analysis
    
    param_names = list(results[0]['params'].keys())
    
    for param_name in param_names:
        # Group results by this parameter value
        groups = {}
        for result in results:
            param_value = result['params'][param_name]
            if param_value not in groups:
                groups[param_value] = []
            
            # Get average metrics
            if 'average' in result:
                groups[param_value].append(result['average'])
            elif 'avg_f1' in result:
                groups[param_value].append({
                    'f1': result['avg_f1'],
                    'detection_rate': result['avg_detection'],
                    'fpr': result['avg_fpr']
                })
        
        # Compute statistics for each value
        param_analysis = {}
        for value, metrics_list in groups.items():
            param_analysis[value] = {
                'n_configs': len(metrics_list),
                'mean_f1': np.mean([m['f1'] for m in metrics_list]),
                'std_f1': np.std([m['f1'] for m in metrics_list]),
                'mean_detection': np.mean([m['detection_rate'] for m in metrics_list]),
                'mean_fpr': np.mean([m['fpr'] for m in metrics_list])
            }
        
        analysis[param_name] = param_analysis
    
    return analysis


def print_sensitivity_analysis(analysis: Dict):
    """Print sensitivity analysis results."""
    print("\n" + "="*70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    
    for param_name, param_data in analysis.items():
        print(f"\n{param_name}:")
        print(f"{'─'*70}")
        
        # Sort by value
        sorted_values = sorted(param_data.keys())
        
        for value in sorted_values:
            stats = param_data[value]
            print(f"\n  {param_name} = {value}:")
            print(f"    Tested in {stats['n_configs']} configurations")
            print(f"    Mean F1: {stats['mean_f1']:.3f} (±{stats['std_f1']:.3f})")
            print(f"    Mean Detection: {stats['mean_detection']:.1%}")
            print(f"    Mean FPR: {stats['mean_fpr']:.1%}")
        
        # Find optimal value
        optimal_value = max(param_data.keys(), 
                          key=lambda v: param_data[v]['mean_f1'])
        print(f"\n  ✅ Optimal value: {optimal_value} "
              f"(F1 = {param_data[optimal_value]['mean_f1']:.3f})")


def generate_config_code(params: Dict, output_file: str = "optimal_layer1_config.py"):
    """Generate Python code with optimal configuration."""
    
    code = '''"""
Optimal Layer 1 DBSCAN Configuration
=====================================

Generated from hyperparameter tuning results.
Date: {timestamp}

Usage:
    from optimal_layer1_config import OPTIMAL_PARAMS
    detector = Layer1Detector(**OPTIMAL_PARAMS)
"""

# Optimal parameters for Layer 1 Detection
OPTIMAL_PARAMS = {{
    # PCA Configuration
    'pca_dims': {pca_dims},
    
    # DBSCAN Configuration
    'dbscan_min_samples': {dbscan_min_samples},
    'dbscan_eps_multiplier': {dbscan_eps_multiplier},
    
    # Magnitude Filter Configuration
    'mad_k_normal': {mad_k_normal},
    'mad_k_warmup': {mad_k_warmup},
    
    # Voting Configuration
    'voting_threshold_normal': {voting_threshold_normal},
    'voting_threshold_warmup': {voting_threshold_warmup},
    
    # Training Configuration
    'warmup_rounds': {warmup_rounds}
}}

# Parameter Explanations
PARAM_DESCRIPTIONS = {{
    'pca_dims': 'Number of PCA dimensions for dimensionality reduction',
    'dbscan_min_samples': 'Minimum points to form a DBSCAN cluster',
    'dbscan_eps_multiplier': 'Multiplier for auto-computed DBSCAN epsilon',
    'mad_k_normal': 'MAD multiplier for magnitude threshold (normal mode)',
    'mad_k_warmup': 'MAD multiplier for magnitude threshold (warmup mode)',
    'voting_threshold_normal': 'Voting threshold to flag clients (normal mode)',
    'voting_threshold_warmup': 'Voting threshold to flag clients (warmup mode)',
    'warmup_rounds': 'Number of initial rounds with looser thresholds'
}}

# Expected Performance
EXPECTED_PERFORMANCE = {{
    'detection_rate': 'See tuning results',
    'false_positive_rate': 'See tuning results',
    'f1_score': 'See tuning results'
}}

def get_layer1_config():
    """Return optimal Layer 1 configuration."""
    return OPTIMAL_PARAMS.copy()

def print_config():
    """Print configuration with descriptions."""
    print("\\nOptimal Layer 1 DBSCAN Configuration:")
    print("=" * 50)
    for param, value in OPTIMAL_PARAMS.items():
        desc = PARAM_DESCRIPTIONS.get(param, '')
        print(f"  {{param}}: {{value}}")
        if desc:
            print(f"    → {{desc}}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
'''.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **params
    )
    
    with open(output_file, 'w') as f:
        f.write(code)
    
    print(f"\n💾 Generated config code: {output_file}")
    return output_file


def generate_report(data: Dict, output_file: str = None):
    """Generate comprehensive analysis report."""
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"layer1_tuning_report_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        # Header
        f.write("="*70 + "\n")
        f.write("LAYER 1 DBSCAN HYPERPARAMETER TUNING REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {data.get('timestamp', 'N/A')}\n\n")
        
        # Test configuration
        f.write("TEST CONFIGURATION\n")
        f.write("-"*70 + "\n")
        if 'param_grid' in data:
            f.write("Parameter Grid:\n")
            for param, values in data['param_grid'].items():
                f.write(f"  {param}: {values}\n")
        if 'fixed_params' in data:
            f.write("\nFixed Parameters:\n")
            for param, value in data['fixed_params'].items():
                f.write(f"  {param}: {value}\n")
        f.write("\n")
        
        # Results summary
        results = data.get('results', [])
        if results:
            f.write("RESULTS SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Total configurations tested: {len(results)}\n\n")
            
            # Top 5
            f.write("TOP 5 CONFIGURATIONS:\n\n")
            for rank, result in enumerate(results[:5], 1):
                f.write(f"Rank #{rank}:\n")
                f.write(f"  Parameters: {result['params']}\n")
                
                if 'average' in result:
                    avg = result['average']
                    f.write(f"  F1 Score: {avg['f1']:.3f}\n")
                    f.write(f"  Detection Rate: {avg['detection_rate']:.1%}\n")
                    f.write(f"  False Positive Rate: {avg['fpr']:.1%}\n")
                elif 'avg_f1' in result:
                    f.write(f"  F1 Score: {result['avg_f1']:.3f}\n")
                    f.write(f"  Detection Rate: {result['avg_detection']:.1%}\n")
                    f.write(f"  False Positive Rate: {result['avg_fpr']:.1%}\n")
                f.write("\n")
        
        # Recommendation
        if results:
            best = results[0]
            f.write("\n" + "="*70 + "\n")
            f.write("RECOMMENDED CONFIGURATION\n")
            f.write("="*70 + "\n\n")
            f.write("Parameters:\n")
            for param, value in best['params'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\nExpected Performance:\n")
            if 'average' in best:
                f.write(f"  F1 Score: {best['average']['f1']:.3f}\n")
                f.write(f"  Detection Rate: {best['average']['detection_rate']:.1%}\n")
                f.write(f"  FPR: {best['average']['fpr']:.1%}\n")
            elif 'avg_f1' in best:
                f.write(f"  F1 Score: {best['avg_f1']:.3f}\n")
                f.write(f"  Detection Rate: {best['avg_detection']:.1%}\n")
                f.write(f"  FPR: {best['avg_fpr']:.1%}\n")
    
    print(f"📄 Generated report: {output_file}")
    return output_file


def main():
    """Main analysis function."""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*18 + "LAYER 1 RESULTS ANALYZER" + " "*26 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    # Find JSON files
    json_files = list(Path('.').glob('layer1_*.json'))
    
    if not json_files:
        print("❌ No result files found!")
        print("   Please run test scripts first:")
        print("   - python test_layer1_hyperparam_quick.py")
        print("   - python test_layer1_hyperparam_comprehensive.py")
        return
    
    print(f"📂 Found {len(json_files)} result file(s):\n")
    for i, f in enumerate(json_files, 1):
        print(f"   {i}. {f.name}")
    
    # Analyze each file
    for json_file in json_files:
        print(f"\n\n{'='*70}")
        print(f"ANALYZING: {json_file.name}")
        print(f"{'='*70}")
        
        # Load data
        data = load_results(str(json_file))
        results = data.get('results', [])
        
        if not results:
            print("⚠️  No results found in file")
            continue
        
        print(f"\n✓ Loaded {len(results)} configurations")
        
        # Best configuration
        best = results[0]
        print(f"\n🏆 Best Configuration:")
        print(f"   Parameters: {best['params']}")
        if 'average' in best:
            print(f"   F1: {best['average']['f1']:.3f}")
            print(f"   Detection: {best['average']['detection_rate']:.1%}")
            print(f"   FPR: {best['average']['fpr']:.1%}")
        elif 'avg_f1' in best:
            print(f"   F1: {best['avg_f1']:.3f}")
            print(f"   Detection: {best['avg_detection']:.1%}")
            print(f"   FPR: {best['avg_fpr']:.1%}")
        
        # Sensitivity analysis
        print(f"\n📊 Running sensitivity analysis...")
        sensitivity = analyze_param_sensitivity(results)
        print_sensitivity_analysis(sensitivity)
        
        # Generate outputs
        print(f"\n📝 Generating outputs...")
        
        # Config code
        config_file = generate_config_code(best['params'])
        
        # Report
        report_file = generate_report(data)
        
        print(f"\n✅ Analysis complete for {json_file.name}!")
    
    print("\n\n" + "="*70)
    print("🎉 ALL ANALYSES COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - optimal_layer1_config.py (use this in your code)")
    print("  - layer1_tuning_report_*.txt (detailed report)")
    print("\nNext steps:")
    print("  1. Review the optimal configuration")
    print("  2. Update your Layer1Detector with these parameters")
    print("  3. Run full FL experiments to validate")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()