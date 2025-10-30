"""
Layer 1 Hyperparameter Analysis Tool - Version 2
=================================================

Script phân tích kết quả từ:
- test_layer1_hyperparam_comprehensive.py
- test_layer1_hyperparam_quick.py

Features:
1. Load và parse JSON results
2. Statistical analysis cho mỗi parameter
3. Generate optimal configuration
4. Export detailed report
5. Handle missing parameters (FIXED!)

Version 2.0 - 2025-10-29
- Fixed KeyError cho missing params (pca_dims, warmup_rounds)
- Better error handling
- Cleaner output format
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


# ============================================================================
# CONFIGURATION & DEFAULTS
# ============================================================================

# Default values cho tất cả parameters
DEFAULT_PARAMS = {
    'pca_dims': 20,
    'dbscan_min_samples': 3,
    'dbscan_eps_multiplier': 0.5,
    'mad_k_normal': 4.0,
    'mad_k_warmup': 6.0,
    'voting_threshold_normal': 2,
    'voting_threshold_warmup': 3,
    'warmup_rounds': 10
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_json_results(json_file: Path) -> Dict:
    """Load và validate JSON results file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Validate structure
        if 'results' not in data:
            raise ValueError(f"Missing 'results' key in {json_file.name}")
        
        if not data['results']:
            raise ValueError(f"Empty results in {json_file.name}")
        
        return data
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {json_file.name}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading {json_file.name}: {e}")


def get_complete_params(params: Dict, data: Dict) -> Dict:
    """
    Ensure all parameters are present.
    
    Comprehensive test chỉ lưu grid params.
    Function này merge:
    1. Params từ test results (grid params)
    2. Fixed params từ data (nếu có)
    3. Default values (fallback)
    
    Args:
        params: Parameters từ test result
        data: Full data dict từ JSON
    
    Returns:
        Complete parameter dict với tất cả keys
    """
    complete = params.copy()
    
    # Get fixed params từ data (nếu có)
    fixed_params = data.get('fixed_params', {})
    
    # Fill missing params
    for key, default_value in DEFAULT_PARAMS.items():
        if key not in complete:
            # Try fixed_params first, then default
            complete[key] = fixed_params.get(key, default_value)
    
    return complete


def extract_metrics(result: Dict) -> Dict:
    """Extract metrics từ result dict (handle different formats)."""
    if 'average' in result:
        # Quick test format
        return result['average']
    elif 'avg_f1' in result:
        # Comprehensive test format
        return {
            'f1': result['avg_f1'],
            'detection_rate': result.get('avg_detection', 0),
            'fpr': result.get('avg_fpr', 0)
        }
    else:
        # Fallback
        return {
            'f1': 0,
            'detection_rate': 0,
            'fpr': 0
        }


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_parameter_sensitivity(results: List[Dict], data: Dict) -> Dict:
    """
    Analyze how each parameter affects performance.
    
    Returns:
        Dict[param_name -> Dict[value -> statistics]]
    """
    if not results:
        return {}
    
    analysis = {}
    param_names = list(results[0]['params'].keys())
    
    for param_name in param_names:
        # Group results by parameter value
        groups = {}
        
        for result in results:
            param_value = result['params'][param_name]
            
            if param_value not in groups:
                groups[param_value] = []
            
            metrics = extract_metrics(result)
            groups[param_value].append(metrics)
        
        # Compute statistics for each value
        param_stats = {}
        for value, metrics_list in groups.items():
            f1_scores = [m['f1'] for m in metrics_list]
            detection_rates = [m['detection_rate'] for m in metrics_list]
            fprs = [m['fpr'] for m in metrics_list]
            
            param_stats[value] = {
                'n_configs': len(metrics_list),
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'min_f1': np.min(f1_scores),
                'max_f1': np.max(f1_scores),
                'mean_detection': np.mean(detection_rates),
                'mean_fpr': np.mean(fprs)
            }
        
        analysis[param_name] = param_stats
    
    return analysis


def print_sensitivity_analysis(analysis: Dict):
    """Print parameter sensitivity analysis."""
    print("\n" + "="*70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    
    for param_name, param_stats in analysis.items():
        print(f"\n📊 {param_name}:")
        print("─"*70)
        
        # Sort by value
        sorted_values = sorted(param_stats.keys())
        
        for value in sorted_values:
            stats = param_stats[value]
            
            print(f"\n  {param_name} = {value}:")
            print(f"    Tested in: {stats['n_configs']} configurations")
            print(f"    Mean F1: {stats['mean_f1']:.4f} (±{stats['std_f1']:.4f})")
            print(f"    F1 Range: [{stats['min_f1']:.4f}, {stats['max_f1']:.4f}]")
            print(f"    Mean Detection: {stats['mean_detection']:.2%}")
            print(f"    Mean FPR: {stats['mean_fpr']:.2%}")
        
        # Find optimal value
        optimal_value = max(param_stats.keys(), 
                          key=lambda v: param_stats[v]['mean_f1'])
        
        print(f"\n  ✅ Optimal: {optimal_value} "
              f"(F1={param_stats[optimal_value]['mean_f1']:.4f})")


def print_top_configurations(results: List[Dict], data: Dict, top_n: int = 5):
    """Print top N configurations."""
    print("\n" + "="*70)
    print(f"TOP {top_n} CONFIGURATIONS")
    print("="*70)
    
    for rank, result in enumerate(results[:top_n], 1):
        print(f"\n🏆 Rank #{rank}:")
        print("─"*70)
        
        # Get complete params
        complete_params = get_complete_params(result['params'], data)
        
        # Print params
        print("Parameters:")
        for key, value in sorted(complete_params.items()):
            print(f"  {key}: {value}")
        
        # Print metrics
        metrics = extract_metrics(result)
        print(f"\nPerformance:")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Detection Rate: {metrics['detection_rate']:.2%}")
        print(f"  False Positive Rate: {metrics['fpr']:.2%}")


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def generate_config_file(params: Dict, metrics: Dict, 
                         output_file: str = "optimal_layer1_config.py"):
    """Generate Python configuration file."""
    
    # Ensure all params present
    complete_params = {**DEFAULT_PARAMS, **params}
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    code = f'''"""
Optimal Layer 1 DBSCAN Configuration
=====================================

Auto-generated từ hyperparameter tuning results.
Generated: {timestamp}

Performance Metrics:
  - F1 Score: {metrics.get('f1', 0):.4f}
  - Detection Rate: {metrics.get('detection_rate', 0):.2%}
  - False Positive Rate: {metrics.get('fpr', 0):.2%}

Usage:
    from optimal_layer1_config import OPTIMAL_PARAMS
    detector = Layer1Detector(**OPTIMAL_PARAMS)
"""

# ============================================================================
# OPTIMAL CONFIGURATION
# ============================================================================

OPTIMAL_PARAMS = {{
    # PCA Configuration
    'pca_dims': {complete_params['pca_dims']},
    
    # DBSCAN Configuration  
    'dbscan_min_samples': {complete_params['dbscan_min_samples']},
    'dbscan_eps_multiplier': {complete_params['dbscan_eps_multiplier']},
    
    # Magnitude Filter (MAD-based)
    'mad_k_normal': {complete_params['mad_k_normal']},
    'mad_k_warmup': {complete_params['mad_k_warmup']},
    
    # Voting Mechanism
    'voting_threshold_normal': {complete_params['voting_threshold_normal']},
    'voting_threshold_warmup': {complete_params['voting_threshold_warmup']},
    
    # Training Configuration
    'warmup_rounds': {complete_params['warmup_rounds']}
}}

# ============================================================================
# PARAMETER DESCRIPTIONS
# ============================================================================

PARAM_DESCRIPTIONS = {{
    'pca_dims': 
        'Number of PCA dimensions for dimensionality reduction (20D standard)',
    
    'dbscan_min_samples': 
        'Minimum points to form a DBSCAN cluster (detect groups of attackers)',
    
    'dbscan_eps_multiplier': 
        'Multiplier for auto-computed DBSCAN epsilon (radius for neighborhoods)',
    
    'mad_k_normal': 
        'MAD multiplier for magnitude threshold in normal mode (k=4 ≈ 2.7σ)',
    
    'mad_k_warmup': 
        'MAD multiplier for magnitude threshold in warmup mode (looser)',
    
    'voting_threshold_normal': 
        'Total votes needed to flag client in normal mode (mag=2, dbscan=1)',
    
    'voting_threshold_warmup': 
        'Total votes needed to flag client in warmup mode (stricter)',
    
    'warmup_rounds': 
        'Number of initial rounds with looser thresholds (model stabilization)'
}}

# ============================================================================
# EXPECTED PERFORMANCE
# ============================================================================

EXPECTED_PERFORMANCE = {{
    'f1_score': {metrics.get('f1', 0):.4f},
    'detection_rate': {metrics.get('detection_rate', 0):.4f},
    'false_positive_rate': {metrics.get('fpr', 0):.4f},
    'note': 'Based on synthetic test data. Validate with real FL experiments.'
}}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_layer1_config() -> dict:
    """Return optimal Layer 1 configuration."""
    return OPTIMAL_PARAMS.copy()


def print_config():
    """Print configuration with descriptions."""
    print("\\n" + "="*70)
    print("OPTIMAL LAYER 1 DBSCAN CONFIGURATION")
    print("="*70)
    
    for param, value in OPTIMAL_PARAMS.items():
        desc = PARAM_DESCRIPTIONS.get(param, '')
        print(f"\\n  {{param}}: {{value}}")
        if desc:
            # Wrap long descriptions
            words = desc.split()
            line = "    → "
            for word in words:
                if len(line) + len(word) > 70:
                    print(line)
                    line = "      " + word + " "
                else:
                    line += word + " "
            if line.strip():
                print(line)
    
    print("\\n" + "="*70)
    print("EXPECTED PERFORMANCE")
    print("="*70)
    print(f"  F1 Score: {{EXPECTED_PERFORMANCE['f1_score']:.4f}}")
    print(f"  Detection Rate: {{EXPECTED_PERFORMANCE['detection_rate']:.2%}}")
    print(f"  False Positive Rate: {{EXPECTED_PERFORMANCE['false_positive_rate']:.2%}}")
    print("\\n" + "="*70)


if __name__ == "__main__":
    print_config()
'''
    
    with open(output_file, 'w') as f:
        f.write(code)
    
    print(f"\n💾 Generated: {output_file}")
    return output_file


def generate_text_report(data: Dict, analysis: Dict, 
                        output_file: str = None) -> str:
    """Generate detailed text report."""
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"layer1_analysis_report_{timestamp}.txt"
    
    results = data.get('results', [])
    best = results[0] if results else {}
    
    with open(output_file, 'w') as f:
        # Header
        f.write("="*70 + "\n")
        f.write("LAYER 1 DBSCAN HYPERPARAMETER TUNING - ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {data.get('timestamp', 'N/A')}\n\n")
        
        # Test Configuration
        f.write("TEST CONFIGURATION\n")
        f.write("-"*70 + "\n")
        
        if 'param_grid' in data:
            f.write("Parameter Grid (tested values):\n")
            for param, values in data['param_grid'].items():
                f.write(f"  {param}: {values}\n")
        
        if 'fixed_params' in data:
            f.write("\nFixed Parameters:\n")
            for param, value in data['fixed_params'].items():
                f.write(f"  {param}: {value}\n")
        
        f.write(f"\nTotal configurations tested: {len(results)}\n")
        
        # Best Configuration
        if best:
            f.write("\n" + "="*70 + "\n")
            f.write("RECOMMENDED CONFIGURATION (BEST)\n")
            f.write("="*70 + "\n\n")
            
            complete_params = get_complete_params(best['params'], data)
            f.write("Parameters:\n")
            for param, value in sorted(complete_params.items()):
                f.write(f"  {param}: {value}\n")
            
            metrics = extract_metrics(best)
            f.write("\nExpected Performance:\n")
            f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"  Detection Rate: {metrics['detection_rate']:.2%}\n")
            f.write(f"  False Positive Rate: {metrics['fpr']:.2%}\n")
        
        # Parameter Sensitivity Summary
        f.write("\n" + "="*70 + "\n")
        f.write("PARAMETER SENSITIVITY SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        for param_name, param_stats in analysis.items():
            optimal_value = max(param_stats.keys(),
                              key=lambda v: param_stats[v]['mean_f1'])
            optimal_f1 = param_stats[optimal_value]['mean_f1']
            
            f.write(f"{param_name}:\n")
            f.write(f"  Optimal value: {optimal_value}\n")
            f.write(f"  Best F1: {optimal_f1:.4f}\n")
            f.write(f"  Tested values: {sorted(param_stats.keys())}\n\n")
        
        # Recommendations
        f.write("="*70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. Apply Configuration:\n")
        f.write("   python apply_optimal_config.py\n\n")
        
        f.write("2. Validate with Real FL:\n")
        f.write("   flwr run . --config attack-type=byzantine enable-defense=true\n\n")
        
        f.write("3. Monitor Metrics:\n")
        f.write("   - Track TP/TN/FP/FN in real experiments\n")
        f.write("   - Adjust if FPR too high in production\n\n")
        
        f.write("4. Fine-tune if needed:\n")
        f.write("   - If FPR high: increase thresholds\n")
        f.write("   - If detection low: decrease thresholds\n")
    
    print(f"📄 Generated: {output_file}")
    return output_file


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_single_file(json_file: Path):
    """Analyze a single JSON results file."""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {json_file.name}")
    print(f"{'='*70}\n")
    
    # Load data
    data = load_json_results(json_file)
    results = data['results']
    
    print(f"✓ Loaded {len(results)} configurations")
    
    # Best configuration
    best = results[0]
    complete_params = get_complete_params(best['params'], data)
    metrics = extract_metrics(best)
    
    print(f"\n🏆 BEST CONFIGURATION:")
    print("─"*70)
    print(f"Rank: #1")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Detection Rate: {metrics['detection_rate']:.2%}")
    print(f"False Positive Rate: {metrics['fpr']:.2%}")
    
    print(f"\nOptimal Parameters:")
    for key, value in sorted(complete_params.items()):
        default = DEFAULT_PARAMS.get(key, 'N/A')
        changed = "" if value == default else " ⚠️ (changed from default)"
        print(f"  {key}: {value}{changed}")
    
    # Check if results are suspiciously perfect
    if metrics['f1'] >= 0.999:
        print(f"\n⚠️  WARNING: Results are too perfect (F1 ≈ 1.0)")
        print(f"   This suggests:")
        print(f"   - Synthetic test data may be too easy")
        print(f"   - Need validation with real FL experiments")
        print(f"   - Real-world performance likely lower")
    
    # Top configurations
    print_top_configurations(results, data, top_n=3)
    
    # Parameter sensitivity
    print(f"\n📊 Running sensitivity analysis...")
    analysis = analyze_parameter_sensitivity(results, data)
    print_sensitivity_analysis(analysis)
    
    # Generate outputs
    print(f"\n{'='*70}")
    print("GENERATING OUTPUT FILES")
    print(f"{'='*70}")
    
    config_file = generate_config_file(complete_params, metrics)
    report_file = generate_text_report(data, analysis)
    
    print(f"\n✅ Analysis complete for {json_file.name}!")
    
    return config_file, report_file


def main():
    """Main entry point."""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "LAYER 1 ANALYSIS TOOL v2.0" + " "*27 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    # Find JSON files
    json_files = sorted(Path('.').glob('layer1_*.json'),
                       key=lambda x: x.stat().st_mtime,
                       reverse=True)
    
    if not json_files:
        print("❌ No result files found!")
        print("\n   Looking for files matching: layer1_*.json")
        print("\n   Please run tuning scripts first:")
        print("   - python test_layer1_hyperparam_quick.py")
        print("   - python test_layer1_hyperparam_comprehensive.py")
        return 1
    
    print(f"📂 Found {len(json_files)} result file(s):\n")
    for i, f in enumerate(json_files, 1):
        size_kb = f.stat().st_size / 1024
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        print(f"   {i}. {f.name}")
        print(f"      Size: {size_kb:.1f} KB")
        print(f"      Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Analyze each file
    all_outputs = []
    
    for json_file in json_files:
        try:
            config_file, report_file = analyze_single_file(json_file)
            all_outputs.append((config_file, report_file))
        except Exception as e:
            print(f"\n❌ Error analyzing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n\n" + "="*70)
    print("🎉 ALL ANALYSES COMPLETE!")
    print("="*70)
    
    if all_outputs:
        print("\n📦 Generated files:")
        for config_file, report_file in all_outputs:
            print(f"  ✓ {config_file}")
            print(f"  ✓ {report_file}")
        
        print("\n📝 NEXT STEPS:")
        print("  1. Review optimal_layer1_config.py")
        print("  2. Apply config: python apply_optimal_config.py")
        print("  3. Validate with real FL experiments")
        print("  4. Monitor TP/TN/FP/FN metrics")
        
        print("\n⚠️  IMPORTANT:")
        print("  If test results show F1 ≈ 1.0, validate with real data!")
        print("  Synthetic tests may be too easy compared to real FL.\n")
    else:
        print("\n❌ No outputs generated - check errors above\n")
        return 1
    
    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)