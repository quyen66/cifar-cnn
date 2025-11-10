#!/usr/bin/env python3
"""
Layer1 Results Analyzer - Focus on FP and Detection
====================================================

Priority:
1. FP = 0 (no false positives)
2. Detection Rate > 80%
3. High accuracy

Usage:
    python analyze_layer1_results.py param_test_results_*.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_results(results_file):
    """Load results from JSON."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data['results']


def calculate_metrics(result):
    """Calculate detection metrics from result."""
    
    # Get detection stats from result
    detection = result.get('detection', {})
    
    tp = detection.get('tp', 0)
    fp = detection.get('fp', 0)
    fn = detection.get('fn', 0)
    tn = detection.get('tn', 0)
    
    # Calculate rates
    total_malicious = tp + fn
    total_benign = fp + tn
    
    detection_rate = tp / total_malicious if total_malicious > 0 else 0
    fpr = fp / total_benign if total_benign > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'detection_rate': detection_rate,
        'fpr': fpr,
        'precision': precision,
        'accuracy': result.get('accuracy', 0),
    }


def find_optimal_configs(results):
    """
    Find optimal configs based on priority:
    1. FP = 0
    2. Detection > 80%
    3. Accuracy > 50%
    """
    
    print("\n" + "="*80)
    print("🎯 FINDING OPTIMAL CONFIGS")
    print("="*80)
    
    # Calculate metrics for all
    results_with_metrics = []
    for r in results:
        metrics = calculate_metrics(r)
        results_with_metrics.append({
            'config': r['config'],
            'config_idx': r['config_idx'],
            **metrics
        })
    
    # Filter by criteria
    print(f"\n📊 Total configs tested: {len(results_with_metrics)}")
    
    # Criterion 1: FP = 0
    zero_fp = [r for r in results_with_metrics if r['fp'] == 0]
    print(f"   Configs with FP = 0: {len(zero_fp)}")
    
    if len(zero_fp) == 0:
        print("\n❌ No configs with FP = 0 found!")
        print("   Showing configs with LOWEST FP instead...\n")
        
        # Sort by FP then detection
        sorted_by_fp = sorted(results_with_metrics, 
                             key=lambda x: (x['fp'], -x['detection_rate'], -x['accuracy']))
        zero_fp = sorted_by_fp[:20]  # Top 20 lowest FP
    
    # Criterion 2: Detection > 80%
    high_detection = [r for r in zero_fp if r['detection_rate'] >= 0.80]
    print(f"   └─ With Detection ≥ 80%: {len(high_detection)}")
    
    # Criterion 3: Accuracy > 50%
    high_accuracy = [r for r in high_detection if r['accuracy'] >= 0.50]
    print(f"      └─ With Accuracy ≥ 50%: {len(high_accuracy)}")
    
    # If we have high_accuracy results, use those
    if len(high_accuracy) > 0:
        optimal = high_accuracy
        print(f"\n✅ Found {len(optimal)} OPTIMAL configs!")
    # Otherwise try lowering criteria
    elif len(high_detection) > 0:
        optimal = high_detection
        print(f"\n⚠️  No configs meet all criteria")
        print(f"   Using {len(optimal)} configs with FP=0 and Detection≥80%")
    elif len(zero_fp) > 0:
        optimal = zero_fp
        print(f"\n⚠️  No configs meet detection criteria")
        print(f"   Using {len(optimal)} configs with FP=0")
    else:
        optimal = results_with_metrics[:10]
        print(f"\n⚠️  No configs meet FP criteria")
        print(f"   Using top 10 configs")
    
    # Sort optimal by: FP → Detection → Accuracy
    optimal = sorted(optimal, 
                    key=lambda x: (x['fp'], -x['detection_rate'], -x['accuracy']))
    
    return optimal


def print_config_details(config, rank):
    """Print detailed config info."""
    
    print(f"\n{'─'*80}")
    print(f"RANK #{rank}")
    print(f"{'─'*80}")
    
    # Metrics
    print(f"\n📊 METRICS:")
    print(f"   Accuracy:       {config['accuracy']:.4f}")
    print(f"   Detection Rate: {config['detection_rate']:.2%}")
    print(f"   FPR:            {config['fpr']:.2%}")
    print(f"   Precision:      {config['precision']:.2%}")
    
    print(f"\n📈 DETECTION STATS:")
    print(f"   TP (True Pos):  {config['tp']}")
    print(f"   FP (False Pos): {config['fp']} ← TARGET = 0")
    print(f"   FN (False Neg): {config['fn']}")
    print(f"   TN (True Neg):  {config['tn']}")
    
    # Config params
    print(f"\n⚙️  LAYER1 PARAMETERS:")
    cfg = config['config']
    print(f"   MAD k-normal:        {cfg.get('mad_k_normal', 'N/A')}")
    print(f"   MAD k-warmup:        {cfg.get('mad_k_warmup', 'N/A')}")
    print(f"   Voting normal:       {cfg.get('voting_threshold_normal', 'N/A')}")
    print(f"   Voting warmup:       {cfg.get('voting_threshold_warmup', 'N/A')}")
    print(f"   DBSCAN min_samples:  {cfg.get('dbscan_min_samples', 'N/A')}")
    print(f"   DBSCAN eps_mult:     {cfg.get('dbscan_eps_multiplier', 'N/A')}")
    print(f"   Warmup rounds:       {cfg.get('warmup_rounds', 'N/A')}")
    print(f"   PCA dims:            {cfg.get('pca_dims', 'N/A')}")


def print_summary(optimal):
    """Print summary of optimal configs."""
    
    print("\n" + "="*80)
    print("📋 TOP 10 CONFIGS SUMMARY")
    print("="*80)
    
    print(f"\n{'Rank':<6} {'FP':<4} {'Detection':<12} {'Accuracy':<10} {'MAD-k':<8} {'Vote':<8} {'Config#':<8}")
    print("─"*80)
    
    for i, cfg in enumerate(optimal[:10], 1):
        mad_k = cfg['config'].get('mad_k_normal', 0)
        vote = cfg['config'].get('voting_threshold_normal', 0)
        
        print(f"#{i:<5} {cfg['fp']:<4} {cfg['detection_rate']:>6.1%}       "
              f"{cfg['accuracy']:>6.2%}     {mad_k:<8.1f} {vote:<8} {cfg['config_idx']:<8}")


def analyze_patterns(optimal):
    """Analyze patterns in optimal configs."""
    
    print("\n" + "="*80)
    print("🔍 PARAMETER PATTERNS IN OPTIMAL CONFIGS")
    print("="*80)
    
    # Collect param values
    params = defaultdict(list)
    
    for cfg in optimal[:10]:  # Top 10
        for key, value in cfg['config'].items():
            params[key].append(value)
    
    print("\n📊 Most Common Values in Top 10:")
    
    for param_name, values in params.items():
        # Count frequency
        from collections import Counter
        freq = Counter(values)
        most_common = freq.most_common(3)
        
        print(f"\n   {param_name}:")
        for value, count in most_common:
            print(f"      {value}: {count}/10 times ({count*10}%)")


def main():
    """Main analysis function."""
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_layer1_results.py <results_file.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    print("\n" + "="*80)
    print("LAYER1 RESULTS ANALYZER")
    print("="*80)
    print(f"\n📁 Loading: {results_file}")
    
    # Load results
    results = load_results(results_file)
    print(f"   Total results: {len(results)}")
    
    # Find optimal
    optimal = find_optimal_configs(results)
    
    # Print top 10 details
    print("\n" + "="*80)
    print("🏆 TOP 10 OPTIMAL CONFIGS (Detailed)")
    print("="*80)
    
    for i, cfg in enumerate(optimal[:10], 1):
        print_config_details(cfg, i)
    
    # Summary table
    print_summary(optimal)
    
    # Pattern analysis
    analyze_patterns(optimal)
    
    # Save optimal configs
    output_file = 'optimal_layer1_configs.json'
    with open(output_file, 'w') as f:
        json.dump({
            'optimal_configs': optimal[:10],
            'selection_criteria': {
                'priority_1': 'FP = 0',
                'priority_2': 'Detection Rate ≥ 80%',
                'priority_3': 'Accuracy ≥ 50%'
            }
        }, f, indent=2)
    
    print(f"\n💾 Optimal configs saved to: {output_file}")
    
    # Final recommendation
    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80)
    
    if len(optimal) > 0:
        best = optimal[0]
        print(f"\n🎯 BEST CONFIG (Rank #1):")
        print(f"   FP: {best['fp']}")
        print(f"   Detection: {best['detection_rate']:.1%}")
        print(f"   Accuracy: {best['accuracy']:.2%}")
        print(f"\n   Apply to pyproject.toml:")
        print(f"   ─────────────────────────")
        cfg = best['config']
        print(f"   mad-k-normal = {cfg.get('mad_k_normal', 'N/A')}")
        print(f"   mad-k-warmup = {cfg.get('mad_k_warmup', 'N/A')}")
        print(f"   voting-threshold-normal = {cfg.get('voting_threshold_normal', 'N/A')}")
        print(f"   voting-threshold-warmup = {cfg.get('voting_threshold_warmup', 'N/A')}")
        print(f"   dbscan-min-samples = {cfg.get('dbscan_min_samples', 'N/A')}")
        print(f"   dbscan-eps-multiplier = {cfg.get('dbscan_eps_multiplier', 'N/A')}")
        print(f"   warmup-rounds = {cfg.get('warmup_rounds', 'N/A')}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()