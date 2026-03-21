#!/usr/bin/env python3
"""
Layer1 Results Analyzer - Realistic Layer 1 Performance Targets
================================================================

Priority (Layer 1 Isolated Performance):
1. Detection Rate: 75-82% (catch most attackers)
2. FPR: 5-8% (some false positives acceptable for Layer 1)
3. F1 Score: 75-80% (balanced precision/recall)

Rationale:
- Layer 1 is first-line detection, not final decision
- Some FP acceptable as Layer 2 + Reputation will filter
- FPR=0% unrealistic and indicates too conservative settings
- Need balance between catching attackers and minimizing false alarms

Usage:
    python analyze_layer1_results_v2.py param_test_results_*.json
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
    
    # F1 score
    recall = detection_rate
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'detection_rate': detection_rate,
        'fpr': fpr,
        'precision': precision,
        'f1': f1,
        'accuracy': result.get('accuracy', 0),
    }


def grade_config(metrics):
    """Grade config based on realistic Layer 1 targets."""
    
    detection = metrics['detection_rate']
    fpr = metrics['fpr']
    f1 = metrics['f1']
    
    # Ideal ranges
    detection_ideal = 0.75 <= detection <= 0.82
    fpr_ideal = 0.05 <= fpr <= 0.08
    f1_ideal = 0.75 <= f1 <= 0.80
    
    # Acceptable ranges
    detection_acceptable = 0.68 <= detection < 0.75 or 0.82 < detection <= 0.90
    fpr_acceptable = 0.03 <= fpr < 0.05 or 0.08 < fpr <= 0.12
    f1_acceptable = 0.70 <= f1 < 0.75 or 0.80 < f1 <= 0.85
    
    # Count how many are ideal
    ideal_count = sum([detection_ideal, fpr_ideal, f1_ideal])
    acceptable_count = sum([detection_acceptable, fpr_acceptable, f1_acceptable])
    
    # Grade
    if ideal_count >= 2:
        return 'A', 100 - abs(detection - 0.78) * 50 - abs(fpr - 0.065) * 200 - abs(f1 - 0.775) * 50
    elif ideal_count >= 1 or acceptable_count >= 2:
        return 'B', 80 - abs(detection - 0.75) * 50 - abs(fpr - 0.08) * 150
    elif acceptable_count >= 1:
        return 'C', 60
    elif detection >= 0.55 and fpr <= 0.15:
        return 'D', 50
    else:
        return 'F', 0


def find_optimal_configs(results):
    """
    Find optimal configs based on realistic Layer 1 targets:
    1. Detection: 75-82%
    2. FPR: 5-8%
    3. F1: 75-80%
    """
    
    print("\n" + "="*80)
    print("🎯 FINDING OPTIMAL CONFIGS (Realistic Layer 1 Targets)")
    print("="*80)
    
    # Calculate metrics for all
    results_with_metrics = []
    for r in results:
        metrics = calculate_metrics(r)
        grade, score = grade_config(metrics)
        results_with_metrics.append({
            'config': r['config'],
            'config_idx': r['config_idx'],
            'grade': grade,
            'score': score,
            **metrics
        })
    
    # Stats
    print(f"\n📊 Total configs tested: {len(results_with_metrics)}")
    
    # Filter by criteria
    print(f"\n🎯 TARGET RANGES:")
    print(f"   Detection Rate: 75-82% (ideal) or 68-90% (acceptable)")
    print(f"   FPR: 5-8% (ideal) or 3-12% (acceptable)")
    print(f"   F1 Score: 75-80% (ideal) or 70-85% (acceptable)")
    
    # Grade distribution
    grade_counts = defaultdict(int)
    for r in results_with_metrics:
        grade_counts[r['grade']] += 1
    
    print(f"\n📈 GRADE DISTRIBUTION:")
    for grade in ['A', 'B', 'C', 'D', 'F']:
        count = grade_counts.get(grade, 0)
        pct = count / len(results_with_metrics) * 100 if results_with_metrics else 0
        print(f"   Grade {grade}: {count:3d} configs ({pct:5.1f}%)")
    
    # Filter by ranges
    ideal = [r for r in results_with_metrics 
             if 0.75 <= r['detection_rate'] <= 0.82 
             and 0.05 <= r['fpr'] <= 0.08 
             and 0.75 <= r['f1'] <= 0.80]
    
    print(f"\n✨ IDEAL CONFIGS (all 3 criteria in ideal range):")
    print(f"   Count: {len(ideal)}")
    
    acceptable = [r for r in results_with_metrics
                 if 0.68 <= r['detection_rate'] <= 0.90
                 and 0.03 <= r['fpr'] <= 0.12
                 and 0.70 <= r['f1'] <= 0.85]
    
    print(f"\n✅ ACCEPTABLE CONFIGS (all 3 criteria in acceptable range):")
    print(f"   Count: {len(acceptable)}")
    
    # Partial matches
    detection_match = [r for r in results_with_metrics 
                      if 0.75 <= r['detection_rate'] <= 0.82]
    fpr_match = [r for r in results_with_metrics 
                if 0.05 <= r['fpr'] <= 0.08]
    f1_match = [r for r in results_with_metrics 
               if 0.75 <= r['f1'] <= 0.80]
    
    print(f"\n📊 PARTIAL MATCHES:")
    print(f"   Detection 75-82%: {len(detection_match)} configs")
    print(f"   FPR 5-8%: {len(fpr_match)} configs")
    print(f"   F1 75-80%: {len(f1_match)} configs")
    
    # Select optimal
    if len(ideal) > 0:
        optimal = ideal
        print(f"\n🎉 Using {len(optimal)} IDEAL configs")
    elif len(acceptable) > 0:
        optimal = acceptable
        print(f"\n✅ Using {len(optimal)} ACCEPTABLE configs")
    else:
        # Sort by grade score
        optimal = sorted(results_with_metrics, key=lambda x: -x['score'])[:20]
        print(f"\n⚠️  No configs meet criteria, showing top 20 by score")
    
    # Sort optimal by: grade score (descending)
    optimal = sorted(optimal, key=lambda x: -x['score'])
    
    return optimal


def print_config_details(config, rank):
    """Print detailed config info."""
    
    grade = config['grade']
    score = config['score']
    
    # Grade emoji
    grade_emoji = {
        'A': '🏆',
        'B': '⭐',
        'C': '✅',
        'D': '⚠️',
        'F': '❌'
    }
    
    print(f"\n{'─'*80}")
    print(f"RANK #{rank} - GRADE {grade} {grade_emoji.get(grade, '')} (Score: {score:.1f})")
    print(f"{'─'*80}")
    
    # Metrics with target indicators
    detection = config['detection_rate']
    fpr = config['fpr']
    f1 = config['f1']
    
    # Check if in target ranges
    det_status = '🎯' if 0.75 <= detection <= 0.82 else ('✅' if 0.68 <= detection <= 0.90 else '❌')
    fpr_status = '🎯' if 0.05 <= fpr <= 0.08 else ('✅' if 0.03 <= fpr <= 0.12 else '❌')
    f1_status = '🎯' if 0.75 <= f1 <= 0.80 else ('✅' if 0.70 <= f1 <= 0.85 else '❌')
    
    print(f"\n📊 METRICS:")
    print(f"   Detection Rate: {detection:>6.1%} {det_status}  (target: 75-82%)")
    print(f"   FPR:            {fpr:>6.1%} {fpr_status}  (target: 5-8%)")
    print(f"   F1 Score:       {f1:>6.1%} {f1_status}  (target: 75-80%)")
    print(f"   Precision:      {config['precision']:>6.1%}")
    print(f"   Accuracy:       {config['accuracy']:>6.2%}")
    
    print(f"\n📈 DETECTION STATS:")
    print(f"   TP (True Pos):  {config['tp']}")
    print(f"   FP (False Pos): {config['fp']}")
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
    print("📋 TOP 20 CONFIGS SUMMARY")
    print("="*80)
    
    print(f"\n{'Rank':<6} {'Grade':<7} {'Detect':<9} {'FPR':<9} {'F1':<9} {'MAD-k':<7} {'Vote':<6} {'Cfg#':<6}")
    print("─"*80)
    
    for i, cfg in enumerate(optimal[:20], 1):
        mad_k = cfg['config'].get('mad_k_normal', 0)
        vote = cfg['config'].get('voting_threshold_normal', 0)
        
        grade_emoji = {
            'A': '🏆',
            'B': '⭐',
            'C': '✅',
            'D': '⚠️',
            'F': '❌'
        }
        
        print(f"#{i:<5} {cfg['grade']} {grade_emoji.get(cfg['grade'], '')}    "
              f"{cfg['detection_rate']:>6.1%}    "
              f"{cfg['fpr']:>6.1%}    "
              f"{cfg['f1']:>6.1%}    "
              f"{mad_k:<7.1f} {vote:<6} {cfg['config_idx']:<6}")


def analyze_patterns(optimal):
    """Analyze patterns in optimal configs."""
    
    print("\n" + "="*80)
    print("🔍 PARAMETER PATTERNS IN TOP CONFIGS")
    print("="*80)
    
    # Separate by grade
    grade_a = [c for c in optimal if c['grade'] == 'A']
    grade_b = [c for c in optimal if c['grade'] == 'B']
    
    if len(grade_a) > 0:
        print(f"\n🏆 GRADE A CONFIGS ({len(grade_a)}):")
        analyze_param_group(grade_a)
    
    if len(grade_b) > 0:
        print(f"\n⭐ GRADE B CONFIGS ({len(grade_b)}):")
        analyze_param_group(grade_b)
    
    # Overall top 10
    print(f"\n📊 TOP 10 OVERALL:")
    analyze_param_group(optimal[:10])


def analyze_param_group(configs):
    """Analyze parameter patterns in a group of configs."""
    
    from collections import Counter
    
    params = defaultdict(list)
    for cfg in configs:
        for key, value in cfg['config'].items():
            params[key].append(value)
    
    for param_name in ['mad_k_normal', 'voting_threshold_normal', 'dbscan_min_samples', 
                       'dbscan_eps_multiplier', 'pca_dims']:
        if param_name in params:
            values = params[param_name]
            freq = Counter(values)
            most_common = freq.most_common(3)
            
            # Calculate range
            min_val = min(values)
            max_val = max(values)
            avg_val = sum(values) / len(values)
            
            print(f"\n   {param_name}:")
            print(f"      Range: {min_val} - {max_val}, Avg: {avg_val:.2f}")
            print(f"      Most common: {most_common[0][0]} ({most_common[0][1]}x)")


def main():
    """Main analysis function."""
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_layer1_results_v2.py <results_file.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    print("\n" + "="*80)
    print("LAYER1 RESULTS ANALYZER V2 - Realistic Targets")
    print("="*80)
    print(f"\n📁 Loading: {results_file}")
    
    # Load results
    results = load_results(results_file)
    print(f"   Total results: {len(results)}")
    
    # Find optimal
    optimal = find_optimal_configs(results)
    
    # Print top 20 details
    print("\n" + "="*80)
    print("🏆 TOP 20 OPTIMAL CONFIGS (Detailed)")
    print("="*80)
    
    for i, cfg in enumerate(optimal[:20], 1):
        print_config_details(cfg, i)
    
    # Summary table
    print_summary(optimal)
    
    # Pattern analysis
    analyze_patterns(optimal)
    
    # Save optimal configs
    output_file = 'optimal_layer1_configs_v2.json'
    with open(output_file, 'w') as f:
        json.dump({
            'optimal_configs': optimal[:20],
            'selection_criteria': {
                'detection_rate': '75-82% (ideal) or 68-90% (acceptable)',
                'fpr': '5-8% (ideal) or 3-12% (acceptable)',
                'f1_score': '75-80% (ideal) or 70-85% (acceptable)',
                'rationale': 'Layer 1 is first-line detection. Some FP acceptable as later layers will filter.'
            },
            'grading_scale': {
                'A': '2+ metrics in ideal range',
                'B': '1+ ideal or 2+ acceptable',
                'C': '1+ acceptable',
                'D': 'Detection ≥55%, FPR ≤15%',
                'F': 'Below minimum thresholds'
            }
        }, f, indent=2)
    
    print(f"\n💾 Optimal configs saved to: {output_file}")
    
    # Final recommendation
    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80)
    
    if len(optimal) > 0:
        best = optimal[0]
        print(f"\n🎯 BEST CONFIG (Rank #1) - Grade {best['grade']}:")
        print(f"   Detection: {best['detection_rate']:.1%} {'🎯' if 0.75 <= best['detection_rate'] <= 0.82 else ''}")
        print(f"   FPR: {best['fpr']:.1%} {'🎯' if 0.05 <= best['fpr'] <= 0.08 else ''}")
        print(f"   F1: {best['f1']:.1%} {'🎯' if 0.75 <= best['f1'] <= 0.80 else ''}")
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
        print(f"   pca-dims = {cfg.get('pca_dims', 'N/A')}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()