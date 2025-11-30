#!/usr/bin/env python3
"""
H Weights Results Analyzer
===========================

Analyze H weight optimization results and identify best configurations.

Phase 1: Analyze ONLY H weight impact on system performance.

Usage:
    python3 analyze_h_weights.py <h_weight_results_final.json>

Output:
    - Console report with rankings
    - optimal_h_weights_TIMESTAMP.json
    - Recommendations for Phase 2
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import statistics


class HWeightAnalyzer:
    """Analyzer for H weight optimization results."""
    
    def __init__(self, results_file: str):
        """
        Initialize analyzer.
        
        Args:
            results_file: Path to results JSON
        """
        self.results_file = results_file
        
        # Load results
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        self.metadata = data.get('metadata', {})
        self.results = data.get('results', [])
        
        # Filter successful results
        self.valid_results = [
            r for r in self.results 
            if r.get('status') == 'success' and r.get('metrics', {}).get('final_accuracy') is not None
        ]
        
        print(f"✅ Loaded results: {len(self.results)} total, {len(self.valid_results)} valid")
    
    def calculate_composite_score(self, dr: float, fpr: float, accuracy: float = None) -> float:
        """
        Calculate composite score based on optimization document.
        
        Formula: Score = DR - (FPR × 2.0) + (Accuracy × 0.5)
        
        Priorities:
        1. FPR <= 10% (ideally < 5%) - Heavy penalty
        2. DR >= 80% - Required detection
        3. Accuracy - Model quality
        
        Args:
            dr: Detection rate [0, 1]
            fpr: False positive rate [0, 1]
            accuracy: Model accuracy [0, 1] (optional)
        
        Returns:
            Composite score
        """
        if dr is None or fpr is None:
            return -999.0
        
        score = dr - (fpr * 2.0)
        
        # Add accuracy bonus if available
        if accuracy is not None:
            score += (accuracy * 0.5)
        
        return score
    
    def calculate_f1_score(self, tp: int, fp: int, fn: int) -> float:
        """Calculate F1 score."""
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def rank_configs(self, metric: str = 'composite_enhanced') -> List[Dict]:
        """
        Rank configurations by metric.
        
        Args:
            metric: Ranking metric
                - 'composite_enhanced': DR - (FPR × 2.0) + (Accuracy × 0.5)
                - 'detection_rate': Pure DR
                - 'accuracy': Model accuracy
                - 'f1_score': F1 score
        
        Returns:
            Sorted list of results with scores
        """
        scored_results = []
        
        for result in self.valid_results:
            metrics = result.get('metrics', {})
            
            dr = metrics.get('detection_rate')
            fpr = metrics.get('false_positive_rate')
            acc = metrics.get('final_accuracy')
            avg_h = metrics.get('avg_h_score') or metrics.get('h_score_mean')
            tp = metrics.get('true_positives', 0)
            fp = metrics.get('false_positives', 0)
            fn = metrics.get('false_negatives', 0)
            
            # Calculate scores
            composite = self.calculate_composite_score(dr, fpr, acc)
            f1 = self.calculate_f1_score(tp, fp, fn)
            
            if metric == 'composite_enhanced':
                score = composite
            elif metric == 'detection_rate':
                score = dr if dr is not None else -999.0
            elif metric == 'accuracy':
                score = acc if acc is not None else -999.0
            elif metric == 'f1_score':
                score = f1
            else:
                score = composite
            
            scored_results.append({
                'result': result,
                'score': score,
                'composite': composite,
                'f1': f1,
                'avg_h_score': avg_h
            })
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_results
    
    def analyze_weight_impact(self) -> Dict:
        """Analyze impact of individual weight components."""
        
        # Group by dominant weight
        cv_dominant = []
        sim_dominant = []
        cluster_dominant = []
        balanced = []
        
        for result in self.valid_results:
            config = result.get('config', {})
            cv = config.get('weight_cv', 0)
            sim = config.get('weight_sim', 0)
            cluster = config.get('weight_cluster', 0)
            
            metrics = result.get('metrics', {})
            dr = metrics.get('detection_rate', 0)
            fpr = metrics.get('false_positive_rate', 0)
            
            max_weight = max(cv, sim, cluster)
            
            if max_weight == cv and cv >= 0.5:
                cv_dominant.append({'dr': dr, 'fpr': fpr})
            elif max_weight == sim and sim >= 0.5:
                sim_dominant.append({'dr': dr, 'fpr': fpr})
            elif max_weight == cluster and cluster >= 0.3:
                cluster_dominant.append({'dr': dr, 'fpr': fpr})
            else:
                balanced.append({'dr': dr, 'fpr': fpr})
        
        def avg_metrics(group):
            if not group:
                return {'dr': 0, 'fpr': 0, 'count': 0}
            return {
                'dr': statistics.mean([x['dr'] for x in group]),
                'fpr': statistics.mean([x['fpr'] for x in group]),
                'count': len(group)
            }
        
        return {
            'cv_dominant': avg_metrics(cv_dominant),
            'sim_dominant': avg_metrics(sim_dominant),
            'cluster_dominant': avg_metrics(cluster_dominant),
            'balanced': avg_metrics(balanced)
        }
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        
        print()
        print("=" * 80)
        print("H WEIGHTS OPTIMIZATION ANALYSIS")
        print("=" * 80)
        print()
        
        # Overall statistics
        print("📊 OVERALL STATISTICS")
        print("-" * 80)
        print(f"   Total configs tested: {len(self.results)}")
        print(f"   Valid results:        {len(self.valid_results)}")
        print(f"   Failed tests:         {len(self.results) - len(self.valid_results)}")
        print()
        
        # Metric ranges
        if self.valid_results:
            drs = [r.get('metrics', {}).get('detection_rate', 0) for r in self.valid_results]
            fprs = [r.get('metrics', {}).get('false_positive_rate', 0) for r in self.valid_results]
            accs = [r.get('metrics', {}).get('final_accuracy', 0) for r in self.valid_results]
            
            print("📈 METRIC RANGES")
            print("-" * 80)
            print(f"   Detection Rate:  {min(drs):.2%} - {max(drs):.2%} "
                  f"(mean: {statistics.mean(drs):.2%})")
            print(f"   FP Rate:         {min(fprs):.2%} - {max(fprs):.2%} "
                  f"(mean: {statistics.mean(fprs):.2%})")
            print(f"   Accuracy:        {min(accs):.4f} - {max(accs):.4f} "
                  f"(mean: {statistics.mean(accs):.4f})")
            print()
        
        # Weight impact analysis
        print("🔍 WEIGHT COMPONENT IMPACT")
        print("-" * 80)
        
        impact = self.analyze_weight_impact()
        
        for group_name, metrics in impact.items():
            if metrics['count'] > 0:
                print(f"   {group_name:20s}: "
                      f"DR={metrics['dr']:.2%}, FPR={metrics['fpr']:.2%} "
                      f"(n={metrics['count']})")
        print()
        
        # Top configurations
        print("🏆 TOP 5 CONFIGURATIONS (by Enhanced Composite Score)")
        print("-" * 80)
        print(f"   {'Rank':<6} {'Config':<8} {'CV':<6} {'Sim':<6} {'Cluster':<8} "
              f"{'DR':<8} {'FPR':<8} {'Acc':<7} {'H_avg':<7} {'Score':<8}")
        print("-" * 80)
        
        ranked = self.rank_configs('composite_enhanced')
        
        for rank, item in enumerate(ranked[:5], 1):
            result = item['result']
            config = result.get('config', {})
            metrics = result.get('metrics', {})
            
            cv = config.get('weight_cv', 0)
            sim = config.get('weight_sim', 0)
            cluster = config.get('weight_cluster', 0)
            dr = metrics.get('detection_rate', 0)
            fpr = metrics.get('false_positive_rate', 0)
            acc = metrics.get('final_accuracy', 0)
            avg_h = item.get('avg_h_score', 0)
            
            # Formatting
            h_str = f"{avg_h:.3f}" if avg_h else "N/A"
            acc_str = f"{acc:.3f}" if acc else "N/A"
            
            print(f"   {rank:<6} #{config.get('config_id', '?'):<7} "
                  f"{cv:<6.2f} {sim:<6.2f} {cluster:<8.2f} "
                  f"{dr:<8.2%} {fpr:<8.2%} {acc_str:<7} {h_str:<7} {item['score']:<8.3f}")
        
        print()
        
        # Best by objective
        print("🎯 BEST BY OBJECTIVE")
        print("-" * 80)
        
        # Best detection rate
        best_dr = max(ranked, key=lambda x: x['result']['metrics'].get('detection_rate', 0))
        print(f"   Best Detection Rate:")
        print(f"      Config #{best_dr['result']['config']['config_id']}: "
              f"DR={best_dr['result']['metrics']['detection_rate']:.2%}, "
              f"FPR={best_dr['result']['metrics']['false_positive_rate']:.2%}")
        print(f"      Weights: CV={best_dr['result']['config']['weight_cv']:.2f}, "
              f"sim={best_dr['result']['config']['weight_sim']:.2f}, "
              f"cluster={best_dr['result']['config']['weight_cluster']:.2f}")
        print()
        
        # Best FPR (lowest)
        best_fpr = min(ranked, key=lambda x: x['result']['metrics'].get('false_positive_rate', 1))
        print(f"   Best FP Rate (lowest):")
        print(f"      Config #{best_fpr['result']['config']['config_id']}: "
              f"DR={best_fpr['result']['metrics']['detection_rate']:.2%}, "
              f"FPR={best_fpr['result']['metrics']['false_positive_rate']:.2%}")
        print(f"      Weights: CV={best_fpr['result']['config']['weight_cv']:.2f}, "
              f"sim={best_fpr['result']['config']['weight_sim']:.2f}, "
              f"cluster={best_fpr['result']['config']['weight_cluster']:.2f}")
        print()
        
        # Priority analysis
        print("=" * 80)
        print("🎯 PRIORITY ANALYSIS (Based on Optimization Document)")
        print("=" * 80)
        print()
        print("Priorities:")
        print("   1. FPR ≤ 10% (ideally < 5%) - Protect good clients")
        print("   2. DR ≥ 80% - Detect attacks")
        print("   3. Accuracy - Model quality")
        print()
        
        # Find configs meeting priorities
        priority_met = []
        for item in ranked:
            metrics = item['result'].get('metrics', {})
            fpr = metrics.get('false_positive_rate', 1.0)
            dr = metrics.get('detection_rate', 0.0)
            
            if fpr <= 0.10 and dr >= 0.80:
                priority_met.append(item)
        
        print(f"Configs meeting BOTH priorities: {len(priority_met)}/{len(ranked)}")
        print()
        
        if priority_met:
            print("✅ EXCELLENT (FPR ≤ 5%, DR ≥ 80%):")
            excellent = [x for x in priority_met 
                        if x['result']['metrics']['false_positive_rate'] <= 0.05]
            if excellent:
                for item in excellent[:3]:
                    config = item['result']['config']
                    metrics = item['result']['metrics']
                    avg_h = item.get('avg_h_score')
                    h_str = f", H_avg={avg_h:.3f}" if avg_h else ""
                    print(f"   Config #{config['config_id']:2d}: "
                          f"DR={metrics['detection_rate']:.1%}, "
                          f"FPR={metrics['false_positive_rate']:.1%}, "
                          f"Acc={metrics.get('final_accuracy', 0):.3f}"
                          f"{h_str}")
            else:
                print("   None")
            print()
            
            print("✅ GOOD (FPR 5-10%, DR ≥ 80%):")
            good = [x for x in priority_met 
                   if 0.05 < x['result']['metrics']['false_positive_rate'] <= 0.10]
            if good:
                for item in good[:3]:
                    config = item['result']['config']
                    metrics = item['result']['metrics']
                    avg_h = item.get('avg_h_score')
                    h_str = f", H_avg={avg_h:.3f}" if avg_h else ""
                    print(f"   Config #{config['config_id']:2d}: "
                          f"DR={metrics['detection_rate']:.1%}, "
                          f"FPR={metrics['false_positive_rate']:.1%}, "
                          f"Acc={metrics.get('final_accuracy', 0):.3f}"
                          f"{h_str}")
            else:
                print("   None")
            print()
        else:
            print("⚠️  NO configs meet both priorities!")
            print()
            print("Possible issues:")
            print("   1. Testing with IID data (H score will be ~0)")
            print("   2. Need more H weight combinations")
            print("   3. Thresholds too strict")
            print()
            print("Check avg_H_score in results:")
            if ranked:
                h_scores = [x.get('avg_h_score') for x in ranked[:5] if x.get('avg_h_score')]
                if h_scores:
                    avg_of_avgs = sum(h_scores) / len(h_scores)
                    print(f"   Average H score (top 5): {avg_of_avgs:.3f}")
                    if avg_of_avgs < 0.3:
                        print("   ⚠️  H scores too low! Likely using IID data!")
                        print("   → Use run_h_weight_tests_noniid.py instead")
                else:
                    print("   ⚠️  No H scores tracked in results")
            print()
        
        # Recommendations
        print("=" * 80)
        print("💡 RECOMMENDATIONS")
        print("=" * 80)
        print()
        
        best_overall = ranked[0]
        best_config = best_overall['result']['config']
        best_metrics = best_overall['result']['metrics']
        best_avg_h = best_overall.get('avg_h_score')
        
        print(f"🎯 RECOMMENDED H WEIGHTS for Phase 2:")
        print(f"   weight-cv:      {best_config['weight_cv']:.2f}")
        print(f"   weight-sim:     {best_config['weight_sim']:.2f}")
        print(f"   weight-cluster: {best_config['weight_cluster']:.2f}")
        print()
        print(f"   Expected Performance:")
        print(f"      Detection Rate: {best_metrics['detection_rate']:.2%}")
        print(f"      FP Rate:        {best_metrics['false_positive_rate']:.2%}", end="")
        
        # FPR status
        fpr = best_metrics['false_positive_rate']
        if fpr <= 0.05:
            print(" ✅ Excellent!")
        elif fpr <= 0.10:
            print(" ✅ Good")
        else:
            print(" ⚠️  High (> 10%)")
        
        print(f"      Accuracy:       {best_metrics['final_accuracy']:.4f}")
        
        if best_avg_h:
            print(f"      Avg H Score:    {best_avg_h:.3f}", end="")
            if best_avg_h < 0.3:
                print(" ⚠️  Low (check if using IID data)")
            elif best_avg_h >= 0.5:
                print(" ✅ Good (Non-IID detected)")
            else:
                print(" ⚠️  Moderate")
        
        print()
        print(f"   Composite Score: {best_overall['score']:.3f}")
        print(f"   Formula: DR - (FPR × 2.0) + (Acc × 0.5)")
        print()
        
        # Analysis of H score impact
        if best_avg_h:
            print(f"📊 H Score Analysis:")
            print(f"   Avg H = {best_avg_h:.3f}")
            print()
            if best_avg_h < 0.3:
                print("   ⚠️  CRITICAL ISSUE:")
                print("   H score too low for Non-IID Handler to work properly!")
                print()
                print("   Likely causes:")
                print("   1. Testing with IID data (WRONG for Non-IID Handler)")
                print("   2. H weights not reflecting data heterogeneity")
                print()
                print("   Solutions:")
                print("   1. Re-run with non-IID data:")
                print("      python3 run_h_weight_tests_noniid.py suite.json")
                print("   2. Check data partition in results:")
                print("      Should be 'dirichlet' not 'iid'")
            elif best_avg_h >= 0.5:
                print("   ✅ H score indicates Non-IID Handler is active")
                print("   Adaptive thresholds likely being applied correctly")
            else:
                print("   ⚠️  Moderate H score")
                print("   Non-IID Handler partially active")
                print("   Consider testing with more heterogeneous data")
            print()
        
        # Save optimal config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        optimal_file = f"optimal_h_weights_{timestamp}.json"
        
        optimal_data = {
            'metadata': {
                'source_file': self.results_file,
                'analysis_timestamp': datetime.now().isoformat(),
                'optimization_phase': 'Phase 1 - H Weights',
                'ranking_metric': 'composite_enhanced',
                'scoring_formula': 'DR - (FPR × 2.0) + (Accuracy × 0.5)',
                'priorities': {
                    'priority_1': 'FPR <= 10% (ideally < 5%)',
                    'priority_2': 'DR >= 80%',
                    'priority_3': 'Accuracy - highest possible'
                }
            },
            'optimal_config': {
                'id': f"h_weights_opt_{best_config['config_id']:03d}",
                'params': best_config,
                'metrics': {
                    'tp': best_metrics.get('true_positives'),
                    'fp': best_metrics.get('false_positives'),
                    'tn': best_metrics.get('true_negatives'),
                    'fn': best_metrics.get('false_negatives'),
                    'detection_rate': best_metrics['detection_rate'],
                    'fpr': best_metrics['false_positive_rate'],
                    'accuracy': best_metrics['final_accuracy'],
                    'avg_h_score': best_avg_h
                },
                'score': best_overall['score']
            },
            'top_5_configs': [
                {
                    'rank': i + 1,
                    'id': f"h_weights_opt_{item['result']['config']['config_id']:03d}",
                    'params': item['result']['config'],
                    'metrics': {
                        'tp': item['result']['metrics'].get('true_positives'),
                        'fp': item['result']['metrics'].get('false_positives'),
                        'tn': item['result']['metrics'].get('true_negatives'),
                        'fn': item['result']['metrics'].get('false_negatives'),
                        'detection_rate': item['result']['metrics']['detection_rate'],
                        'fpr': item['result']['metrics']['false_positive_rate'],
                        'accuracy': item['result']['metrics'].get('final_accuracy'),
                        'avg_h_score': item.get('avg_h_score')
                    },
                    'score': item['score']
                }
                for i, item in enumerate(ranked[:5])
            ],
            'weight_impact_analysis': impact,
            'configs_meeting_priorities': {
                'excellent': [
                    {
                        'config_id': x['result']['config']['config_id'],
                        'fpr': x['result']['metrics']['false_positive_rate'],
                        'dr': x['result']['metrics']['detection_rate'],
                        'accuracy': x['result']['metrics'].get('final_accuracy'),
                        'avg_h_score': x.get('avg_h_score')
                    }
                    for x in priority_met if x['result']['metrics']['false_positive_rate'] <= 0.05
                ] if priority_met else [],
                'good': [
                    {
                        'config_id': x['result']['config']['config_id'],
                        'fpr': x['result']['metrics']['false_positive_rate'],
                        'dr': x['result']['metrics']['detection_rate'],
                        'accuracy': x['result']['metrics'].get('final_accuracy'),
                        'avg_h_score': x.get('avg_h_score')
                    }
                    for x in priority_met 
                    if 0.05 < x['result']['metrics']['false_positive_rate'] <= 0.10
                ] if priority_met else []
            }
        }
        
        with open(optimal_file, 'w') as f:
            json.dump(optimal_data, f, indent=2)
        
        print(f"💾 Optimal configuration saved: {optimal_file}")
        print()
        print("=" * 80)
        print("NEXT STEPS: Phase 2 Optimization")
        print("=" * 80)
        print()
        print("1. Update pyproject.toml with optimal H weights:")
        print(f"   [tool.flwr.app.config.defense.noniid]")
        print(f"   weight-cv = {best_config['weight_cv']:.2f}")
        print(f"   weight-sim = {best_config['weight_sim']:.2f}")
        print(f"   weight-cluster = {best_config['weight_cluster']:.2f}")
        print()
        print("2. Generate Phase 2 suite (optimize other params):")
        print("   python3 generate_noniid_params.py  # Full suite with optimal H weights")
        print()
        print("3. Run Phase 2 tests:")
        print("   python3 run_param_tests.py noniid_suite_*.json")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_h_weights.py <h_weight_results_final.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        sys.exit(1)
    
    # Analyze
    analyzer = HWeightAnalyzer(results_file)
    analyzer.generate_report()


if __name__ == "__main__":
    main()