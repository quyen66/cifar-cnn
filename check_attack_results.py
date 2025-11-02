#!/usr/bin/env python3
"""
Comprehensive Attack Results Checker
=====================================
Phân tích và so sánh kết quả từ tất cả các attack tests.

Usage:
    python check_attack_results.py [results_dir]
    
Example:
    python check_attack_results.py results/comprehensive_attacks
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from collections import defaultdict


# =================================================================
# DATA LOADING
# =================================================================

def load_metadata(model_dir: str) -> Dict:
    """Load metadata.json from a model directory."""
    metadata_file = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}


def extract_all_results(results_dir: str) -> Dict[str, Dict]:
    """
    Extract results from all experiment directories.
    
    Returns:
        Dict mapping experiment_name -> metadata
    """
    experiments = {}
    
    # Find all subdirectories
    for exp_dir in glob.glob(os.path.join(results_dir, "*")):
        if not os.path.isdir(exp_dir):
            continue
        
        exp_name = os.path.basename(exp_dir)
        
        # Find the latest model in this experiment
        model_dirs = glob.glob(os.path.join(exp_dir, "*"))
        if not model_dirs:
            continue
        
        # Get the most recent model
        latest_model = max(model_dirs, key=os.path.getctime)
        
        metadata = load_metadata(latest_model)
        if metadata:
            experiments[exp_name] = metadata
    
    return experiments


def parse_experiment_name(exp_name: str) -> Dict:
    """
    Parse experiment name to extract attack info.
    
    Example:
        "byzantine_signflip_scale2_defense" -> {
            'attack_type': 'byzantine',
            'variant': 'signflip_scale2',
            'has_defense': True
        }
    """
    info = {
        'attack_type': 'unknown',
        'variant': 'default',
        'has_defense': 'defense' in exp_name and 'nodefense' not in exp_name
    }
    
    # Baseline
    if 'baseline' in exp_name or 'clean' in exp_name:
        info['attack_type'] = 'none'
        return info
    
    # Label flip
    if 'labelflip' in exp_name:
        info['attack_type'] = 'label_flip'
        if 'reverse' in exp_name:
            if '50pct' in exp_name:
                info['variant'] = 'reverse_50%'
            else:
                info['variant'] = 'reverse_100%'
        elif 'random' in exp_name:
            info['variant'] = 'random'
    
    # Byzantine
    elif 'byzantine' in exp_name:
        info['attack_type'] = 'byzantine'
        
        # Extract type
        if 'signflip' in exp_name:
            byz_type = 'sign_flip'
        elif 'scaled' in exp_name:
            byz_type = 'scaled'
        elif 'random' in exp_name:
            byz_type = 'random'
        else:
            byz_type = 'unknown'
        
        # Extract scale
        if 'scale1' in exp_name:
            scale = '1.0'
        elif 'scale2' in exp_name:
            scale = '2.0'
        elif 'scale3' in exp_name:
            scale = '3.0'
        elif 'scale5' in exp_name:
            scale = '5.0'
        elif 'scale10' in exp_name:
            scale = '10.0'
        else:
            scale = 'unknown'
        
        # Extract attack ratio
        if '10pct' in exp_name:
            ratio = '10%'
        elif '20pct' in exp_name:
            ratio = '20%'
        elif '40pct' in exp_name:
            ratio = '40%'
        else:
            ratio = '30%'  # default
        
        info['variant'] = f"{byz_type}_{scale}_{ratio}"
    
    # Gaussian
    elif 'gaussian' in exp_name:
        info['attack_type'] = 'gaussian'
        if 'mild' in exp_name:
            info['variant'] = 'mild_0.01'
        elif 'medium' in exp_name:
            info['variant'] = 'medium_0.1'
        elif 'strong' in exp_name:
            info['variant'] = 'strong_0.2'
        elif 'verystrong' in exp_name:
            info['variant'] = 'verystrong_0.5'
    
    return info


# =================================================================
# ANALYSIS FUNCTIONS
# =================================================================

def analyze_accuracy_comparison(experiments: Dict) -> None:
    """Compare accuracy between attack types and defense effectiveness."""
    print("\n" + "="*80)
    print("📊 ACCURACY COMPARISON ANALYSIS")
    print("="*80 + "\n")
    
    # Group by attack type
    attack_groups = defaultdict(lambda: {'with_defense': [], 'without_defense': []})
    baseline_acc = None
    
    for exp_name, metadata in experiments.items():
        info = parse_experiment_name(exp_name)
        best_acc = metadata.get('best_accuracy', 0)
        
        if info['attack_type'] == 'none':
            baseline_acc = best_acc
            print(f"🎯 BASELINE (No Attack): {best_acc:.4f} ({best_acc*100:.2f}%)")
            continue
        
        key = f"{info['attack_type']}_{info['variant']}"
        if info['has_defense']:
            attack_groups[key]['with_defense'].append(best_acc)
        else:
            attack_groups[key]['without_defense'].append(best_acc)
    
    print("\n" + "-"*80)
    print("Attack Type Breakdown:")
    print("-"*80 + "\n")
    
    # Sort by attack type
    sorted_groups = sorted(attack_groups.items(), key=lambda x: x[0])
    
    for key, values in sorted_groups:
        attack_type, variant = key.split('_', 1)
        
        without_def = values['without_defense']
        with_def = values['with_defense']
        
        print(f"📌 {attack_type.upper()} - {variant}:")
        
        if without_def:
            avg_without = np.mean(without_def)
            print(f"   ❌ Without Defense: {avg_without:.4f} ({avg_without*100:.2f}%)")
            if baseline_acc:
                drop = (baseline_acc - avg_without) / baseline_acc * 100
                print(f"      └─ Drop from baseline: {drop:.2f}%")
        
        if with_def:
            avg_with = np.mean(with_def)
            print(f"   ✅ With Defense:    {avg_with:.4f} ({avg_with*100:.2f}%)")
            if baseline_acc:
                drop = (baseline_acc - avg_with) / baseline_acc * 100
                print(f"      └─ Drop from baseline: {drop:.2f}%")
        
        if without_def and with_def:
            improvement = (np.mean(with_def) - np.mean(without_def)) / np.mean(without_def) * 100
            print(f"   🛡️  Defense Improvement: {improvement:+.2f}%")
        
        print()


def analyze_detection_metrics(experiments: Dict) -> None:
    """Analyze detection metrics for experiments with defense."""
    print("\n" + "="*80)
    print("🎯 DETECTION METRICS ANALYSIS")
    print("="*80 + "\n")
    
    defense_experiments = {k: v for k, v in experiments.items() 
                          if v.get('defense_stats')}
    
    if not defense_experiments:
        print("⚠️  No experiments with defense metrics found!")
        return
    
    # Group by attack type
    attack_groups = defaultdict(list)
    
    for exp_name, metadata in defense_experiments.items():
        info = parse_experiment_name(exp_name)
        if info['attack_type'] == 'none':
            continue
        
        defense_stats = metadata.get('defense_stats', {})
        attack_groups[info['attack_type']].append((exp_name, defense_stats))
    
    for attack_type, exp_list in sorted(attack_groups.items()):
        print(f"\n{'━'*80}")
        print(f"🎯 {attack_type.upper()} Attack Detection")
        print(f"{'━'*80}\n")
        
        for exp_name, stats in exp_list:
            info = parse_experiment_name(exp_name)
            
            total_tp = stats.get('total_tp', 0)
            total_fp = stats.get('total_fp', 0)
            total_fn = stats.get('total_fn', 0)
            total_tn = stats.get('total_tn', 0)
            
            # Calculate metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
            
            print(f"   Variant: {info['variant']}")
            print(f"   ├─ TP: {total_tp:4d}  FP: {total_fp:4d}")
            print(f"   ├─ FN: {total_fn:4d}  TN: {total_tn:4d}")
            print(f"   ├─ Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
            print(f"   └─ False Positive Rate: {fpr:.3f} ({fpr*100:.1f}%)")
            print()


def create_summary_table(experiments: Dict) -> None:
    """Create a comprehensive summary table."""
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE SUMMARY TABLE")
    print("="*80 + "\n")
    
    # Organize data
    summary_data = []
    
    for exp_name, metadata in experiments.items():
        info = parse_experiment_name(exp_name)
        
        best_acc = metadata.get('best_accuracy', 0)
        final_acc = metadata.get('accuracy_history', [0])[-1]
        
        defense_stats = metadata.get('defense_stats', {})
        total_tp = defense_stats.get('total_tp', 0)
        total_fp = defense_stats.get('total_fp', 0)
        
        summary_data.append({
            'experiment': exp_name,
            'attack_type': info['attack_type'],
            'variant': info['variant'],
            'defense': '✓' if info['has_defense'] else '✗',
            'best_acc': best_acc,
            'final_acc': final_acc,
            'tp': total_tp if info['has_defense'] else '-',
            'fp': total_fp if info['has_defense'] else '-'
        })
    
    # Sort by attack type and defense
    summary_data.sort(key=lambda x: (x['attack_type'], not (x['defense'] == '✓'), x['variant']))
    
    # Print table
    print(f"{'Attack Type':<15} {'Variant':<25} {'Def':<4} {'Best Acc':>10} {'Final Acc':>10} {'TP':>6} {'FP':>6}")
    print("─"*90)
    
    current_attack = None
    for row in summary_data:
        if row['attack_type'] != current_attack:
            if current_attack is not None:
                print()
            current_attack = row['attack_type']
        
        print(f"{row['attack_type']:<15} {row['variant']:<25} {row['defense']:<4} "
              f"{row['best_acc']:>10.4f} {row['final_acc']:>10.4f} "
              f"{str(row['tp']):>6} {str(row['fp']):>6}")


# =================================================================
# VISUALIZATION
# =================================================================

def plot_accuracy_comparison(experiments: Dict, save_dir: str) -> None:
    """Create comparison plots for accuracy."""
    print("\n📊 Generating accuracy comparison plots...")
    
    # Prepare data
    attack_types = ['label_flip', 'byzantine', 'gaussian']
    data_by_type = defaultdict(lambda: {'no_defense': [], 'with_defense': []})
    baseline_acc = None
    
    for exp_name, metadata in experiments.items():
        info = parse_experiment_name(exp_name)
        best_acc = metadata.get('best_accuracy', 0)
        
        if info['attack_type'] == 'none':
            baseline_acc = best_acc
            continue
        
        if info['attack_type'] in attack_types:
            key = 'with_defense' if info['has_defense'] else 'no_defense'
            data_by_type[info['attack_type']][key].append(best_acc)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Grouped bar chart
    ax1 = axes[0]
    x = np.arange(len(attack_types))
    width = 0.35
    
    no_def_means = [np.mean(data_by_type[at]['no_defense']) if data_by_type[at]['no_defense'] else 0 
                    for at in attack_types]
    with_def_means = [np.mean(data_by_type[at]['with_defense']) if data_by_type[at]['with_defense'] else 0 
                      for at in attack_types]
    
    bars1 = ax1.bar(x - width/2, no_def_means, width, label='No Defense', 
                    color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, with_def_means, width, label='With Defense', 
                    color='#27ae60', alpha=0.8)
    
    if baseline_acc:
        ax1.axhline(y=baseline_acc, color='#3498db', linestyle='--', 
                   linewidth=2, label='Baseline (No Attack)')
    
    ax1.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Attack Impact on Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([at.replace('_', ' ').title() for at in attack_types])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Defense improvement percentage
    ax2 = axes[1]
    improvements = []
    for at in attack_types:
        if data_by_type[at]['no_defense'] and data_by_type[at]['with_defense']:
            no_def = np.mean(data_by_type[at]['no_defense'])
            with_def = np.mean(data_by_type[at]['with_defense'])
            improvement = (with_def - no_def) / no_def * 100 if no_def > 0 else 0
            improvements.append(improvement)
        else:
            improvements.append(0)
    
    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax2.bar(attack_types, improvements, color=colors, alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Defense System Improvement', fontsize=14, fontweight='bold')
    ax2.set_xticklabels([at.replace('_', ' ').title() for at in attack_types])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'accuracy_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved accuracy comparison plot: {save_path}")
    plt.close()


def plot_detection_performance(experiments: Dict, save_dir: str) -> None:
    """Create plots for detection performance."""
    print("\n📊 Generating detection performance plots...")
    
    defense_experiments = {k: v for k, v in experiments.items() 
                          if v.get('defense_stats')}
    
    if not defense_experiments:
        print("⚠️  No defense experiments found for plotting")
        return
    
    # Prepare data
    attack_types = ['label_flip', 'byzantine', 'gaussian']
    metrics_by_type = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': [], 'fpr': []})
    
    for exp_name, metadata in defense_experiments.items():
        info = parse_experiment_name(exp_name)
        if info['attack_type'] not in attack_types:
            continue
        
        stats = metadata.get('defense_stats', {})
        total_tp = stats.get('total_tp', 0)
        total_fp = stats.get('total_fp', 0)
        total_fn = stats.get('total_fn', 0)
        total_tn = stats.get('total_tn', 0)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
        
        metrics_by_type[info['attack_type']]['precision'].append(precision)
        metrics_by_type[info['attack_type']]['recall'].append(recall)
        metrics_by_type[info['attack_type']]['f1'].append(f1)
        metrics_by_type[info['attack_type']]['fpr'].append(fpr)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detection Performance Metrics', fontsize=16, fontweight='bold')
    
    metric_names = ['precision', 'recall', 'f1', 'fpr']
    metric_titles = ['Precision', 'Recall', 'F1 Score', 'False Positive Rate']
    
    for idx, (metric, title) in enumerate(zip(metric_names, metric_titles)):
        ax = axes[idx // 2, idx % 2]
        
        means = [np.mean(metrics_by_type[at][metric]) if metrics_by_type[at][metric] else 0 
                for at in attack_types]
        stds = [np.std(metrics_by_type[at][metric]) if metrics_by_type[at][metric] else 0 
               for at in attack_types]
        
        x = np.arange(len(attack_types))
        bars = ax.bar(x, means, yerr=stds, capsize=5, 
                     color='#3498db', alpha=0.8, error_kw={'linewidth': 2})
        
        ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} by Attack Type', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([at.replace('_', ' ').title() for at in attack_types])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1 if metric != 'fpr' else 0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'detection_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved detection performance plot: {save_path}")
    plt.close()


# =================================================================
# MAIN
# =================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze comprehensive attack test results'
    )
    parser.add_argument(
        'results_dir',
        nargs='?',
        default='results/comprehensive_attacks',
        help='Directory containing experiment results (default: results/comprehensive_attacks)'
    )
    
    args = parser.parse_args()
    results_dir = args.results_dir
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"\n❌ Results directory not found: {results_dir}")
        print("Please run test_all_attacks.sh first!")
        return
    
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE ATTACK RESULTS ANALYSIS")
    print("="*80)
    print(f"\n📁 Results directory: {results_dir}\n")
    
    # Load all experiments
    print("Loading experiment results...")
    experiments = extract_all_results(results_dir)
    
    if not experiments:
        print("❌ No experiments found!")
        print("Please run test_all_attacks.sh first!")
        return
    
    print(f"✓ Found {len(experiments)} experiments\n")
    
    # Run analyses
    analyze_accuracy_comparison(experiments)
    analyze_detection_metrics(experiments)
    create_summary_table(experiments)
    
    # Create visualizations
    plot_dir = os.path.join('results', 'analysis_plots')
    plot_accuracy_comparison(experiments, plot_dir)
    plot_detection_performance(experiments, plot_dir)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📊 Plots saved to: {plot_dir}")
    print(f"📁 Results directory: {results_dir}\n")


if __name__ == "__main__":
    main()