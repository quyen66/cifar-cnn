#!/usr/bin/env python3
"""
Analyze Detection Metrics (TP/FP/FN/TN)
Extract and visualize defense performance from experiments
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_model_metadata(model_dir="results/defense_test"):
    """Load metadata from saved model directory."""
    metadata_file = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return None


def extract_detection_metrics(results_dir):
    """Extract detection metrics from all experiments."""
    experiments = {}
    
    # Find all subdirectories
    for subdir in glob.glob(os.path.join(results_dir, "*")):
        if not os.path.isdir(subdir):
            continue
        
        exp_name = os.path.basename(subdir)
        
        # Find the latest model in this experiment
        model_dirs = glob.glob(os.path.join(subdir, "*"))
        if not model_dirs:
            continue
        
        # Get the most recent model
        latest_model = max(model_dirs, key=os.path.getctime)
        
        metadata = load_model_metadata(latest_model)
        if metadata:
            experiments[exp_name] = {
                'accuracy_history': metadata.get('accuracy_history', []),
                'best_accuracy': metadata.get('best_accuracy', 0),
                'config': metadata.get('config', {}),
                'defense_stats': metadata.get('defense_stats', {}),
                'algorithm': metadata.get('algorithm', 'Unknown')
            }
    
    return experiments


def print_detection_summary(experiments):
    """Print detailed detection metrics summary."""
    print("\n" + "="*80)
    print("🎯 DETECTION METRICS ANALYSIS")
    print("="*80 + "\n")
    
    # Group by attack type
    attack_groups = {
        'Byzantine': [],
        'Gaussian': [],
        'Label Flip': []
    }
    
    for exp_name, data in experiments.items():
        if 'byzantine' in exp_name:
            attack_groups['Byzantine'].append((exp_name, data))
        elif 'gaussian' in exp_name:
            attack_groups['Gaussian'].append((exp_name, data))
        elif 'labelflip' in exp_name:
            attack_groups['Label Flip'].append((exp_name, data))
    
    for attack_name, exp_list in attack_groups.items():
        if not exp_list:
            continue
        
        print(f"{'━'*80}")
        print(f"🎯 {attack_name} Attack (30%)")
        print(f"{'━'*80}")
        
        for exp_name, data in exp_list:
            is_defense = 'defense' in exp_name and 'nodefense' not in exp_name
            defense_stats = data.get('defense_stats', {})
            
            if not is_defense:
                # No defense - just show accuracy
                best_acc = data['best_accuracy']
                print(f"\n📊 NO DEFENSE:")
                print(f"   Final Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
                continue
            
            # With defense - show detection metrics
            print(f"\n📊 WITH DEFENSE:")
            
            # Get totals
            total_tp = defense_stats.get('total_tp', 0)
            total_fp = defense_stats.get('total_fp', 0)
            total_fn = defense_stats.get('total_fn', 0)
            total_tn = defense_stats.get('total_tn', 0)
            
            # Calculate overall metrics
            total_malicious_seen = total_tp + total_fn
            total_benign_seen = total_fp + total_tn
            
            detection_rate = (total_tp / total_malicious_seen * 100) if total_malicious_seen > 0 else 0
            fpr = (total_fp / total_benign_seen * 100) if total_benign_seen > 0 else 0
            precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0
            recall = detection_rate
            
            # F1 Score
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            print(f"\n   📈 Overall Metrics (Across All Rounds):")
            print(f"      Total Samples: {total_malicious_seen + total_benign_seen}")
            print(f"         Malicious: {total_malicious_seen}")
            print(f"         Benign: {total_benign_seen}")
            
            print(f"\n   📊 Confusion Matrix:")
            print(f"                        Predicted")
            print(f"                 Malicious    Benign")
            print(f"      Actual")
            print(f"      Malicious    {total_tp:5d}      {total_fn:5d}")
            print(f"      Benign       {total_fp:5d}      {total_tn:5d}")
            
            print(f"\n   ✅ Performance Metrics:")
            print(f"      Detection Rate (Recall): {detection_rate:.2f}%  ({total_tp}/{total_malicious_seen})")
            print(f"      False Positive Rate:     {fpr:.2f}%  ({total_fp}/{total_benign_seen})")
            print(f"      Precision:               {precision:.2f}%  ({total_tp}/{total_tp + total_fp})")
            print(f"      F1-Score:                {f1:.2f}%")
            
            # Final accuracy
            best_acc = data['best_accuracy']
            print(f"\n   🎯 Final Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
            
            # Ground truth info
            ground_truth = defense_stats.get('ground_truth_malicious', [])
            if ground_truth:
                print(f"\n   🎯 Ground Truth:")
                print(f"      Malicious Client IDs: {sorted(ground_truth)}")
            
            # Per-round breakdown (if available)
            detection_history = defense_stats.get('detection_history', [])
            if detection_history and len(detection_history) > 0:
                print(f"\n   📊 Round-by-Round Stats (Last 5 rounds):")
                print(f"      Round | TP | FP | FN | TN | Det.Rate | FPR")
                print(f"      ------|----|----|----|----|----------|-----")
                for round_data in detection_history[-5:]:
                    r = round_data['round']
                    tp = round_data['tp']
                    fp = round_data['fp']
                    fn = round_data['fn']
                    tn = round_data['tn']
                    dr = round_data.get('detection_rate', 0)
                    fpr_r = round_data.get('fpr', 0)
                    print(f"      {r:5d} | {tp:2d} | {fp:2d} | {fn:2d} | {tn:2d} |  {dr:6.1f}% | {fpr_r:5.1f}%")
        
        print()
    
    print("="*80 + "\n")


def plot_detection_metrics(experiments, save_dir='results'):
    """Plot detection metrics over rounds."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detection Metrics Analysis\n(Byzantine, Gaussian, Label Flip Attacks)', 
                 fontsize=16, fontweight='bold')
    
    attack_types = ['byzantine', 'gaussian', 'labelflip']
    attack_names = {
        'byzantine': 'Byzantine',
        'gaussian': 'Gaussian',
        'labelflip': 'Label Flip'
    }
    
    colors = {
        'byzantine': '#e74c3c',
        'gaussian': '#f39c12',
        'labelflip': '#3498db'
    }
    
    # Extract data for each attack type
    all_data = {}
    for attack_type in attack_types:
        for exp_name, data in experiments.items():
            if attack_type in exp_name and 'defense' in exp_name and 'nodefense' not in exp_name:
                defense_stats = data.get('defense_stats', {})
                detection_history = defense_stats.get('detection_history', [])
                
                if detection_history:
                    all_data[attack_type] = {
                        'rounds': [h['round'] for h in detection_history],
                        'tp': [h['tp'] for h in detection_history],
                        'fp': [h['fp'] for h in detection_history],
                        'fn': [h['fn'] for h in detection_history],
                        'tn': [h['tn'] for h in detection_history],
                        'detection_rate': [h.get('detection_rate', 0) for h in detection_history],
                        'fpr': [h.get('fpr', 0) for h in detection_history]
                    }
    
    # Plot 1: Detection Rate over rounds
    ax1 = axes[0, 0]
    for attack_type, data in all_data.items():
        ax1.plot(data['rounds'], data['detection_rate'], 
                label=attack_names[attack_type],
                color=colors[attack_type], linewidth=2.5, marker='o', markersize=5)
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Detection Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Detection Rate Over Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Plot 2: False Positive Rate over rounds
    ax2 = axes[0, 1]
    for attack_type, data in all_data.items():
        ax2.plot(data['rounds'], data['fpr'],
                label=attack_names[attack_type],
                color=colors[attack_type], linewidth=2.5, marker='s', markersize=5)
    ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('False Positive Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('False Positive Rate Over Time', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(30, max([max(d['fpr']) for d in all_data.values() if d['fpr']])+5)])
    
    # Plot 3: Cumulative TP/FP
    ax3 = axes[1, 0]
    for attack_type, data in all_data.items():
        cumulative_tp = np.cumsum(data['tp'])
        cumulative_fp = np.cumsum(data['fp'])
        ax3.plot(data['rounds'], cumulative_tp, 
                label=f'{attack_names[attack_type]} (TP)',
                color=colors[attack_type], linewidth=2.5, linestyle='-', marker='o', markersize=4)
        ax3.plot(data['rounds'], cumulative_fp,
                label=f'{attack_names[attack_type]} (FP)',
                color=colors[attack_type], linewidth=2.5, linestyle='--', marker='x', markersize=4)
    ax3.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Count', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative True Positives vs False Positives', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary bar chart
    ax4 = axes[1, 1]
    metrics_summary = {}
    for attack_type, data in all_data.items():
        total_tp = sum(data['tp'])
        total_fp = sum(data['fp'])
        total_fn = sum(data['fn'])
        total_tn = sum(data['tn'])
        
        det_rate = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0
        fpr_avg = (total_fp / (total_fp + total_tn) * 100) if (total_fp + total_tn) > 0 else 0
        
        metrics_summary[attack_type] = {
            'Detection Rate': det_rate,
            'FPR': fpr_avg
        }
    
    x = np.arange(len(attack_types))
    width = 0.35
    
    det_rates = [metrics_summary[at]['Detection Rate'] for at in attack_types]
    fprs = [metrics_summary[at]['FPR'] for at in attack_types]
    
    bars1 = ax4.bar(x - width/2, det_rates, width, label='Detection Rate', color='#27ae60', alpha=0.8)
    bars2 = ax4.bar(x + width/2, fprs, width, label='False Positive Rate', color='#e74c3c', alpha=0.8)
    
    ax4.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Overall Performance Summary', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([attack_names[at] for at in attack_types])
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'detection_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved detection metrics plot: {save_path}")
    plt.close()


def main():
    """Main analysis function."""
    results_dir = "results/defense_test"
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        print("Please run test_defense_realworld.sh first!")
        return
    
    print("\n🔍 Loading experiment results...")
    experiments = extract_detection_metrics(results_dir)
    
    if not experiments:
        print("❌ No experiments found!")
        print("Please run test_defense_realworld.sh first!")
        return
    
    # Filter only experiments with defense
    defense_experiments = {k: v for k, v in experiments.items() 
                          if v.get('defense_stats')}
    
    if not defense_experiments:
        print("⚠️  No experiments with defense found!")
        print("Make sure to run with enable-defense=true")
        return
    
    print(f"✓ Found {len(experiments)} experiments")
    print(f"✓ Found {len(defense_experiments)} experiments with defense metrics\n")
    
    # Print detailed summary
    print_detection_summary(experiments)
    
    # Create visualizations
    if defense_experiments:
        plot_detection_metrics(experiments, save_dir='results')
    
    print("\n" + "="*80)
    print("✅ DETECTION METRICS ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📊 Generated files:")
    print("   • results/detection_metrics.png")
    print("\n💡 Key Findings:")
    print("   • Check Detection Rate (should be >80% for Byzantine/Gaussian)")
    print("   • Check False Positive Rate (should be <10%)")
    print("   • Label Flip typically has lower detection (~20-40%)")
    print("\n📝 Next Steps:")
    print("   • Review confusion matrices above")
    print("   • Compare different attack types")
    print("   • Ready for Layer 2 Detection (Week 3) to improve Label Flip detection")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()