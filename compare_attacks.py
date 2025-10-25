"""So sánh chi tiết các attacks từ saved models."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cifar_cnn.model_manager import ModelManager


def parse_model_info(model):
    meta = model['metadata']
    config = meta.get('config', {})
    return {
        'name': model['name'],
        'attack_type': config.get('attack_type', 'none'),
        'attack_ratio': config.get('attack_ratio', 0.0) * 100,
        'partition_type': config.get('partition_type', 'unknown'),
        'num_clients': config.get('num_clients', 'unknown'),
        'best_accuracy': meta.get('best_accuracy', meta.get('final_accuracy', 0)),
        'final_accuracy': meta.get('final_accuracy', meta.get('best_accuracy', 0)),
        'total_rounds': meta.get('total_rounds', meta.get('current_round', 0)),
        'accuracy_history': meta.get('accuracy_history', []),
        'loss_history': meta.get('loss_history', []),
    }


def load_all_models():
    manager = ModelManager()
    models = manager.list_models()
    if not models:
        print("❌ Không có model nào!")
        return []
    
    parsed_models = []
    for model in models:
        try:
            info = parse_model_info(model)
            parsed_models.append(info)
        except Exception as e:
            print(f"⚠️  Skipping {model['name']}: {e}")
    return parsed_models


def group_by_attack(models):
    groups = {'none': [], 'label_flip': {}, 'byzantine': {}, 'gaussian': {}}
    
    for model in models:
        attack_type = model['attack_type']
        attack_ratio = model['attack_ratio']
        
        if attack_type == 'none':
            groups['none'].append(model)
        elif attack_type in groups:
            if attack_ratio not in groups[attack_type]:
                groups[attack_type][attack_ratio] = []
            groups[attack_type][attack_ratio].append(model)
    
    return groups


def plot_all_training_curves(models, save_path='attack_training_curves_unified.png'):
    groups = group_by_attack(models)
    
    colors = {'none': '#2ecc71', 'label_flip': '#e74c3c', 'byzantine': '#f39c12', 'gaussian': '#3498db'}
    linestyles = {0: ('-', 'solid'), 10: ('--', 'dashed'), 20: ('-.', 'dashdot'), 30: (':', 'dotted')}
    markers = {0: 'o', 10: 's', 20: '^', 30: 'D'}
    attack_labels = {'none': 'No Attack', 'label_flip': 'Label Flipping', 
                     'byzantine': 'Byzantine', 'gaussian': 'Gaussian Noise'}
    
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_axes([0.08, 0.1, 0.65, 0.8])
    has_data = False
    
    if groups['none']:
        for model in groups['none']:
            if model['accuracy_history']:
                has_data = True
                rounds = [r for r, _ in model['accuracy_history']]
                accs = [a for _, a in model['accuracy_history']]
                ax.plot(rounds, accs, color=colors['none'], linestyle=linestyles[0][0],
                       marker=markers[0], linewidth=2.5, markersize=6,
                       markevery=max(1, len(rounds)//10), label=attack_labels["none"], alpha=0.9)
    
    for attack_type in ['label_flip', 'byzantine', 'gaussian']:
        if groups[attack_type]:
            for ratio in sorted(groups[attack_type].keys()):
                for model in groups[attack_type][ratio]:
                    if model['accuracy_history']:
                        has_data = True
                        rounds = [r for r, _ in model['accuracy_history']]
                        accs = [a for _, a in model['accuracy_history']]
                        ls = linestyles.get(int(ratio), linestyles[10])[0]
                        marker = markers.get(int(ratio), markers[10])
                        label = f'{attack_labels[attack_type]} {int(ratio)}%'
                        ax.plot(rounds, accs, color=colors[attack_type], linestyle=ls,
                               marker=marker, linewidth=2, markersize=5,
                               markevery=max(1, len(rounds)//10), label=label, alpha=0.8)
    
    if not has_data:
        print("⚠️  Không có data để vẽ!")
        plt.close()
        return
    
    ax.set_xlabel('Round', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Training Curves: Attack Comparison\n(Colors = Attack Types, Line Styles = Attack Ratios)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])
    
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    color_legend_elements = [
        Patch(facecolor=colors['none'], edgecolor='black', label='No Attack'),
        Patch(facecolor=colors['label_flip'], edgecolor='black', label='Label Flipping'),
        Patch(facecolor=colors['byzantine'], edgecolor='black', label='Byzantine'),
        Patch(facecolor=colors['gaussian'], edgecolor='black', label='Gaussian Noise'),
    ]
    
    legend1_ax = fig.add_axes([0.75, 0.55, 0.2, 0.3])
    legend1_ax.axis('off')
    legend1_ax.legend(handles=color_legend_elements, loc='upper left', fontsize=11, 
                     title='Attack Types\n(Colors)', title_fontsize=12,
                     framealpha=1.0, edgecolor='black', fancybox=True, shadow=True)
    
    linestyle_legend_elements = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5, 
               marker='o', markersize=7, label='0% (Baseline)'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2, 
               marker='s', markersize=6, label='10% Attack'),
        Line2D([0], [0], color='gray', linestyle='-.', linewidth=2, 
               marker='^', markersize=6, label='20% Attack'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=2, 
               marker='D', markersize=6, label='30% Attack'),
    ]
    
    legend2_ax = fig.add_axes([0.75, 0.15, 0.2, 0.3])
    legend2_ax.axis('off')
    legend2_ax.legend(handles=linestyle_legend_elements, loc='upper left', fontsize=11, 
                     title='Attack Ratios\n(Line Styles)', title_fontsize=12,
                     framealpha=1.0, edgecolor='black', fancybox=True, shadow=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.show()


def plot_accuracy_matrix(models, save_path='attack_comparison_matrix.png'):
    groups = group_by_attack(models)
    attack_types = ['label_flip', 'byzantine', 'gaussian']
    attack_labels = ['Label\nFlipping', 'Byzantine', 'Gaussian\nNoise']
    
    all_ratios = set()
    for attack_type in attack_types:
        all_ratios.update(groups[attack_type].keys())
    
    if not all_ratios:
        print("⚠️  Không có data!")
        return
    
    ratios = sorted(all_ratios)
    matrix = np.zeros((len(attack_types), len(ratios)))
    
    for i, attack_type in enumerate(attack_types):
        for j, ratio in enumerate(ratios):
            if ratio in groups[attack_type]:
                avg_acc = np.mean([m['best_accuracy'] for m in groups[attack_type][ratio]])
                matrix[i, j] = avg_acc
            else:
                matrix[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline_acc = np.mean([m['best_accuracy'] for m in groups['none']]) if groups['none'] else 0
    
    sns.heatmap(matrix, annot=True, fmt='.4f', cmap='RdYlGn',
                xticklabels=[f'{r:.0f}%' for r in ratios], yticklabels=attack_labels,
                cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1, ax=ax,
                linewidths=2, linecolor='white')
    
    ax.set_xlabel('Attack Ratio', fontsize=13, fontweight='bold')
    ax.set_ylabel('Attack Type', fontsize=13, fontweight='bold')
    title = 'Attack Impact Matrix\n'
    if baseline_acc > 0:
        title += f'(Baseline: {baseline_acc:.4f})'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.show()


def generate_comparison_table(models):
    groups = group_by_attack(models)
    
    print(f"\n{'='*100}")
    print("ATTACK COMPARISON TABLE")
    print(f"{'='*100}\n")
    
    if groups['none']:
        baseline_acc = np.mean([m['best_accuracy'] for m in groups['none']])
        print(f"📊 BASELINE: {baseline_acc:.4f} ({len(groups['none'])} models)\n")
    else:
        baseline_acc = 0
    
    print(f"{'Attack Type':<20} {'Ratio':<10} {'Accuracy':<12} {'Degradation':<15} {'Models':<10}")
    print("-" * 100)
    
    for attack_type in ['label_flip', 'byzantine', 'gaussian']:
        attack_name = {'label_flip': 'Label Flipping', 'byzantine': 'Byzantine', 
                      'gaussian': 'Gaussian Noise'}[attack_type]
        
        if groups[attack_type]:
            for ratio in sorted(groups[attack_type].keys()):
                avg_acc = np.mean([m['best_accuracy'] for m in groups[attack_type][ratio]])
                deg_str = f"-{(baseline_acc - avg_acc) / baseline_acc * 100:.2f}%" if baseline_acc > 0 else "N/A"
                print(f"{attack_name:<20} {ratio:.0f}%{'':<6} {avg_acc:.4f}{'':<6} {deg_str:<15} {len(groups[attack_type][ratio]):<10}")
    
    print("-" * 100 + "\n")


def main():
    print("\n🔬 ATTACK COMPARISON ANALYSIS")
    print("="*100)
    
    print("\n📂 Loading models...")
    models = load_all_models()
    
    if not models:
        return
    
    print(f"✅ Loaded {len(models)} models\n")
    
    generate_comparison_table(models)
    
    print("📊 Generating visualizations...\n")
    
    try:
        plot_all_training_curves(models)
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    try:
        plot_accuracy_matrix(models)
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    print("\n✅ Complete!")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()