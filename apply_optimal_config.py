"""
Auto-Update Layer 1 DBSCAN Parameters
======================================

Script này tự động update optimal parameters vào file layer1_dbscan.py
sau khi chạy hyperparameter tuning.

Usage:
    python apply_optimal_config.py [json_file]
    
    # Hoặc tự động tìm latest JSON
    python apply_optimal_config.py
"""

import json
import re
from pathlib import Path
from typing import Dict
from datetime import datetime
import shutil


def find_latest_results() -> Path:
    """Tìm file JSON results mới nhất."""
    json_files = sorted(Path('.').glob('layer1_*.json'), 
                       key=lambda x: x.stat().st_mtime, 
                       reverse=True)
    
    if not json_files:
        raise FileNotFoundError(
            "No layer1_*.json files found. "
            "Run test_layer1_hyperparam_quick.py first!"
        )
    
    return json_files[0]


def load_optimal_params(json_file: Path) -> Dict:
    """Load optimal parameters từ JSON results."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    if not results:
        raise ValueError("No results found in JSON file")
    
    # Best config is first in sorted list
    best = results[0]
    params = best['params']
    
    # Default values (nếu params không có trong test results)
    defaults = {
        'pca_dims': 20,
        'dbscan_min_samples': 3,
        'dbscan_eps_multiplier': 0.5,
        'mad_k_normal': 4.0,
        'mad_k_warmup': 6.0,
        'voting_threshold_normal': 2,
        'voting_threshold_warmup': 3,
        'warmup_rounds': 10
    }
    
    # Merge với defaults
    full_params = {**defaults, **params}
    
    # Get performance metrics
    if 'average' in best:
        metrics = best['average']
    elif 'avg_f1' in best:
        metrics = {
            'f1': best['avg_f1'],
            'detection_rate': best['avg_detection'],
            'fpr': best['avg_fpr']
        }
    else:
        metrics = {}
    
    return full_params, metrics


def backup_file(filepath: Path):
    """Backup file trước khi modify."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.with_suffix(f'.backup_{timestamp}')
    shutil.copy2(filepath, backup_path)
    print(f"📦 Backed up to: {backup_path.name}")
    return backup_path


def update_layer1_file(filepath: Path, params: Dict, metrics: Dict = None):
    """
    Update parameters trực tiếp trong layer1_dbscan.py file.
    
    Updates:
    1. __init__ default values
    2. Hardcoded k values in magnitude filter
    3. Hardcoded eps multiplier in DBSCAN
    4. Hardcoded voting thresholds
    """
    
    # Read current content
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    print(f"\n{'='*70}")
    print(f"UPDATING FILE: {filepath}")
    print(f"{'='*70}\n")
    
    # Extract params with defaults
    pca_dims = params.get('pca_dims', 20)
    dbscan_min_samples = params.get('dbscan_min_samples', 3)
    warmup_rounds = params.get('warmup_rounds', 10)
    mad_k_normal = params.get('mad_k_normal', 4.0)
    mad_k_warmup = params.get('mad_k_warmup', 6.0)
    
    # For comprehensive test, có thêm các params này
    eps_multiplier = params.get('dbscan_eps_multiplier', 0.5)
    voting_threshold_normal = params.get('voting_threshold_normal', 2)
    voting_threshold_warmup = params.get('voting_threshold_warmup', 3)
    
    changes = []
    
    # 1. Update __init__ default values
    print("📝 Updating __init__ default parameters...")
    
    # Update pca_dims default
    pattern = r'pca_dims: int = \d+'
    replacement = f'pca_dims: int = {pca_dims}'
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        changes.append(f"  ✓ pca_dims: {pca_dims}")
    
    # Update dbscan_min_samples default
    pattern = r'dbscan_min_samples: int = \d+'
    replacement = f'dbscan_min_samples: int = {dbscan_min_samples}'
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        changes.append(f"  ✓ dbscan_min_samples: {dbscan_min_samples}")
    
    # Update warmup_rounds default
    pattern = r'warmup_rounds: int = \d+'
    replacement = f'warmup_rounds: int = {warmup_rounds}'
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        changes.append(f"  ✓ warmup_rounds: {warmup_rounds}")
    
    # 2. Update hardcoded k values in magnitude filter
    print("\n📝 Updating magnitude filter k values...")
    
    # Find the line: k = 6.0 if is_warmup else 4.0
    pattern = r'k = [\d.]+\s+if is_warmup else\s+[\d.]+'
    replacement = f'k = {mad_k_warmup} if is_warmup else {mad_k_normal}'
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        changes.append(f"  ✓ MAD k_warmup: {mad_k_warmup}")
        changes.append(f"  ✓ MAD k_normal: {mad_k_normal}")
    
    # 3. Update hardcoded eps multiplier in DBSCAN
    print("\n📝 Updating DBSCAN eps multiplier...")
    
    # Find: eps = 0.5 * median_dist
    pattern = r'eps = [\d.]+\s*\*\s*median_dist'
    replacement = f'eps = {eps_multiplier} * median_dist'
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        changes.append(f"  ✓ eps_multiplier: {eps_multiplier}")
    
    # 4. Update hardcoded voting thresholds
    print("\n📝 Updating voting thresholds...")
    
    # Find: threshold = 3 if is_warmup else 2
    pattern = r'threshold = \d+\s+if is_warmup else\s+\d+'
    replacement = f'threshold = {voting_threshold_warmup} if is_warmup else {voting_threshold_normal}'
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        changes.append(f"  ✓ voting_threshold_warmup: {voting_threshold_warmup}")
        changes.append(f"  ✓ voting_threshold_normal: {voting_threshold_normal}")
    
    # Check if anything changed
    if content == original_content:
        print("\n⚠️  No changes made - patterns not found or values already optimal")
        return False
    
    # Write updated content
    with open(filepath, 'w') as f:
        f.write(content)
    
    # Print summary
    print(f"\n{'='*70}")
    print("✅ UPDATE COMPLETE!")
    print(f"{'='*70}\n")
    
    print("📊 Changes Applied:")
    for change in changes:
        print(change)
    
    if metrics:
        print(f"\n🎯 Expected Performance:")
        print(f"  • Detection Rate: {metrics.get('detection_rate', 0):.1%}")
        print(f"  • False Positive Rate: {metrics.get('fpr', 0):.1%}")
        print(f"  • F1 Score: {metrics.get('f1', 0):.3f}")
    
    print(f"\n💡 Tip: Review the changes in {filepath.name}")
    print(f"    Compare with backup if needed")
    
    return True


def generate_summary(params: Dict, metrics: Dict, output_file: str = "applied_config_summary.txt"):
    """Generate summary of applied configuration."""
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LAYER 1 OPTIMAL CONFIGURATION - APPLIED\n")
        f.write("="*70 + "\n\n")
        f.write(f"Applied: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PARAMETERS:\n")
        f.write("-"*70 + "\n")
        for key, value in sorted(params.items()):
            f.write(f"  {key}: {value}\n")
        
        if metrics:
            f.write("\nEXPECTED PERFORMANCE:\n")
            f.write("-"*70 + "\n")
            f.write(f"  Detection Rate: {metrics.get('detection_rate', 0):.1%}\n")
            f.write(f"  False Positive Rate: {metrics.get('fpr', 0):.1%}\n")
            f.write(f"  F1 Score: {metrics.get('f1', 0):.3f}\n")
        
        f.write("\nNEXT STEPS:\n")
        f.write("-"*70 + "\n")
        f.write("  1. Review changes in layer1_dbscan.py\n")
        f.write("  2. Run tests to verify\n")
        f.write("  3. Run full FL experiments\n")
        f.write("  4. Monitor performance\n")
    
    print(f"\n📄 Summary saved to: {output_file}")


def main():
    """Main function."""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "APPLY OPTIMAL LAYER 1 CONFIG" + " "*25 + "║")
    print("║" + " "*20 + "Auto-Update Script" + " "*30 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    try:
        # Find latest results
        print("🔍 Finding latest tuning results...")
        json_file = find_latest_results()
        print(f"   Found: {json_file.name}\n")
        
        # Load optimal params
        print("📂 Loading optimal parameters...")
        params, metrics = load_optimal_params(json_file)
        print(f"   ✓ Loaded best configuration (F1: {metrics.get('f1', 0):.3f})\n")
        
        # Find layer1_dbscan.py file
        print("🔍 Finding layer1_dbscan.py...")
        
        # Try multiple possible locations
        possible_paths = [
            Path('/mnt/project/cifar_cnn/defense/layer1_dbscan.py'),
            Path('cifar_cnn/defense/layer1_dbscan.py'),
            Path('../cifar_cnn/defense/layer1_dbscan.py'),
        ]
        
        layer1_file = None
        for path in possible_paths:
            if path.exists():
                layer1_file = path
                break
        
        if not layer1_file:
            print("❌ Could not find layer1_dbscan.py")
            print("   Tried locations:")
            for path in possible_paths:
                print(f"     - {path}")
            return
        
        print(f"   Found: {layer1_file}\n")
        
        # Confirm before updating
        print("⚠️  About to update parameters in layer1_dbscan.py")
        print("\n   Optimal Parameters:")
        for key, value in sorted(params.items()):
            print(f"     {key}: {value}")
        
        if metrics:
            print(f"\n   Expected Performance:")
            print(f"     Detection: {metrics.get('detection_rate', 0):.1%}")
            print(f"     FPR: {metrics.get('fpr', 0):.1%}")
            print(f"     F1: {metrics.get('f1', 0):.3f}")
        
        response = input("\n   Continue? [Y/n]: ").strip().lower()
        if response and response != 'y':
            print("\n❌ Aborted by user")
            return
        
        print()
        
        # Backup original file
        print("📦 Creating backup...")
        backup_path = backup_file(layer1_file)
        print()
        
        # Update file
        success = update_layer1_file(layer1_file, params, metrics)
        
        if success:
            # Generate summary
            generate_summary(params, metrics)
            
            print("\n" + "="*70)
            print("🎉 SUCCESS!")
            print("="*70)
            print("\n✅ Parameters updated in layer1_dbscan.py")
            print(f"✅ Backup saved: {backup_path.name}")
            print("✅ Summary saved: applied_config_summary.txt")
            
            print("\n📝 NEXT STEPS:")
            print("   1. Review changes: diff layer1_dbscan.py " + backup_path.name)
            print("   2. Run tests: python test_layer1.py")
            print("   3. Run FL: flwr run . --config attack-type=byzantine enable-defense=true")
            
        else:
            print("\n⚠️  No updates performed")
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()