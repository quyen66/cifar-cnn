#!/usr/bin/env python3
"""
Hybrid H-Score Parameter Test Runner - FINAL VERSION
=====================================================

Features:
- Tests each config against ALL 11 attack scenarios
- LIVE output streaming (see progress in real-time)
- AUTO-SAVES individual log files per test
- Resume support (skip completed tests)
- CSV results with all metrics

Usage:
    python3 run_param_tests.py hybrid_h_weights_grid1_20260131.json
    python3 run_param_tests.py hybrid_h_weights_grid1_20260131.json --resume
"""

import json
import subprocess
import csv
import re
import time
import sys
import os
import argparse
import shutil
from pathlib import Path
from datetime import datetime


# ============================================================================
# ATTACK SCENARIOS TO TEST
# ============================================================================
ATTACK_SCENARIOS = [
    "slow_poison",
    "minsum",
    "alie",
    "on_off",
    "minmax",
    "backdoor",
    "label_flip",
    "random_noise",
    "none",
    "sign_flip",
    "gaussian_noise",
]


class HybridHScoreTestRunner:
    """Run parameter tests with attack scenarios and resume capability."""
    
    def __init__(self, suite_file: str, resume: bool = False, log_dir: str = "tuning_logs"):
        """
        Initialize test runner.
        
        Args:
            suite_file: Path to JSON suite file
            resume: If True, skip completed tests
            log_dir: Directory to save individual test logs
        """
        self.suite_file = suite_file
        self.resume = resume
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Load suite
        with open(suite_file, 'r') as f:
            self.suite = json.load(f)
        
        # Setup paths
        self.backup_config = "pyproject.toml.backup"
        self.active_config = "pyproject.toml"
        self.results_file = "hybrid_h_weights_results.csv"
        
        # Get configs
        self.configs = self.suite['suite']['h_weights']
        self.total_configs = len(self.configs)
        self.total_attacks = len(ATTACK_SCENARIOS)
        self.total_tests = self.total_configs * self.total_attacks
        
        print(f"📦 Loaded suite: {suite_file}")
        print(f"   Total configs: {self.total_configs}")
        print(f"   Attack scenarios: {self.total_attacks}")
        print(f"   Total tests: {self.total_tests} (configs × attacks)")
        print(f"   Resume mode: {resume}")
        print(f"   Log directory: {log_dir}/")
        print()
    
    def get_completed_tests(self):
        """Get set of completed (config_id, attack) pairs."""
        completed = set()
        
        if not os.path.exists(self.results_file):
            return completed
        
        try:
            with open(self.results_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'config_id' in row and 'attack_type' in row:
                        completed.add((int(row['config_id']), row['attack_type']))
        except Exception as e:
            print(f"⚠️  Warning: Could not read results file: {e}")
        
        return completed
    
    def backup_original_config(self):
        """Backup original pyproject.toml."""
        if not os.path.exists(self.backup_config):
            print(f"📦 Creating backup: {self.backup_config}")
            shutil.copy(self.active_config, self.backup_config)
        else:
            print(f"ℹ️  Using existing backup: {self.backup_config}")
    
    def restore_original_config(self):
        """Restore original pyproject.toml."""
        if os.path.exists(self.backup_config):
            print(f"🔄 Restoring original configuration...")
            shutil.copy(self.backup_config, self.active_config)
    
    def update_config(self, config: dict, attack_type: str):
        """Update pyproject.toml with new H weights and attack type."""
        # Read from backup (clean version)
        with open(self.backup_config, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        in_noniid_section = False
        
        for line in lines:
            stripped = line.strip()
            
            # Update attack-type
            if stripped.startswith('attack-type ='):
                new_lines.append(f'attack-type = "{attack_type}"\n')
                continue
            
            # Update save-dir to avoid overwriting models
            if stripped.startswith('save-dir ='):
                cfg_id = config['config_id']
                new_lines.append(
                    f'save-dir = "saved_models/tuning/cfg{cfg_id:03d}_{attack_type}"\n'
                )
                continue
            
            # Detect noniid section
            if stripped.startswith('[tool.flwr.app.config.defense.noniid]'):
                in_noniid_section = True
                new_lines.append(line)
                continue
            
            # Exit section when new section starts
            if in_noniid_section and stripped.startswith('['):
                in_noniid_section = False
            
            # Replace H weights in noniid section
            if in_noniid_section:
                if stripped.startswith('weight-h-grad'):
                    new_lines.append(f'weight-h-grad = {config["weight_h_grad"]}\n')
                elif stripped.startswith('weight-h-loss'):
                    new_lines.append(f'weight-h-loss = {config["weight_h_loss"]}\n')
                elif stripped.startswith('weight-h-acc'):
                    new_lines.append(f'weight-h-acc = {config["weight_h_acc"]}\n')
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Write updated config
        with open(self.active_config, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    
    def get_log_filename(self, config_id: int, attack_type: str) -> str:
        """Generate log filename for this test."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cfg{config_id:03d}_{attack_type}_{timestamp}.log"
        return os.path.join(self.log_dir, filename)
    
    def parse_metrics(self, output: str) -> dict:
        """Parse metrics from experiment output."""
        metrics = {
            'final_accuracy': 0.0,
            'avg_h_score': 0.0,
            'final_mode': 'UNKNOWN',
            'detection_rate': 0.0,
            'false_positive_rate': 0.0
        }
        
        # Extract final accuracy (last one)
        acc_matches = re.findall(r"Accuracy:\s+([0-9.]+)", output)
        if acc_matches:
            metrics['final_accuracy'] = float(acc_matches[-1])
        
        # Extract H-scores and average
        h_matches = re.findall(r"H_final=([0-9.]+)", output)
        if h_matches:
            h_scores = [float(h) for h in h_matches]
            metrics['avg_h_score'] = sum(h_scores) / len(h_scores)
        
        # Extract final mode
        mode_matches = re.findall(r"Mode['\"]?:\s*['\"]?([A-Z]+)['\"]?", output)
        if mode_matches:
            metrics['final_mode'] = mode_matches[-1]
        
        # Extract detection metrics (if available)
        dr_matches = re.findall(r"Detection Rate:\s*([0-9.]+)", output)
        if dr_matches:
            metrics['detection_rate'] = float(dr_matches[-1])
        
        fpr_matches = re.findall(r"False Positive Rate:\s*([0-9.]+)", output)
        if fpr_matches:
            metrics['false_positive_rate'] = float(fpr_matches[-1])
        
        return metrics
    
    def run_single_test(self, config: dict, attack_type: str, test_num: int) -> dict:
        """Run single test configuration with specific attack."""
        
        config_id = config['config_id']
        grad = config['weight_h_grad']
        loss = config['weight_h_loss']
        acc = config['weight_h_acc']
        profile = config['profile']
        
        # Get log filename
        log_file = self.get_log_filename(config_id, attack_type)
        
        print(f"\n{'='*80}")
        print(f"🧪 Test {test_num}/{self.total_tests}")
        print(f"   Config #{config_id}/{self.total_configs} | Attack: {attack_type}")
        print(f"   Profile: {profile}")
        print(f"   Weights: Grad={grad:.2f}, Loss={loss:.2f}, Acc={acc:.2f}")
        print(f"   Log: {log_file}")
        print(f"{'='*80}\n")
        
        # Update configuration
        self.update_config(config, attack_type)
        
        # Run experiment WITH LIVE OUTPUT + LOG SAVING
        start_time = time.time()
        status = "SUCCESS"
        output_buffer = []
        
        try:
            # Open log file for writing
            with open(log_file, 'w', encoding='utf-8') as log_f:
                # Write header to log
                log_f.write(f"{'='*80}\n")
                log_f.write(f"Test: Config #{config_id} × {attack_type}\n")
                log_f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                log_f.write(f"Profile: {profile}\n")
                log_f.write(f"Weights: Grad={grad:.2f}, Loss={loss:.2f}, Acc={acc:.2f}\n")
                log_f.write(f"{'='*80}\n\n")
                log_f.flush()
                
                # Run with real-time output streaming
                process = subprocess.Popen(
                    ["flwr", "run", "."],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1  # Line buffered
                )
                
                # Stream output line by line
                for line in process.stdout:
                    # Print to console (LIVE)
                    print(line, end='', flush=True)
                    # Save to log file
                    log_f.write(line)
                    log_f.flush()
                    # Save for parsing
                    output_buffer.append(line)
                
                # Wait for completion
                process.wait(timeout=1800)  # 30 min timeout
                
                duration = time.time() - start_time
                output = ''.join(output_buffer)
                
                if process.returncode != 0:
                    print(f"\n⚠️  Process returned error code: {process.returncode}")
                    status = "FAILED"
                
                # Parse metrics
                metrics = self.parse_metrics(output)
                
                # Write summary to log file
                log_f.write(f"\n{'='*80}\n")
                log_f.write(f"TEST SUMMARY\n")
                log_f.write(f"{'='*80}\n")
                log_f.write(f"Status: {status}\n")
                log_f.write(f"Duration: {duration:.2f}s\n")
                log_f.write(f"Final Accuracy: {metrics['final_accuracy']:.4f}\n")
                log_f.write(f"Avg H-score: {metrics['avg_h_score']:.4f}\n")
                log_f.write(f"Final Mode: {metrics['final_mode']}\n")
                log_f.write(f"Detection Rate: {metrics['detection_rate']:.4f}\n")
                log_f.write(f"False Positive Rate: {metrics['false_positive_rate']:.4f}\n")
                log_f.write(f"{'='*80}\n")
            
            # Log summary to console
            print(f"\n{'='*80}")
            icon = "✅" if status == "SUCCESS" else "❌"
            print(f"{icon} Test {test_num} Complete:")
            print(f"   Accuracy: {metrics['final_accuracy']:.4f}")
            print(f"   Avg H-score: {metrics['avg_h_score']:.4f}")
            print(f"   Mode: {metrics['final_mode']}")
            print(f"   DR: {metrics['detection_rate']:.4f}, FPR: {metrics['false_positive_rate']:.4f}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   💾 Log saved: {log_file}")
            print(f"{'='*80}\n")
            
            # Return result
            return {
                'config_id': config_id,
                'weight_h_grad': grad,
                'weight_h_loss': loss,
                'weight_h_acc': acc,
                'profile': profile,
                'attack_type': attack_type,
                'final_accuracy': metrics['final_accuracy'],
                'avg_h_score': metrics['avg_h_score'],
                'final_mode': metrics['final_mode'],
                'detection_rate': metrics['detection_rate'],
                'false_positive_rate': metrics['false_positive_rate'],
                'duration_sec': round(duration, 2),
                'status': status,
                'log_file': log_file
            }
            
        except subprocess.TimeoutExpired:
            if 'process' in locals():
                process.kill()
            print(f"\n❌ TIMEOUT! Execution took too long.")
            
            # Save timeout to log
            with open(log_file, 'a', encoding='utf-8') as log_f:
                log_f.write(f"\n{'='*80}\n")
                log_f.write(f"TIMEOUT after 1800 seconds\n")
                log_f.write(f"{'='*80}\n")
            
            return {
                'config_id': config_id,
                'weight_h_grad': grad,
                'weight_h_loss': loss,
                'weight_h_acc': acc,
                'profile': profile,
                'attack_type': attack_type,
                'status': 'TIMEOUT',
                'duration_sec': 1800,
                'log_file': log_file
            }
            
        except Exception as e:
            print(f"\n❌ EXCEPTION: {e}")
            
            # Save error to log
            with open(log_file, 'a', encoding='utf-8') as log_f:
                log_f.write(f"\n{'='*80}\n")
                log_f.write(f"EXCEPTION: {str(e)}\n")
                log_f.write(f"{'='*80}\n")
            
            return {
                'config_id': config_id,
                'weight_h_grad': grad,
                'weight_h_loss': loss,
                'weight_h_acc': acc,
                'profile': profile,
                'attack_type': attack_type,
                'status': f'ERROR: {str(e)}',
                'log_file': log_file
            }
    
    def run_all_tests(self):
        """Run all tests (configs × attacks) with resume support."""
        
        # Backup original config
        self.backup_original_config()
        
        # Check resume
        completed = set()
        if self.resume:
            completed = self.get_completed_tests()
            print(f"🔄 Resume mode: Found {len(completed)} completed tests")
            print()
        
        # Open results file
        file_exists = os.path.exists(self.results_file)
        
        try:
            with open(self.results_file, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'config_id', 'weight_h_grad', 'weight_h_loss', 'weight_h_acc',
                    'profile', 'attack_type', 'final_accuracy', 'avg_h_score', 
                    'final_mode', 'detection_rate', 'false_positive_rate', 
                    'duration_sec', 'status', 'log_file'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header if new file
                if not file_exists:
                    writer.writeheader()
                
                # Run tests
                tested = 0
                skipped = 0
                test_num = 0
                
                # Loop through configs and attacks
                for config in self.configs:
                    config_id = config['config_id']
                    
                    for attack_type in ATTACK_SCENARIOS:
                        test_num += 1
                        
                        # Check if already done
                        if self.resume and (config_id, attack_type) in completed:
                            skipped += 1
                            print(f"⏭️  Skipping Test {test_num}: "
                                  f"Config #{config_id} × {attack_type} (already done)")
                            continue
                        
                        # Run test
                        result = self.run_single_test(config, attack_type, test_num)
                        
                        # Write result
                        writer.writerow(result)
                        csvfile.flush()
                        
                        tested += 1
                        
                        # Progress
                        total_done = tested + skipped
                        pct = (total_done / self.total_tests) * 100
                        print(f"\n📊 Overall Progress: {total_done}/{self.total_tests} ({pct:.1f}%)")
                        print(f"   Tested: {tested}, Skipped: {skipped}")
                        remaining = self.total_tests - total_done
                        est_hours = remaining * 40 / 60
                        print(f"   Remaining: {remaining} tests (~{est_hours:.1f} hours)\n")
                
                print(f"\n✅ All tests complete!")
                print(f"   Total tested: {tested}")
                print(f"   Total skipped: {skipped}")
                print(f"   Logs saved to: {self.log_dir}/")
        
        except KeyboardInterrupt:
            print(f"\n🛑 Tests interrupted by user")
            print(f"   Results saved to: {self.results_file}")
            print(f"   Logs saved to: {self.log_dir}/")
            print(f"   Resume with: --resume flag")
        
        finally:
            # Restore original config
            self.restore_original_config()
            print(f"\n📊 Results saved to: {self.results_file}")
            print(f"📁 Logs saved to: {self.log_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Run hybrid H-score parameter tests with attack scenarios'
    )
    parser.add_argument(
        'suite_file',
        help='JSON suite file from param_generator.py'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous run (skip completed tests)'
    )
    parser.add_argument(
        '--log-dir',
        default='tuning_logs',
        help='Directory to save individual test logs (default: tuning_logs)'
    )
    
    args = parser.parse_args()
    
    # Validate suite file
    if not os.path.exists(args.suite_file):
        print(f"❌ Suite file not found: {args.suite_file}")
        return 1
    
    print("=" * 80)
    print("HYBRID H-SCORE PARAMETER TEST RUNNER - FINAL VERSION")
    print("Features: Live Output + Auto Log Saving + Resume Support")
    print("=" * 80)
    print()
    
    print(f"📋 Attack Scenarios ({len(ATTACK_SCENARIOS)} total):")
    for i, attack in enumerate(ATTACK_SCENARIOS, 1):
        print(f"   {i:2d}. {attack}")
    print()
    
    # Run tests
    runner = HybridHScoreTestRunner(
        suite_file=args.suite_file,
        resume=args.resume,
        log_dir=args.log_dir
    )
    
    # Estimate time
    print(f"⏱️  Estimated Time:")
    print(f"   Per test: ~40 minutes")
    print(f"   Total: ~{runner.total_tests * 40 / 60:.1f} hours "
          f"({runner.total_tests * 40 / 60 / 24:.1f} days)")
    print()
    
    input("Press Enter to start tests (or Ctrl+C to cancel)...")
    print()
    
    runner.run_all_tests()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())