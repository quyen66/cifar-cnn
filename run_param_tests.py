"""
Automatic Parameter Test Runner
================================

Đọc param suite JSON, generate shell commands, execute tests.

Flow:
1. Load param suite JSON
2. For each layer and config:
   - Merge config với default
   - Generate flwr run command
   - Execute và track results
3. Save all results to JSON

Usage:
    python run_param_tests.py param_suite_quick.json
"""

import json
import subprocess
from subprocess import Popen, PIPE
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import time


class ParamTestRunner:
    """Tự động chạy tests với các param combinations."""
    
    def __init__(self, suite_file: str, enable_model_saving: bool = False, resume_from: str = None):
        """
        Args:
            suite_file: Path to param suite JSON
            enable_model_saving: Save models during tests (default: False for speed)
            resume_from: Path to previous results JSON to resume from
        """
        self.suite_file = Path(suite_file)
        self.enable_model_saving = enable_model_saving
        self.resume_from = resume_from
        
        # Load suite
        with open(self.suite_file, 'r') as f:
            data = json.load(f)
        
        self.suite = data['suite']
        self.default_config = data['default_config']
        
        # Test configuration
        self.num_clients = 40
        self.num_rounds = 25  # Reduced from 30 for faster testing
        
        # Attack scenario (FIXED để focus vào param tuning)
        self.attack_type = "byzantine"
        self.attack_ratio = 0.3
        self.byzantine_type = "sign_flip"
        self.byzantine_scale = 2.0
        
        # Results
        self.results = []
        self.failed_tests = []
        self.completed_tests = set()  # Track completed test IDs
        
        # Load previous results if resuming
        if self.resume_from:
            self._load_previous_results()
    
    def generate_run_config(self, layer: str, config: Dict) -> str:
        """
        Generate --run-config string cho flwr run.
        
        Args:
            layer: Layer name
            config: Param config cho layer đó
        
        Returns:
            run-config string (with proper quotes)
        """
        # Merge với default
        full_config = self.default_config.copy()
        full_config.update(config)
        
        # Base config
        parts = [
            f'num-clients={self.num_clients}',
            f'num-server-rounds={self.num_rounds}',
            f'partition-type="iid"',
            f'attack-type="{self.attack_type}"',
            f'attack-ratio={self.attack_ratio}',
            f'byzantine-type="{self.byzantine_type}"',
            f'byzantine-scale={self.byzantine_scale}',
            'enable-defense=true',
        ]
        
        # Model saving
        if self.enable_model_saving:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir_path = f'results/param_tuning/{layer}/{timestamp}'
            from pathlib import Path
            Path(save_dir_path).mkdir(parents=True, exist_ok=True)
            parts.append(f'auto-save=true')
            parts.append(f'save-dir="{save_dir_path}"')
            parts.append(f'save-interval=10')
        else:
            parts.append('auto-save=false')
        
        # Add defense params với CORRECT format: defense.layer.param-name
        # Map layer name to config key
        layer_prefix_map = {
            'layer1': 'defense.layer1',
            'layer2': 'defense.layer2',
            'noniid': 'defense.noniid',
            'filtering': 'defense.filtering',
            'reputation': 'defense.reputation',
            'mode': 'defense.mode'
        }
        
        # Get prefix cho layer này
        if layer in layer_prefix_map:
            prefix = layer_prefix_map[layer]
            
            # Add params với đúng format
            for key, value in config.items():
                # Convert snake_case to kebab-case
                param_name = key.replace('_', '-')
                
                # Format: defense.layer1.pca-dims=20
                if isinstance(value, str):
                    parts.append(f'{prefix}.{param_name}="{value}"')
                else:
                    parts.append(f'{prefix}.{param_name}={value}')
        
        # Join
        run_config = ' '.join(parts)
        
        return run_config
    
    def run_single_test(self, layer: str, config_idx: int, config: Dict) -> Dict:
        """
        Chạy 1 test với 1 param config.
        
        Returns:
            Dict với test results
        """
        print(f"\n{'='*70}")
        print(f"🧪 Testing {layer.upper()} - Config #{config_idx + 1}")
        print(f"{'='*70}")
        print("Parameters:")
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")
        print()
        
        # Generate run config
        run_config = self.generate_run_config(layer, config)
        
        # Build command
        cmd = ['flwr', 'run', '.', '--run-config', run_config]
        
        # Execute
        start_time = time.time()
        
        print(f"⏳ Starting test...")
        print(f"   Command: flwr run . --run-config '{run_config[:100]}...'")
        print(f"\n{'='*70}")
        print("TEST OUTPUT (REAL-TIME):")
        print(f"{'='*70}\n")
        
        # Create output log file
        import os
        os.makedirs('test_logs', exist_ok=True)
        log_file = f"test_logs/test_{layer}_{config_idx}.log"
        
        try:
            # Run WITHOUT capturing - output goes directly to terminal
            # But also save to file using tee-like approach
            with open(log_file, 'w') as f:
                # Use shell=False but redirect
                process = Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr to stdout
                    text=True,
                    bufsize=0  # Unbuffered
                )
                
                # Read and display + save simultaneously
                output_lines = []
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    # Show immediately
                    print(line, end='', flush=True)
                    # Save to file
                    f.write(line)
                    f.flush()
                    # Keep for parsing
                    output_lines.append(line)
                
                # Wait for completion
                process.wait()
            
            elapsed = time.time() - start_time
            
            print(f"\n{'='*70}")
            print(f"⏱️  Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"{'='*70}\n")
            
            # Combine output
            full_output = ''.join(output_lines)
            
            if process.returncode == 0:
                print(f"✅ Test completed successfully")
                
                # Parse accuracy từ output
                accuracy = self._parse_accuracy(full_output)
                
                # If accuracy is 0, save full log
                if accuracy == 0.0:
                    print(f"  ⚠️  Accuracy=0.0, full log saved to: {log_file}")
                else:
                    print(f"  ✓ Parsed accuracy: {accuracy:.3f}")
                
                return {
                    'layer': layer,
                    'config_idx': config_idx,
                    'config': config,
                    'success': True,
                    'accuracy': accuracy,
                    'elapsed_time': elapsed,
                    'log_file': log_file,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                print(f"❌ Test failed with return code: {process.returncode}")
                print(f"   Full log: {log_file}")
                
                return {
                    'layer': layer,
                    'config_idx': config_idx,
                    'config': config,
                    'success': False,
                    'error': f'Process failed with code {process.returncode}',
                    'elapsed_time': elapsed,
                    'log_file': log_file,
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n💥 Exception: {str(e)}")
            return {
                'layer': layer,
                'config_idx': config_idx,
                'config': config,
                'success': False,
                'error': str(e),
                'elapsed_time': elapsed,
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_accuracy(self, output: str) -> float:
        """
        Parse final accuracy từ flwr output.
        
        Tries multiple patterns to extract accuracy.
        """
        lines = output.split('\n')
        
        # Pattern 1: Look for "accuracy: 0.xxx" or "accuracy=0.xxx"
        import re
        for line in reversed(lines):
            if 'accuracy' in line.lower():
                # Try pattern: accuracy: 0.754 or accuracy=0.754 or accuracy 0.754
                numbers = re.findall(r'accuracy[\s:=]+(\d+\.\d+)', line.lower())
                if numbers:
                    return float(numbers[0])
                
                # Fallback: any decimal number in line with "accuracy"
                numbers = re.findall(r'\d+\.\d+', line)
                if numbers:
                    # Take first reasonable accuracy (0.0 - 1.0)
                    for num in numbers:
                        val = float(num)
                        if 0.0 <= val <= 1.0:
                            return val
        
        # Pattern 2: Look for percentage "75.4%" and convert
        for line in reversed(lines):
            if 'accuracy' in line.lower() or 'acc' in line.lower():
                percentages = re.findall(r'(\d+\.?\d*)%', line)
                if percentages:
                    return float(percentages[0]) / 100.0
        
        print("  ⚠️  Could not parse accuracy from output, returning 0.0")
        return 0.0  # Default nếu không parse được
    
    def run_all_tests(self):
        """Chạy tất cả tests trong suite."""
        
        print("\n" + "="*70)
        print("STARTING PARAMETER TESTS")
        print("="*70)
        print(f"Suite file: {self.suite_file}")
        
        total_tests = sum(len(configs) for configs in self.suite.values())
        print(f"Total tests: {total_tests}")
        print(f"Attack scenario: {self.attack_type} @ {self.attack_ratio:.0%}")
        print("="*70 + "\n")
        
        test_num = 0
        
        # Test từng layer
        for layer, configs in self.suite.items():
            print(f"\n{'='*70}")
            print(f"🔍 Testing {layer.upper()}")
            print(f"   Configs: {len(configs)}")
            print(f"{'='*70}")
            
            for idx, config in enumerate(configs):
                test_num += 1
                
                # Skip if already completed
                if self._is_test_completed(layer, idx):
                    print(f"\n[{test_num}/{total_tests}] ⏭️  Skipping {layer.upper()} config #{idx+1} (already completed)")
                    continue
                
                print(f"\n[{test_num}/{total_tests}] ", end='')
                
                result = self.run_single_test(layer, idx, config)
                
                if result['success']:
                    self.results.append(result)
                else:
                    self.failed_tests.append(result)
                
                # Save intermediate results
                if test_num % 5 == 0:
                    self._save_intermediate_results()
                
                # Small delay
                time.sleep(2)
        
        # Final save
        self._save_final_results()
    
    def _save_intermediate_results(self):
        """Save intermediate results."""
        output_file = f"param_test_results_intermediate.json"
        
        data = {
            'suite_file': str(self.suite_file),
            'test_config': {
                'num_clients': self.num_clients,
                'num_rounds': self.num_rounds,
                'attack_type': self.attack_type,
                'attack_ratio': self.attack_ratio
            },
            'timestamp': datetime.now().isoformat(),
            'completed_tests': len(self.results),
            'failed_tests': len(self.failed_tests),
            'results': self.results,
            'failures': self.failed_tests
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_final_results(self):
        """Save final results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"param_test_results_{timestamp}.json"
        
        data = {
            'suite_file': str(self.suite_file),
            'test_config': {
                'num_clients': self.num_clients,
                'num_rounds': self.num_rounds,
                'attack_type': self.attack_type,
                'attack_ratio': self.attack_ratio
            },
            'start_time': self.results[0]['timestamp'] if self.results else None,
            'end_time': datetime.now().isoformat(),
            'total_tests': len(self.results) + len(self.failed_tests),
            'successful_tests': len(self.results),
            'failed_tests': len(self.failed_tests),
            'results': self.results,
            'failures': self.failed_tests
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETE!")
        print("="*70)
        print(f"Results saved to: {output_file}")
        print(f"Successful: {len(self.results)}")
        print(f"Failed: {len(self.failed_tests)}")
        print("="*70 + "\n")
        
        return output_file
    
    def _load_previous_results(self):
        """Load previous results để resume."""
        try:
            with open(self.resume_from, 'r') as f:
                data = json.load(f)
            
            # Load results
            self.results = data.get('results', [])
            self.failed_tests = data.get('failures', [])
            
            # Track completed tests
            for result in self.results + self.failed_tests:
                test_id = f"{result['layer']}_{result['config_idx']}"
                self.completed_tests.add(test_id)
            
            print(f"\n{'='*70}")
            print(f"📂 RESUMING FROM PREVIOUS RUN")
            print(f"{'='*70}")
            print(f"Previous results file: {self.resume_from}")
            print(f"Completed tests: {len(self.completed_tests)}")
            print(f"  ✓ Successful: {len(self.results)}")
            print(f"  ✗ Failed: {len(self.failed_tests)}")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"⚠️  Could not load previous results: {e}")
            print("Starting from scratch...\n")
            self.completed_tests = set()
    
    def _is_test_completed(self, layer: str, config_idx: int) -> bool:
        """Check if test đã chạy rồi."""
        test_id = f"{layer}_{config_idx}"
        return test_id in self.completed_tests


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parameter tuning tests')
    parser.add_argument('suite_file', help='Path to param suite JSON file')
    parser.add_argument('--resume', '-r', help='Resume from previous results JSON file')
    parser.add_argument('--save-models', action='store_true', help='Enable model saving')
    
    args = parser.parse_args()
    
    if not Path(args.suite_file).exists():
        print(f"❌ File not found: {args.suite_file}")
        sys.exit(1)
    
    # Check resume file if provided
    if args.resume and not Path(args.resume).exists():
        print(f"❌ Resume file not found: {args.resume}")
        sys.exit(1)
    
    # Create runner
    runner = ParamTestRunner(
        suite_file=args.suite_file,
        enable_model_saving=args.save_models,
        resume_from=args.resume
    )
    
    # Run all tests (will skip completed ones if resuming)
    runner.run_all_tests()


if __name__ == "__main__":
    main()