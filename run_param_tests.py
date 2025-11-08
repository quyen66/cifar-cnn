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
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import time


class ParamTestRunner:
    """Tự động chạy tests với các param combinations."""
    
    def __init__(self, suite_file: str):
        """
        Args:
            suite_file: Path to param suite JSON
        """
        self.suite_file = Path(suite_file)
        
        # Load suite
        with open(self.suite_file, 'r') as f:
            data = json.load(f)
        
        self.suite = data['suite']
        self.default_config = data['default_config']
        
        # Test configuration
        self.num_clients = 40
        self.num_rounds = 30  # Giảm từ 50 → 30 để chạy nhanh hơn
        
        # Attack scenario (FIXED để focus vào param tuning)
        self.attack_type = "byzantine"
        self.attack_ratio = 0.3
        self.byzantine_type = "sign_flip"
        self.byzantine_scale = 2.0
        
        # Results
        self.results = []
        self.failed_tests = []
    
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
        
        # Save dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f'"results/param_tuning/{layer}/{timestamp}"'
        parts.append(f'save-dir={save_dir}')
        parts.append('save-interval=10')
        
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
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Test completed in {elapsed:.1f}s")
                
                # Parse accuracy từ output
                accuracy = self._parse_accuracy(result.stdout)
                
                # If accuracy is 0, save output for debugging
                if accuracy == 0.0:
                    debug_file = f"debug_output_{layer}_{config_idx}.txt"
                    with open(debug_file, 'w') as f:
                        f.write("=== STDOUT ===\n")
                        f.write(result.stdout)
                        f.write("\n\n=== STDERR ===\n")
                        f.write(result.stderr)
                    print(f"  ⚠️  Accuracy=0.0, saved output to {debug_file}")
                
                return {
                    'layer': layer,
                    'config_idx': config_idx,
                    'config': config,
                    'success': True,
                    'accuracy': accuracy,
                    'elapsed_time': elapsed,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                print(f"❌ Test failed: {result.stderr[:200]}")
                return {
                    'layer': layer,
                    'config_idx': config_idx,
                    'config': config,
                    'success': False,
                    'error': result.stderr[:500],
                    'elapsed_time': elapsed,
                    'timestamp': datetime.now().isoformat()
                }
        
        except subprocess.TimeoutExpired:
            print(f"⏱️  Test timeout after 10 minutes")
            return {
                'layer': layer,
                'config_idx': config_idx,
                'config': config,
                'success': False,
                'error': 'Timeout',
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
            return {
                'layer': layer,
                'config_idx': config_idx,
                'config': config,
                'success': False,
                'error': str(e),
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


def main():
    """Main function."""
    
    if len(sys.argv) < 2:
        print("Usage: python run_param_tests.py <suite_json_file>")
        print("Example: python run_param_tests.py param_suite_quick.json")
        sys.exit(1)
    
    suite_file = sys.argv[1]
    
    if not Path(suite_file).exists():
        print(f"❌ File not found: {suite_file}")
        sys.exit(1)
    
    # Create runner
    runner = ParamTestRunner(suite_file)
    
    # Run all tests
    runner.run_all_tests()


if __name__ == "__main__":
    main()