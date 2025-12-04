#!/usr/bin/env python3
"""
FIXED: H Weights Test Runner - Non-IID Version
===============================================

Fixed version with:
- NO os.chdir() to prevent path errors.
- Robust directory creation.
- Corrected metric parsing.
- Subprocess cwd management.
"""

import json
import subprocess
import sys
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import re


class HWeightTestRunnerNonIID:
    """Test runner for H weight optimization with Non-IID data (FIXED)."""
    
    def __init__(self, suite_file: str, resume_file: Optional[str] = None):
        """Initialize test runner."""
        
        # Store original directory (absolute path)
        self.original_dir = Path.cwd().resolve()
        
        self.suite_file = suite_file
        self.resume_file = resume_file
        
        # Find project root
        self.project_root = self._find_project_root()
        if not self.project_root:
            raise RuntimeError(
                "❌ Could not find project root!\n"
                "   Project root must contain:\n"
                "   - pyproject.toml\n"
                "   - cifar_cnn/ directory\n"
            )
        
        # Verify pyproject.toml
        pyproject_path = self.project_root / 'pyproject.toml'
        if not pyproject_path.exists():
            raise RuntimeError(f"❌ pyproject.toml not found at {pyproject_path}")
        
        print(f"✅ Project root: {self.project_root}")
        print(f"   pyproject.toml: {pyproject_path}")
        print()
        print("⚠️  NON-IID TEST MODE (FIXED PARSER & ROBUST PATHS):")
        print("   partition-type: dirichlet")
        print("   alpha: 0.5 (moderate non-IID)")
        print()
        
        # Load suite
        suite_path = Path(suite_file)
        if not suite_path.is_absolute():
            suite_path = self.original_dir / suite_file
        
        with open(suite_path, 'r') as f:
            data = json.load(f)
        
        self.suite = data.get('suite', data)
        self.metadata = data.get('metadata', {})
        self.configs = self.suite.get('h_weights', [])
        
        # Results tracking
        self.results = []
        self.completed_ids = set()
        
        # Resume if specified
        if resume_file:
            self._load_resume_file()
        
        # Setup logging directory
        self.log_dir = self.project_root / "h_weight_logs_noniid"
        # Ensure it exists immediately
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Loaded suite: {len(self.configs)} configs")
        if resume_file:
            print(f"   Resuming from: {len(self.completed_ids)} completed")
        print()
    
    def _find_project_root(self) -> Optional[Path]:
        """Find project root."""
        current = Path.cwd().resolve()
        
        for parent in [current] + list(current.parents):
            pyproject = parent / 'pyproject.toml'
            cifar_dir = parent / 'cifar_cnn'
            
            if pyproject.exists() and cifar_dir.exists():
                return parent
        
        return None
    
    def _load_resume_file(self):
        """Load previous results for resume."""
        resume_path = Path(self.resume_file)
        if not resume_path.is_absolute():
            resume_path = self.original_dir / self.resume_file
        
        with open(resume_path, 'r') as f:
            resume_data = json.load(f)
        
        self.results = resume_data.get('results', [])
        self.completed_ids = {r['config_id'] for r in self.results}
    
    def _convert_to_kebab_case(self, snake_case: str) -> str:
        """Convert snake_case to kebab-case."""
        return snake_case.replace('_', '-')
    
    def _generate_run_config(self, config: Dict) -> str:
        """Generate flwr run-config string with NON-IID data."""
        parts = [
            'num-clients=40',
            'num-server-rounds=20',
            'partition-type="dirichlet"',
            'alpha=0.5',
            'attack-type="byzantine"',
            'attack-ratio=0.3',
            'byzantine-type="sign_flip"',
            'byzantine-scale=2.0',
            'enable-defense=true',
            'auto-save=false',
        ]
        
        # Add H weight params
        h_weight_params = ['weight_cv', 'weight_sim', 'weight_cluster']
        for key in h_weight_params:
            if key in config:
                kebab_key = self._convert_to_kebab_case(key)
                value = config[key]
                parts.append(f'defense.noniid.{kebab_key}={value}')
        
        return ' '.join(parts)
    
    def _parse_metrics(self, output: str) -> Dict:
        """Parse metrics from Flower output (SMART VERSION)."""
        metrics = {
            'final_accuracy': None,
            'detection_rate': None,     # Sẽ tính từ TP/FN
            'false_positive_rate': None, # Sẽ tính từ FP/TN
            'true_positives': None,
            'false_positives': None,
            'true_negatives': None,
            'false_negatives': None,
            'h_score_mean': None,
            'h_score_max': None
        }
        
        lines = output.split('\n')
        
        # 1. Final accuracy
        for i, line in enumerate(lines):
            if "'accuracy':" in line or '"accuracy":' in line:
                search_lines = lines[i:min(i+30, len(lines))]
                combined = ' '.join(search_lines)
                match = re.search(r'\(20,\s*(\d+\.?\d*)\)', combined)
                if match:
                    metrics['final_accuracy'] = float(match.group(1))
                    break
                all_accs = re.findall(r'\(\d+,\s*(\d+\.?\d*)\)', combined)
                if all_accs:
                    metrics['final_accuracy'] = float(all_accs[-1])
                    break
        
        # 2. Parse Metrics (TP, FP, TN, FN) - ƯU TIÊN TÌM DÒNG METRICS CUỐI CÙNG
        for line in reversed(lines):
            # Regex khớp với log: "TP=4, FP=2, FN=8, TN=26"
            match = re.search(r'TP=(\d+),\s*FP=(\d+),\s*FN=(\d+),\s*TN=(\d+)', line)
            if match:
                metrics['true_positives'] = int(match.group(1))
                metrics['false_positives'] = int(match.group(2))
                metrics['false_negatives'] = int(match.group(3))
                metrics['true_negatives'] = int(match.group(4))
                break
        
        # Cách 2: Nếu không tìm thấy đủ (Log cũ), tự tính toán từ TP, FP
        if metrics['true_positives'] is None:
            for line in reversed(lines):
                # Tìm TP và FP
                match = re.search(r'TP=(\d+),\s*FP=(\d+)', line)
                if match:
                    tp = int(match.group(1))
                    fp = int(match.group(2))
                    metrics['true_positives'] = tp
                    metrics['false_positives'] = fp
                    
                    # TỰ TÍNH FN, TN (Dựa trên config: 40 clients, 30% malicious)
                    # Total Clients = 40
                    # Malicious = 12, Benign = 28
                    total_malicious = 12
                    total_benign = 28
                    
                    metrics['false_negatives'] = max(0, total_malicious - tp)
                    metrics['true_negatives'] = max(0, total_benign - fp)
                    break

        # 3. FIX: TỰ TÍNH DETECTION RATE & FPR TỪ CÁC GIÁ TRỊ ĐÃ PARSE
        # Thay vì regex tìm text "Detect=", ta tính toán trực tiếp cho chính xác
        tp = metrics['true_positives']
        fn = metrics['false_negatives']
        fp = metrics['false_positives']
        tn = metrics['true_negatives']

        if tp is not None and fn is not None:
            # Detection Rate = Recall = TP / (TP + FN)
            if (tp + fn) > 0:
                metrics['detection_rate'] = tp / (tp + fn)
            else:
                metrics['detection_rate'] = 0.0
        
        if fp is not None and tn is not None:
            # False Positive Rate = FP / (TN + FP)
            if (tn + fp) > 0:
                metrics['false_positive_rate'] = fp / (tn + fp)
            else:
                metrics['false_positive_rate'] = 0.0

        # 4. H Score
        h_scores = []
        for line in lines:
            match = re.search(r'Heterogeneity Score H\s*=\s*(\d+\.?\d*)', line)
            if match:
                try: h_scores.append(float(match.group(1)))
                except: continue
        
        if h_scores:
            metrics['h_score_mean'] = sum(h_scores) / len(h_scores)
            metrics['h_score_max'] = max(h_scores)
        else:
            metrics['h_score_mean'] = 0.0
            metrics['h_score_max'] = 0.0
        
        return metrics
    
    
    def _run_single_test(self, config: Dict, test_num: int, total: int) -> Dict:
        """Run single test configuration."""
        
        config_id = config['config_id']
        
        print()
        print("=" * 80)
        print(f"TEST {test_num}/{total}: Config #{config_id} (NON-IID, FIXED)")
        print("=" * 80)
        
        # Display config
        print("📝 H Weight Configuration:")
        print(f"   weight_cv:      {config['weight_cv']:.2f}")
        print(f"   weight_sim:     {config['weight_sim']:.2f}")
        print(f"   weight_cluster: {config['weight_cluster']:.2f}")
        print()
        
        # Generate command
        run_config = self._generate_run_config(config)
        cmd = ['flwr', 'run', '.', '--run-config', run_config]
        
        print("🚀 Running command:")
        print(f"   { ' '.join(cmd[:7]) } ...")
        print()
        
        # Force create log directory again to be absolutely sure
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"test_h_weight_noniid_{config_id}.log"
        
        # Run test
        start_time = datetime.now()
        
        try:
            # FIX: Open file first, then run subprocess without os.chdir
            # Use cwd param in Popen to execute in project root
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(self.project_root)  # <--- CRITICAL FIX
                )
                
                output_lines = []
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)
                    output_lines.append(line)
                
                process.wait()
                output = ''.join(output_lines)
                returncode = process.returncode
        
        except Exception as e:
            print(f"\n❌ Error running test: {e}")
            return {
                'config_id': config_id,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Parse metrics
        metrics = self._parse_metrics(output)
        
        # Build result
        result = {
            'config_id': config_id,
            'config': config,
            'metrics': metrics,
            'status': 'success' if returncode == 0 else 'failed',
            'returncode': returncode,
            'duration_seconds': duration,
            'log_file': str(log_file),
            'timestamp': end_time.isoformat(),
            'data_distribution': 'dirichlet_0.5'
        }
        
        # Summary
        print()
        print(f"📊 Result: {'✅ Success' if returncode == 0 else '❌ Failed'}")
        if metrics['final_accuracy']:
            print(f"   Accuracy: {metrics['final_accuracy']:.4f}")
        if metrics['detection_rate']:
            print(f"   Detection: {metrics['detection_rate']:.1%}")
        
        return result
    
    def _save_intermediate(self, filename: str = None):
        """Save intermediate results."""
        if filename is None:
            filename = str(self.project_root / "h_weight_results_noniid_intermediate_FIXED.json")
        
        data = {
            'metadata': {
                'suite_file': self.suite_file,
                'total_configs': len(self.configs),
                'completed': len(self.results),
                'timestamp': datetime.now().isoformat()
            },
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Intermediate results saved.")
    
    def run_tests(self):
        """Run all tests."""
        print(f"STARTING TESTS: {len(self.configs)} total")
        
        # Filter remaining configs
        remaining_configs = [c for c in self.configs if c['config_id'] not in self.completed_ids]
        
        # Calculate start index correctly
        start_index = len(self.completed_ids) + 1
        
        for idx, config in enumerate(remaining_configs):
            current_test_num = start_index + idx
            
            result = self._run_single_test(config, current_test_num, len(self.configs))
            
            self.results.append(result)
            self.completed_ids.add(config['config_id'])
            
            self._save_intermediate()
        
        # Final save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = self.project_root / f"h_weight_results_noniid_final_{timestamp}.json"
        
        data = {
            'metadata': {
                'suite_file': self.suite_file,
                'completed': len(self.results),
                'timestamp': datetime.now().isoformat()
            },
            'results': self.results
        }
        
        with open(final_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\n✅ ALL DONE. Results: {final_file}")

def main():
    parser = argparse.ArgumentParser(description='Run H weight tests')
    parser.add_argument('suite_file', help='Path to H weights suite JSON')
    parser.add_argument('--resume', help='Path to intermediate results')
    args = parser.parse_args()
    
    try:
        runner = HWeightTestRunnerNonIID(args.suite_file, args.resume)
        runner.run_tests()
    except Exception as e:
        print(f"Critical Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()