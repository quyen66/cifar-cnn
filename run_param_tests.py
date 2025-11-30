#!/usr/bin/env python3
"""
FIXED: H Weights Test Runner - Non-IID Version
===============================================

Fixed version with corrected metric parsing that matches actual log format.

Changes from original:
- Detection: X% (not "Detection Rate:")
- FPR: X% (not "False Positive Rate:")  
- TP=X, FP=X format (single line)
- Accuracy from history dict (20, value)

Usage:
    python3 run_h_weight_tests_noniid_FIXED.py <h_weights_suite.json>
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
        
        # Store original directory
        self.original_dir = Path.cwd()
        
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
        print("⚠️  NON-IID TEST MODE (FIXED PARSER):")
        print("   partition-type: dirichlet")
        print("   alpha: 0.5 (moderate non-IID)")
        print("   Parser: Matches actual log format")
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
        
        # Setup logging
        self.log_dir = self.project_root / "h_weight_logs_noniid"
        self.log_dir.mkdir(exist_ok=True)
        
        print(f"✅ Loaded suite: {len(self.configs)} configs")
        if resume_file:
            print(f"   Resuming from: {len(self.completed_ids)} completed")
        print()
    
    def _find_project_root(self) -> Optional[Path]:
        """Find project root."""
        current = Path.cwd()
        
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
        """
        Parse metrics from Flower output (FIXED VERSION).
        
        Matches actual log format:
        - Detection: 8.3% (not "Detection Rate:")
        - FPR: 0.0% (not "False Positive Rate:")
        - TP=1, FP=0, FN=11, TN=28 (single line)
        - accuracy': [(20, 0.095)] (in history)
        """
        metrics = {
            'final_accuracy': None,
            'detection_rate': None,
            'false_positive_rate': None,
            'true_positives': None,
            'false_positives': None,
            'true_negatives': None,
            'false_negatives': None,
            'h_score_mean': None,
            'h_score_max': None
        }
        
        lines = output.split('\n')
        
        # Pattern 1: Final accuracy from history
        accuracy_found = False
        for i, line in enumerate(lines):
            if "'accuracy':" in line or '"accuracy":' in line:
                # Found accuracy history, look for (20, value) in next few lines
                search_lines = lines[i:min(i+30, len(lines))]
                combined = ' '.join(search_lines)
                
                # Try to find (20, X) pattern
                pattern = r'\(20,\s*(\d+\.?\d*)\)'
                match = re.search(pattern, combined)
                if match:
                    metrics['final_accuracy'] = float(match.group(1))
                    accuracy_found = True
                    break
                
                # Fallback: find last accuracy value
                all_accs = re.findall(r'\(\d+,\s*(\d+\.?\d*)\)', combined)
                if all_accs:
                    metrics['final_accuracy'] = float(all_accs[-1])
                    accuracy_found = True
                    break
        
        # Pattern 2: Detection rate
        # Looking for: "Detection: 8.3%"
        for line in reversed(lines):
            match = re.search(r'Detection:\s*(\d+\.?\d*)%', line)
            if match:
                metrics['detection_rate'] = float(match.group(1)) / 100.0
                break
        
        # Pattern 3: FPR
        # Looking for: "FPR: 0.0%"
        for line in reversed(lines):
            match = re.search(r'FPR:\s*(\d+\.?\d*)%', line)
            if match:
                metrics['false_positive_rate'] = float(match.group(1)) / 100.0
                break
        
        # Pattern 4: Confusion matrix
        # Looking for: "TP=1, FP=0, FN=11, TN=28"
        for line in reversed(lines):
            match = re.search(r'TP=(\d+),\s*FP=(\d+),\s*FN=(\d+),\s*TN=(\d+)', line)
            if match:
                metrics['true_positives'] = int(match.group(1))
                metrics['false_positives'] = int(match.group(2))
                metrics['false_negatives'] = int(match.group(3))
                metrics['true_negatives'] = int(match.group(4))
                break
        
        # Pattern 5: H score (Calculate from per-round logs)
        h_scores = []
        for line in lines:
            # Tìm dòng: "   Heterogeneity: H = 0.726"
            match = re.search(r'Heterogeneity:\s*H\s*=\s*(\d+\.?\d*)', line)
            if match:
                try:
                    val = float(match.group(1))
                    h_scores.append(val)
                except ValueError:
                    continue
        
        # Tự tính toán Mean và Max
        if h_scores:
            metrics['h_score_mean'] = sum(h_scores) / len(h_scores)
            metrics['h_score_max'] = max(h_scores)
        else:
            metrics['h_score_mean'] = 0.0
            metrics['h_score_max'] = 0.0
        # -----------------------------------
        
        return metrics
    
    def _run_single_test(self, config: Dict, test_num: int, total: int) -> Dict:
        """Run single test configuration."""
        
        config_id = config['config_id']
        
        print()
        print("=" * 80)
        print(f"TEST {test_num}/{total}: Config #{config_id} (NON-IID, FIXED PARSER)")
        print("=" * 80)
        print()
        
        # Display config
        print("📝 H Weight Configuration:")
        print(f"   weight_cv:      {config['weight_cv']:.2f}")
        print(f"   weight_sim:     {config['weight_sim']:.2f}")
        print(f"   weight_cluster: {config['weight_cluster']:.2f}")
        print(f"   Profile:        {config.get('profile', 'unknown')}")
        print()
        
        # Generate command
        run_config = self._generate_run_config(config)
        cmd = ['flwr', 'run', '.', '--run-config', run_config]
        
        print("🚀 Running command:")
        print(f"   flwr run . --run-config \"{run_config}\"")
        print()
        
        # Setup logging
        log_file = self.log_dir / f"test_h_weight_noniid_{config_id}.log"
        
        # Run test
        start_time = datetime.now()
        
        try:
            original_cwd = os.getcwd()
            
            try:
                os.chdir(str(self.project_root))
                
                with open(log_file, 'w') as log:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    output_lines = []
                    for line in process.stdout:
                        print(line, end='')
                        log.write(line)
                        output_lines.append(line)
                    
                    process.wait()
                    output = ''.join(output_lines)
                    returncode = process.returncode
            
            finally:
                os.chdir(original_cwd)
        
        except Exception as e:
            print(f"\n❌ Error running test: {e}")
            return {
                'config_id': config_id,
                'config': config,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Parse metrics with FIXED parser
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
        
        # Display summary
        print()
        print("=" * 80)
        print("📊 TEST SUMMARY (NON-IID, FIXED PARSER)")
        print("=" * 80)
        print(f"   Status:          {'✅ Success' if returncode == 0 else '❌ Failed'}")
        print(f"   Duration:        {duration:.1f}s ({duration/60:.1f}min)")
        
        if metrics['final_accuracy']:
            print(f"   Accuracy:        {metrics['final_accuracy']:.4f}")
        else:
            print(f"   Accuracy:        N/A")
        
        if metrics['detection_rate']:
            print(f"   Detection Rate:  {metrics['detection_rate']:.2%}")
        else:
            print(f"   Detection Rate:  N/A")
        
        if metrics['false_positive_rate'] is not None:
            print(f"   FP Rate:         {metrics['false_positive_rate']:.2%}")
        else:
            print(f"   FP Rate:         N/A")
        
        if metrics['h_score_mean']:
            print(f"   H Score (mean):  {metrics['h_score_mean']:.3f}")
        
        print(f"   Log:             {log_file}")
        print()
        
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
                'timestamp': datetime.now().isoformat(),
                'status': 'in_progress',
                'data_distribution': 'dirichlet_0.5',
                'parser_version': 'fixed'
            },
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Intermediate results saved: {filename}")
    
    def run_tests(self):
        """Run all tests."""
        
        print()
        print("=" * 80)
        print("STARTING H WEIGHT OPTIMIZATION TESTS (NON-IID, FIXED PARSER)")
        print("=" * 80)
        print()
        print(f"📊 Total configurations: {len(self.configs)}")
        print(f"   Already completed:    {len(self.completed_ids)}")
        print(f"   Remaining:            {len(self.configs) - len(self.completed_ids)}")
        print(f"   Project root:         {self.project_root}")
        print(f"   Data distribution:    Dirichlet (alpha=0.5)")
        print(f"   Parser version:       FIXED (matches actual logs)")
        print()
        
        # Filter remaining configs
        remaining_configs = [c for c in self.configs if c['config_id'] not in self.completed_ids]
        
        for idx, config in enumerate(remaining_configs, 1):
            test_num = len(self.completed_ids) + idx
            
            # Run test
            result = self._run_single_test(config, test_num, len(self.configs))
            
            # Store result
            self.results.append(result)
            self.completed_ids.add(config['config_id'])
            
            # Intermediate save every 3 tests
            if len(self.results) % 3 == 0:
                self._save_intermediate()
        
        # Final save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = self.project_root / f"h_weight_results_noniid_final_FIXED_{timestamp}.json"
        
        final_data = {
            'metadata': {
                'suite_file': self.suite_file,
                'total_configs': len(self.configs),
                'completed': len(self.results),
                'timestamp': datetime.now().isoformat(),
                'status': 'complete',
                'optimization_phase': 'Phase 1 - H Weights (Non-IID)',
                'data_distribution': 'dirichlet_0.5',
                'parser_version': 'fixed'
            },
            'results': self.results
        }
        
        with open(final_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        print()
        print("=" * 80)
        print("✅ ALL TESTS COMPLETE (NON-IID, FIXED PARSER)")
        print("=" * 80)
        print()
        print(f"📊 Final results saved: {final_file}")
        print()
        print("Next step: Analyze results")
        print(f"  python3 analyze_h_weights.py {final_file}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Run H weight tests with Non-IID data (FIXED)')
    parser.add_argument('suite_file', help='Path to H weights suite JSON')
    parser.add_argument('--resume', help='Path to intermediate results for resume')
    
    args = parser.parse_args()
    
    # Validate files
    suite_path = Path(args.suite_file)
    if not suite_path.exists() and not (Path.cwd() / suite_path).exists():
        print(f"❌ Suite file not found: {args.suite_file}")
        sys.exit(1)
    
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists() and not (Path.cwd() / resume_path).exists():
            print(f"❌ Resume file not found: {args.resume}")
            sys.exit(1)
    
    try:
        runner = HWeightTestRunnerNonIID(args.suite_file, args.resume)
        runner.run_tests()
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()