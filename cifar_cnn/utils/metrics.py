"""Metrics collection for experiments."""

import json
from pathlib import Path
from datetime import datetime
import numpy as np


class MetricsCollector:
    """Thu thập metrics qua các rounds."""
    
    def __init__(self, save_dir="results/"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.history = {
            'rounds': [],
            'accuracy': [],
            'loss': [],
            'detected_clients': [],
            'false_positives': [],
            'detection_rate': [],
            'fpr': [],
            'mode': []
        }
    
    def log_round(self, round_num, accuracy, loss, 
                  detected_clients, total_malicious,
                  false_positives, total_benign,
                  mode="NORMAL"):
        """Log metrics cho 1 round."""
        
        detection_rate = detected_clients / total_malicious if total_malicious > 0 else 0
        fpr = false_positives / total_benign if total_benign > 0 else 0
        
        self.history['rounds'].append(round_num)
        self.history['accuracy'].append(accuracy)
        self.history['loss'].append(loss)
        self.history['detected_clients'].append(detected_clients)
        self.history['false_positives'].append(false_positives)
        self.history['detection_rate'].append(detection_rate)
        self.history['fpr'].append(fpr)
        self.history['mode'].append(mode)
        
        print(f"\n📊 Round {round_num} Metrics:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Detection Rate: {detection_rate*100:.1f}%")
        print(f"  FPR: {fpr*100:.1f}%")
        print(f"  Mode: {mode}")
    
    def get_summary(self):
        """Tóm tắt metrics."""
        summary = {
            'final_accuracy': self.history['accuracy'][-1] if self.history['accuracy'] else 0,
            'avg_detection_rate': np.mean(self.history['detection_rate']) if self.history['detection_rate'] else 0,
            'avg_fpr': np.mean(self.history['fpr']) if self.history['fpr'] else 0,
            'total_rounds': len(self.history['rounds'])
        }
        return summary
    
    def save(self, filename):
        """Save metrics to JSON."""
        filepath = self.save_dir / filename
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'history': self.history,
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Metrics saved to: {filepath}")
    
    def load(self, filename):
        """Load metrics from JSON."""
        filepath = self.save_dir / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.history = data['history']
        return data