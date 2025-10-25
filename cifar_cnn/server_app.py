"""Flower server implementation với Layer 1 Defense + Detection Metrics."""

import torch
import numpy as np
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedProx
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from cifar_cnn.task import get_model, get_parameters, set_parameters
from cifar_cnn.model_manager import ModelManager
from typing import List, Tuple, Optional, Set
from flwr.common import Metrics, FitRes, Parameters
from datetime import datetime

# Import metrics collector
from cifar_cnn.utils import MetricsCollector

# Import defense components
from cifar_cnn.defense import Layer1Detector


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Tính trung bình có trọng số của metrics."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


class CustomFedProx(FedProx):
    """FedProx với Layer 1 Defense + Detection Metrics."""
    
    def __init__(self, *args, 
                 auto_save=True, 
                 save_dir="saved_models", 
                 save_interval=10, 
                 config_metadata=None, 
                 start_round=0,
                 enable_defense=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        # Model saving
        self.auto_save = auto_save
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.config_metadata = config_metadata or {}
        self.start_time = datetime.now()
        self.best_accuracy = 0.0
        self.accuracy_history = []
        self.loss_history = []
        self.current_parameters = None
        self.start_round = start_round
        
        # Defense system
        self.enable_defense = enable_defense
        
        # Ground truth for metrics (malicious client IDs)
        self.malicious_clients = self._identify_malicious_clients()
        
        if self.enable_defense:
            print("\n" + "="*70)
            print("🛡️  LAYER 1 DEFENSE SYSTEM ENABLED")
            print("="*70)
            print("  ✓ Layer 1: Enhanced DBSCAN Detection")
            print("  • Magnitude Filter (MAD)")
            print("  • DBSCAN Clustering (20D PCA)")
            print("  • Voting Mechanism")
            print("  • Detection Metrics: TP/FP/FN/TN")
            print("="*70)
            if self.malicious_clients:
                print(f"  🎯 Ground Truth: {len(self.malicious_clients)} malicious clients")
                print(f"     IDs: {sorted(list(self.malicious_clients))}")
            print("="*70 + "\n")
            
            # Only initialize Layer 1 for now
            self.layer1_detector = Layer1Detector()
            
            # Track detection stats
            self.detection_history = []
            self.total_tp = 0
            self.total_fp = 0
            self.total_fn = 0
            self.total_tn = 0
        
        # Metrics collector
        self.metrics_collector = MetricsCollector(save_dir=save_dir)
    
    def _identify_malicious_clients(self) -> Set[int]:
        """
        Identify malicious clients based on attack configuration.
        
        Assumes malicious clients are the last N clients where:
        N = num_clients * attack_ratio
        
        Returns:
            Set of malicious client IDs
        """
        num_clients = self.config_metadata.get('num_clients', 0)
        attack_ratio = self.config_metadata.get('attack_ratio', 0.0)
        attack_type = self.config_metadata.get('attack_type', 'none')
        
        if attack_type == 'none' or attack_ratio == 0.0:
            return set()
        
        num_malicious = int(num_clients * attack_ratio)
        # Malicious clients are the last N clients
        malicious_ids = set(range(num_clients - num_malicious, num_clients))
        
        return malicious_ids
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results."""
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated is not None:
            _, metrics = aggregated
            if "accuracy" in metrics:
                acc = metrics["accuracy"]
                self.accuracy_history.append((server_round, acc))
                if acc > self.best_accuracy:
                    self.best_accuracy = acc
        
        if self.auto_save and server_round % self.save_interval == 0:
            self._save_checkpoint(server_round)
        
        return aggregated
    
    def aggregate_fit(self, 
                     server_round: int, 
                     results: List[Tuple[any, FitRes]], 
                     failures: List[any]) -> Optional[Tuple[Parameters, dict]]:
        """Aggregate fit results with Layer 1 Defense + Metrics."""
        
        if not results:
            return None
        
        # Track loss
        if results:
            avg_loss = sum([r.metrics.get("train_loss", 0) for _, r in results]) / len(results)
            self.loss_history.append((server_round, avg_loss))
        
        # ============================================
        # LAYER 1 DEFENSE INTEGRATION
        # ============================================
        if self.enable_defense:
            print(f"\n[Round {server_round}] 🛡️  Running Layer 1 Detection...")
            
            # Extract gradients and client IDs
            gradients = []
            client_ids = []
            client_results = []
            
            for client_proxy, fit_res in results:
                # Get client ID (from proxy or generate)
                # Use a consistent ID based on proxy hash
                client_id = hash(str(client_proxy)) % self.config_metadata.get('num_clients', 1000)
                client_ids.append(client_id)
                
                # Convert parameters to numpy arrays (gradients)
                params = parameters_to_ndarrays(fit_res.parameters)
                # Flatten all parameters into single vector
                gradient = np.concatenate([p.flatten() for p in params])
                gradients.append(gradient)
                
                client_results.append((client_proxy, fit_res))
            
            # Run Layer 1 Detection
            detection_results = self.layer1_detector.detect(
                gradients=gradients,
                client_ids=client_ids,
                current_round=server_round
            )
            
            # ============================================
            # CALCULATE DETECTION METRICS (TP/FP/FN/TN)
            # ============================================
            
            # Ground truth
            true_malicious = set()
            true_benign = set()
            for cid in client_ids:
                if cid in self.malicious_clients:
                    true_malicious.add(cid)
                else:
                    true_benign.add(cid)
            
            # Detected as malicious
            detected_malicious = set([cid for cid, is_mal in detection_results.items() if is_mal])
            detected_benign = set([cid for cid, is_mal in detection_results.items() if not is_mal])
            
            # Calculate metrics
            tp = len(true_malicious & detected_malicious)  # True Positives
            fp = len(true_benign & detected_malicious)      # False Positives
            fn = len(true_malicious & detected_benign)      # False Negatives
            tn = len(true_benign & detected_benign)         # True Negatives
            
            # Update totals
            self.total_tp += tp
            self.total_fp += fp
            self.total_fn += fn
            self.total_tn += tn
            
            # Calculate rates
            total_malicious = len(true_malicious)
            total_benign = len(true_benign)
            
            detection_rate = (tp / total_malicious * 100) if total_malicious > 0 else 0
            fpr = (fp / total_benign * 100) if total_benign > 0 else 0
            precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
            recall = detection_rate
            
            # Print detection summary with metrics
            print(f"\n   📊 Detection Metrics (Round {server_round}):")
            print(f"      Ground Truth: {total_malicious} malicious, {total_benign} benign")
            print(f"      ✓ TP (True Positive):  {tp:2d} - Correctly detected malicious")
            print(f"      ✗ FP (False Positive): {fp:2d} - Benign flagged as malicious")
            print(f"      ✗ FN (False Negative): {fn:2d} - Malicious missed")
            print(f"      ✓ TN (True Negative):  {tn:2d} - Correctly identified benign")
            print(f"\n      📈 Rates:")
            print(f"         Detection Rate: {detection_rate:.1f}% ({tp}/{total_malicious})")
            print(f"         False Positive Rate: {fpr:.1f}% ({fp}/{total_benign})")
            print(f"         Precision: {precision:.1f}%")
            print(f"         Recall: {recall:.1f}%")
            
            # Save to history
            self.detection_history.append({
                'round': server_round,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'detection_rate': detection_rate,
                'fpr': fpr,
                'precision': precision,
                'recall': recall,
                'detected_ids': list(detected_malicious),
                'ground_truth_malicious': list(true_malicious)
            })
            
            # Filter malicious clients
            filtered_results = []
            for i, (client_proxy, fit_res) in enumerate(client_results):
                client_id = client_ids[i]
                is_malicious = detection_results.get(client_id, False)
                
                if not is_malicious:
                    filtered_results.append((client_proxy, fit_res))
            
            print(f"\n   🔒 Filtered: {len(detected_malicious)} clients excluded from aggregation")
            print(f"   ✓ Clean clients for aggregation: {len(filtered_results)}/{len(results)}\n")
            
            # Use filtered results for aggregation
            results = filtered_results
            
            # If too few clients left, log warning
            if len(results) < self.min_fit_clients:
                print(f"   ⚠️  Warning: Only {len(results)} clients left after filtering (min={self.min_fit_clients})")
        
        # Standard aggregation with (filtered) results
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        if aggregated is not None:
            self.current_parameters, _ = aggregated
        
        return aggregated
    
    def _generate_model_name(self, server_round, is_final=False):
        """Generate model name với attack info."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_clients = self.config_metadata.get('num_clients', 'unknown')
        partition_type = self.config_metadata.get('partition_type', 'unknown')
        attack_type = self.config_metadata.get('attack_type', 'none')
        attack_ratio = self.config_metadata.get('attack_ratio', 0.0)
        defense_enabled = self.config_metadata.get('enable_defense', False)
        
        if attack_type == "none" or attack_ratio == 0:
            attack_info = "noattack"
        else:
            attack_short = {
                'label_flip': 'labelflip', 
                'byzantine': 'byzantine', 
                'gaussian': 'gaussian',
            }.get(attack_type, attack_type)
            attack_pct = int(attack_ratio * 100)
            attack_info = f"{attack_short}{attack_pct}pct"
        
        defense_info = "defense" if defense_enabled else "nodefense"
        round_suffix = "FINAL" if is_final else f"round{server_round}"
        
        return f"{num_clients}c_{partition_type}_{attack_info}_{defense_info}_{timestamp}_{round_suffix}"
    
    def _save_checkpoint(self, server_round):
        """Save checkpoint with detection metrics."""
        if self.current_parameters is None:
            return
        
        net = get_model()
        params_arrays = parameters_to_ndarrays(self.current_parameters)
        set_parameters(net, params_arrays)
        
        model_name = self._generate_model_name(server_round, is_final=False)
        
        metadata = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "training_time": str(datetime.now() - self.start_time),
            "current_round": server_round,
            "best_accuracy": self.best_accuracy,
            "config": self.config_metadata,
            "accuracy_history": self.accuracy_history,
            "loss_history": self.loss_history,
            "algorithm": "FedProx + Layer1Defense" if self.enable_defense else "FedProx",
        }
        
        # Add defense stats if enabled
        if self.enable_defense:
            metadata["defense_stats"] = {
                "total_tp": self.total_tp,
                "total_fp": self.total_fp,
                "total_fn": self.total_fn,
                "total_tn": self.total_tn,
                "detection_history": self.detection_history,
                "ground_truth_malicious": list(self.malicious_clients),
                "avg_detection_rate": (self.total_tp / (self.total_tp + self.total_fn) * 100) if (self.total_tp + self.total_fn) > 0 else 0,
                "avg_fpr": (self.total_fp / (self.total_fp + self.total_tn) * 100) if (self.total_fp + self.total_tn) > 0 else 0
            }
        
        if self.accuracy_history:
            metadata["current_accuracy"] = self.accuracy_history[-1][1]
        
        manager = ModelManager(save_dir=self.save_dir)
        manager.save_model(net, metadata, model_name=model_name)
        print(f"[Round {server_round}] 💾 Model saved: {model_name}")


def server_fn(context: Context) -> ServerAppComponents:
    """Create server components."""
    
    # Config
    num_rounds = context.run_config.get("num-server-rounds", 50)
    num_clients = context.run_config.get("num-clients", 20)
    fraction_fit = context.run_config.get("fraction-fit", 0.5)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 0.5)
    
    num_fit_clients = max(1, int(num_clients * fraction_fit))
    num_evaluate_clients = max(1, int(num_clients * fraction_evaluate))
    
    min_fit_clients = context.run_config.get("min-fit-clients", num_fit_clients)
    min_evaluate_clients = context.run_config.get("min-evaluate-clients", num_evaluate_clients)
    min_available_clients = context.run_config.get("min-available-clients", num_clients)
    
    proximal_mu = context.run_config.get("proximal-mu", 0.01)
    auto_save = context.run_config.get("auto-save", True)
    save_dir = context.run_config.get("save-dir", "saved_models")
    save_interval = context.run_config.get("save-interval", 10)
    
    # Defense config
    enable_defense = context.run_config.get("enable-defense", False)
    
    # Attack config
    attack_type = context.run_config.get("attack-type", "none")
    attack_ratio = context.run_config.get("attack-ratio", 0.0)
    partition_type = context.run_config.get("partition-type", "iid")
    
    # Resume
    resume_from = context.run_config.get("resume-from", None)
    net = get_model()
    start_round = 0
    
    if resume_from and resume_from.lower() not in ["none", ""]:
        manager = ModelManager(save_dir=save_dir)
        try:
            net, loaded_metadata = manager.load_model(net, resume_from)
            start_round = loaded_metadata.get("current_round", 0)
            print(f"✓ Resumed from round {start_round}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
    
    # Print config
    print(f"\n{'='*70}")
    print("SERVER CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Algorithm: FedProx")
    print(f"  Total clients: {num_clients}")
    print(f"  Clients per round: {num_fit_clients}")
    print(f"  Num rounds: {num_rounds}")
    print(f"  Proximal mu: {proximal_mu}")
    print(f"  Partition: {partition_type}")
    print(f"  Attack: {attack_type} ({attack_ratio*100:.0f}%)")
    print(f"  Defense: {'ENABLED ✓' if enable_defense else 'DISABLED ✗'}")
    print(f"{'='*70}\n")
    
    # Config metadata
    config_metadata = {
        'num_clients': num_clients,
        'partition_type': partition_type,
        'attack_type': attack_type,
        'attack_ratio': attack_ratio,
        'enable_defense': enable_defense,
        'proximal_mu': proximal_mu,
    }
    
    # Strategy
    strategy = CustomFedProx(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=ndarrays_to_parameters(get_parameters(net)),
        proximal_mu=proximal_mu,
        auto_save=auto_save,
        save_dir=save_dir,
        save_interval=save_interval,
        config_metadata=config_metadata,
        start_round=start_round,
        enable_defense=enable_defense
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)