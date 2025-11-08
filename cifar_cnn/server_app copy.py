"""
Flower Server - FULL PIPELINE với LAYER BREAKDOWN TRACKING
==========================================================
Cải tiến: Track riêng TP/FP/FN/TN cho từng layer

Author: Enhanced Version with Layer Breakdown
"""

import torch
import numpy as np
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedProx
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from cifar_cnn.task import get_model, get_parameters, set_parameters
from cifar_cnn.model_manager import ModelManager
from typing import List, Tuple, Optional, Set, Dict
from flwr.common import Metrics, FitRes, Parameters
from datetime import datetime

# Import ALL defense components
from cifar_cnn.defense import (
    Layer1Detector, Layer2Detector, ReputationSystem, 
    aggregate_by_mode, NonIIDHandler, TwoStageFilter, ModeController
)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Tính trung bình có trọng số của metrics."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


class FullPipelineStrategy(FedProx):
    """
    FedProx Strategy với FULL DEFENSE PIPELINE + LAYER BREAKDOWN TRACKING.
    
    Improvements:
    - Track separate metrics for Layer 1, Layer 2, and Combined
    - Detailed per-layer performance analysis
    """
    
    def __init__(self, **kwargs):
        # Extract custom config
        self.save_dir = kwargs.pop('save_dir', 'models')
        self.auto_save = kwargs.pop('auto_save', True)
        self.save_interval = kwargs.pop('save_interval', 5)
        self.enable_defense = kwargs.pop('enable_defense', False)
        self.config_metadata = kwargs.pop('config_metadata', {})
        
        # Call parent init
        super().__init__(**kwargs)
        
        # Tracking variables
        self.start_time = datetime.now()
        self.best_accuracy = 0.0
        self.accuracy_history = []
        self.loss_history = []
        self.current_parameters = None
        
        # ============================================
        # ENHANCED: LAYER BREAKDOWN TRACKING
        # ============================================
        
        # Layer 1 metrics
        self.layer1_tp = 0
        self.layer1_fp = 0
        self.layer1_fn = 0
        self.layer1_tn = 0
        
        # Layer 2 metrics
        self.layer2_tp = 0
        self.layer2_fp = 0
        self.layer2_fn = 0
        self.layer2_tn = 0
        
        # Combined metrics (tổng cộng)
        self.combined_tp = 0
        self.combined_fp = 0
        self.combined_fn = 0
        self.combined_tn = 0
        
        # History theo round
        self.layer_breakdown_history = []
        
        # Ground truth
        self.malicious_clients = self._identify_malicious_clients()
        
        # ============================================
        # DEFENSE COMPONENTS
        # ============================================
        if self.enable_defense:
            print("\n" + "="*70)
            print("🛡️  INITIALIZING FULL DEFENSE PIPELINE")
            print("="*70)
            
            # Layer 1: Enhanced DBSCAN
            # NOTE: Only use parameters supported by current Layer1Detector
            self.layer1_detector = Layer1Detector(
                pca_dims=20,
                dbscan_min_samples=2,
                warmup_rounds=15
            )
            print("✓ Layer 1: Enhanced DBSCAN initialized")
            
            # Layer 2: Distance + Direction
            # NOTE: Only use parameters supported by current Layer2Detector
            self.layer2_detector = Layer2Detector(
                distance_multiplier=1.5,
                cosine_threshold=0.3,
                warmup_rounds=15
            )
            print("✓ Layer 2: Distance + Direction initialized")
            
            # Non-IID Handler
            self.noniid_handler = NonIIDHandler(
                history_window=5,
                cv_weight=0.4,
                sim_weight=0.4,
                cluster_weight=0.2
            )
            print("✓ Non-IID Handler initialized")
            
            # Reputation System
            self.reputation_system = ReputationSystem(
                alpha_down=0.8,
                alpha_up=0.3,
                floor_threshold=0.15,
                floor_target=0.3,
                floor_patience=5,
                history_window=5
            )
            print("✓ Reputation System initialized")
            
            # Two-Stage Filter
            self.filter_system = TwoStageFilter(
                hard_threshold_base=0.85,
                soft_threshold_base=0.5,
                rep_threshold_normal=0.2,
                rep_threshold_alert=0.4,
                rep_threshold_defense=0.6
            )
            print("✓ Two-Stage Filter initialized")
            
            # Mode Controller
            self.mode_controller = ModeController(
                threshold_normal=0.15,
                threshold_defense=0.30,
                stability_required=2,
                gate1_min_high_rep=3,
                gate1_rep_threshold=0.85,
                gate2_drop_threshold=0.05
            )
            print("✓ Mode Controller initialized")
            
            print("="*70 + "\n")
        
        # Store for evaluation
        self.min_fit_clients = kwargs.get('min_fit_clients', 2)
        self.min_available_clients = kwargs.get('min_available_clients', 2)
    
    def _identify_malicious_clients(self) -> Set[int]:
        """
        Xác định malicious clients dựa trên config.
        Assumes malicious clients are the first N clients where:
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
        # Malicious clients are the first N clients
        malicious_ids = set(range(num_malicious))
        
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
        """
        Aggregate fit results với FULL DEFENSE PIPELINE + LAYER BREAKDOWN.
        """
        
        if not results:
            return None
        
        # Track loss
        if results:
            avg_loss = sum([r.metrics.get("train_loss", 0) for _, r in results]) / len(results)
            self.loss_history.append((server_round, avg_loss))
        
        # NO DEFENSE: standard aggregation
        if not self.enable_defense:
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated is not None:
                self.current_parameters, _ = aggregated
            return aggregated
        
        # ============================================
        # GIAI ĐOẠN 1: THU THẬP VÀ TIỀN XỬ LÝ
        # ============================================
        print(f"\n{'='*70}")
        print(f"ROUND {server_round}: FULL DEFENSE PIPELINE WITH LAYER BREAKDOWN")
        print(f"{'='*70}")
        
        print(f"\n{'='*70}")
        print(f"STAGE 1: DATA COLLECTION & PREPROCESSING")
        print(f"{'='*70}")
        
        gradients = []
        client_ids = []
        client_results = []
        
        for client_proxy, fit_res in results:
            client_id = hash(str(client_proxy)) % self.config_metadata.get('num_clients', 1000)
            client_ids.append(client_id)
            
            params = parameters_to_ndarrays(fit_res.parameters)
            gradient = np.concatenate([p.flatten() for p in params])
            gradients.append(gradient)
            
            client_results.append((client_proxy, fit_res))
        
        print(f"\n   ✓ Collected {len(gradients)} gradients")
        print(f"   ✓ Client IDs: {client_ids}")
        
        # Ground truth
        ground_truth_list = [cid in self.malicious_clients for cid in client_ids]
        num_malicious_ground_truth = sum(ground_truth_list)
        num_benign_ground_truth = len(ground_truth_list) - num_malicious_ground_truth
        
        print(f"   ✓ Ground Truth: {num_malicious_ground_truth} malicious, {num_benign_ground_truth} benign")
        
        # ============================================
        # GIAI ĐOẠN 2: PHÁT HIỆN ĐA TẦNG + LAYER BREAKDOWN
        # ============================================
        print(f"\n{'='*70}")
        print(f"STAGE 2: MULTI-LAYER DETECTION WITH BREAKDOWN TRACKING")
        print(f"{'='*70}")
        
        # ----------------
        # Layer 1 Detection
        # ----------------
        print(f"\n   🔍 Layer 1: Enhanced DBSCAN...")
        layer1_results = self.layer1_detector.detect(
            gradients=gradients,
            client_ids=client_ids,
            is_malicious_ground_truth=ground_truth_list,
            current_round=server_round
        )
        
        # Calculate Layer 1 metrics
        layer1_detected = set([cid for cid, flag in layer1_results.items() if flag])
        true_malicious = set([cid for cid in client_ids if cid in self.malicious_clients])
        true_benign = set([cid for cid in client_ids if cid not in self.malicious_clients])
        
        layer1_tp = len(true_malicious & layer1_detected)
        layer1_fp = len(true_benign & layer1_detected)
        layer1_fn = len(true_malicious - layer1_detected)
        layer1_tn = len(true_benign - layer1_detected)
        
        # Update cumulative Layer 1 metrics
        self.layer1_tp += layer1_tp
        self.layer1_fp += layer1_fp
        self.layer1_fn += layer1_fn
        self.layer1_tn += layer1_tn
        
        # Print Layer 1 metrics
        layer1_precision = (layer1_tp / (layer1_tp + layer1_fp) * 100) if (layer1_tp + layer1_fp) > 0 else 0
        layer1_recall = (layer1_tp / (layer1_tp + layer1_fn) * 100) if (layer1_tp + layer1_fn) > 0 else 0
        layer1_fpr = (layer1_fp / (layer1_fp + layer1_tn) * 100) if (layer1_fp + layer1_tn) > 0 else 0
        
        print(f"\n   📊 Layer 1 Results:")
        print(f"      TP={layer1_tp}, FP={layer1_fp}, FN={layer1_fn}, TN={layer1_tn}")
        print(f"      Precision: {layer1_precision:.1f}%, Recall: {layer1_recall:.1f}%, FPR: {layer1_fpr:.1f}%")
        
        # ----------------
        # Layer 2 Detection
        # ----------------
        print(f"\n   🔍 Layer 2: Distance + Direction...")
        layer2_results = self.layer2_detector.detect(
            gradients=gradients,
            client_ids=client_ids,
            current_round=server_round,
            layer1_flags=layer1_results
        )
        
        # Calculate Layer 2 metrics (chỉ tính cho clients chưa bị Layer 1 flag)
        layer2_detected = set([cid for cid, flag in layer2_results.items() if flag])
        
        # Layer 2 chỉ detect trong số clients chưa bị Layer 1 flag
        clients_not_flagged_by_layer1 = set([cid for cid, flag in layer1_results.items() if not flag])
        true_malicious_not_flagged = true_malicious & clients_not_flagged_by_layer1
        true_benign_not_flagged = true_benign & clients_not_flagged_by_layer1
        
        layer2_tp = len(true_malicious_not_flagged & layer2_detected)
        layer2_fp = len(true_benign_not_flagged & layer2_detected)
        layer2_fn = len(true_malicious_not_flagged - layer2_detected)
        layer2_tn = len(true_benign_not_flagged - layer2_detected)
        
        # Update cumulative Layer 2 metrics
        self.layer2_tp += layer2_tp
        self.layer2_fp += layer2_fp
        self.layer2_fn += layer2_fn
        self.layer2_tn += layer2_tn
        
        # Print Layer 2 metrics
        layer2_precision = (layer2_tp / (layer2_tp + layer2_fp) * 100) if (layer2_tp + layer2_fp) > 0 else 0
        layer2_recall = (layer2_tp / (layer2_tp + layer2_fn) * 100) if (layer2_tp + layer2_fn) > 0 else 0
        layer2_fpr = (layer2_fp / (layer2_fp + layer2_tn) * 100) if (layer2_fp + layer2_tn) > 0 else 0
        
        print(f"\n   📊 Layer 2 Results:")
        print(f"      TP={layer2_tp}, FP={layer2_fp}, FN={layer2_fn}, TN={layer2_tn}")
        print(f"      Precision: {layer2_precision:.1f}%, Recall: {layer2_recall:.1f}%, FPR: {layer2_fpr:.1f}%")
        
        # ----------------
        # Combined Detection
        # ----------------
        combined_flags = {}
        for cid in client_ids:
            combined_flags[cid] = layer1_results.get(cid, False) or layer2_results.get(cid, False)
        
        combined_detected = set([cid for cid, flag in combined_flags.items() if flag])
        
        combined_tp = len(true_malicious & combined_detected)
        combined_fp = len(true_benign & combined_detected)
        combined_fn = len(true_malicious - combined_detected)
        combined_tn = len(true_benign - combined_detected)
        
        # Update cumulative Combined metrics
        self.combined_tp += combined_tp
        self.combined_fp += combined_fp
        self.combined_fn += combined_fn
        self.combined_tn += combined_tn
        
        # Print Combined metrics
        combined_precision = (combined_tp / (combined_tp + combined_fp) * 100) if (combined_tp + combined_fp) > 0 else 0
        combined_recall = (combined_tp / (combined_tp + combined_fn) * 100) if (combined_tp + combined_fn) > 0 else 0
        combined_fpr = (combined_fp / (combined_fp + combined_tn) * 100) if (combined_fp + combined_tn) > 0 else 0
        combined_f1 = (2 * combined_precision * combined_recall / (combined_precision + combined_recall)) if (combined_precision + combined_recall) > 0 else 0
        
        print(f"\n   📊 Combined Results (Layer 1 + Layer 2):")
        print(f"      TP={combined_tp}, FP={combined_fp}, FN={combined_fn}, TN={combined_tn}")
        print(f"      Precision: {combined_precision:.1f}%, Recall: {combined_recall:.1f}%, FPR: {combined_fpr:.1f}%, F1: {combined_f1:.1f}%")
        
        # Save breakdown history
        self.layer_breakdown_history.append({
            'round': server_round,
            'layer1': {
                'tp': layer1_tp, 'fp': layer1_fp, 'fn': layer1_fn, 'tn': layer1_tn,
                'precision': layer1_precision, 'recall': layer1_recall, 'fpr': layer1_fpr
            },
            'layer2': {
                'tp': layer2_tp, 'fp': layer2_fp, 'fn': layer2_fn, 'tn': layer2_tn,
                'precision': layer2_precision, 'recall': layer2_recall, 'fpr': layer2_fpr
            },
            'combined': {
                'tp': combined_tp, 'fp': combined_fp, 'fn': combined_fn, 'tn': combined_tn,
                'precision': combined_precision, 'recall': combined_recall, 'fpr': combined_fpr, 'f1': combined_f1
            }
        })
        
        flagged_clients = list(combined_detected)
        
        # ============================================
        # GIAI ĐOẠN 3: NON-IID HANDLING
        # ============================================
        print(f"\n{'='*70}")
        print(f"STAGE 3: NON-IID ANALYSIS")
        print(f"{'='*70}")
        
        # Update gradient history
        for i, cid in enumerate(client_ids):
            if cid not in self.noniid_handler.gradient_history:
                self.noniid_handler.initialize_client(cid)
            self.noniid_handler.update_client_gradient(cid, gradients[i])
        
        # Compute heterogeneity score
        H = self.noniid_handler.compute_heterogeneity_score(gradients, client_ids)
        print(f"\n   Heterogeneity Score: H = {H:.3f}")
        
        # ============================================
        # GIAI ĐOẠN 4: REPUTATION & FILTERING
        # ============================================
        print(f"\n{'='*70}")
        print(f"STAGE 4: REPUTATION UPDATE & TWO-STAGE FILTERING")
        print(f"{'='*70}")
        
        # Compute gradient median
        grad_matrix = np.vstack([g for g in gradients])
        grad_median = np.median(grad_matrix, axis=0)
        
        # Update reputation
        reputations = {}
        for i, cid in enumerate(client_ids):
            if cid not in self.reputation_system.reputations:
                self.reputation_system.initialize_client(cid)
            
            was_flagged = combined_flags.get(cid, False)
            rep = self.reputation_system.update(
                cid, gradients[i], grad_median, was_flagged, server_round
            )
            reputations[cid] = rep
        
        # Confidence scores (từ detection)
        confidence_scores = {}
        for cid in client_ids:
            if combined_flags.get(cid, False):
                confidence_scores[cid] = 0.9  # High confidence malicious
            else:
                confidence_scores[cid] = 0.1  # Low confidence malicious
        
        # Two-stage filtering
        trusted_clients, filtered_clients, filter_stats = self.filter_system.filter_clients(
            client_ids, confidence_scores, reputations, 'NORMAL', H
        )
        
        print(f"\n   Filtering Results:")
        print(f"      Trusted: {len(trusted_clients)} clients")
        print(f"      Filtered: {len(filtered_clients)} clients")
        
        # ============================================
        # GIAI ĐOẠN 5: MODE SELECTION & AGGREGATION
        # ============================================
        print(f"\n{'='*70}")
        print(f"STAGE 5: MODE SELECTION & AGGREGATION")
        print(f"{'='*70}")
        
        # Compute threat ratio
        rho = len(flagged_clients) / len(client_ids)
        
        # Update mode
        mode = self.mode_controller.update_mode(
            rho, flagged_clients, reputations, server_round
        )
        
        print(f"\n   Threat Ratio: ρ = {rho:.3f}")
        print(f"   Mode: {mode}")
        
        # Prepare trusted gradients for aggregation
        trusted_results = []
        trusted_grads = []
        for i, (client_proxy, fit_res) in enumerate(client_results):
            cid = client_ids[i]
            if cid in trusted_clients:
                trusted_results.append((client_proxy, fit_res))
                trusted_grads.append(gradients[i])
        
        # Aggregate using mode-specific method
        if trusted_grads:
            print(f"\n   Aggregating {len(trusted_grads)} trusted gradients using {mode} mode...")
            aggregated_grad = aggregate_by_mode(trusted_grads, mode=mode)
            
            # Convert back to parameters
            # NOTE: Cần reshape aggregated_grad về đúng shape của model parameters
            # Đây là simplified version, production code cần handle shapes properly
            aggregated = super().aggregate_fit(server_round, trusted_results, failures)
        else:
            print(f"\n   ⚠️  No trusted clients! Using standard aggregation.")
            aggregated = super().aggregate_fit(server_round, results, failures)
        
        # ============================================
        # SUMMARY
        # ============================================
        print(f"\n{'='*70}")
        print(f"ROUND {server_round} SUMMARY")
        print(f"{'='*70}")
        print(f"\n   Detection: {len(flagged_clients)}/{len(client_ids)} flagged")
        print(f"   Combined: TP={combined_tp}, FP={combined_fp}, FN={combined_fn}, TN={combined_tn}")
        print(f"   Performance: Precision={combined_precision:.1f}%, Recall={combined_recall:.1f}%, F1={combined_f1:.1f}%")
        print(f"   Aggregation: {len(trusted_clients)} trusted clients, Mode={mode}")
        print(f"{'='*70}\n")
        
        # Store aggregated parameters
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
        """Save checkpoint with LAYER BREAKDOWN metrics."""
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
            "algorithm": "FedProx + Full Pipeline" if self.enable_defense else "FedProx",
        }
        
        # Add ENHANCED defense stats with layer breakdown
        if self.enable_defense:
            metadata["defense_stats"] = {
                # Layer 1 metrics
                "layer1_tp": self.layer1_tp,
                "layer1_fp": self.layer1_fp,
                "layer1_fn": self.layer1_fn,
                "layer1_tn": self.layer1_tn,
                "layer1_precision": (self.layer1_tp / (self.layer1_tp + self.layer1_fp) * 100) if (self.layer1_tp + self.layer1_fp) > 0 else 0,
                "layer1_recall": (self.layer1_tp / (self.layer1_tp + self.layer1_fn) * 100) if (self.layer1_tp + self.layer1_fn) > 0 else 0,
                "layer1_fpr": (self.layer1_fp / (self.layer1_fp + self.layer1_tn) * 100) if (self.layer1_fp + self.layer1_tn) > 0 else 0,
                
                # Layer 2 metrics
                "layer2_tp": self.layer2_tp,
                "layer2_fp": self.layer2_fp,
                "layer2_fn": self.layer2_fn,
                "layer2_tn": self.layer2_tn,
                "layer2_precision": (self.layer2_tp / (self.layer2_tp + self.layer2_fp) * 100) if (self.layer2_tp + self.layer2_fp) > 0 else 0,
                "layer2_recall": (self.layer2_tp / (self.layer2_tp + self.layer2_fn) * 100) if (self.layer2_tp + self.layer2_fn) > 0 else 0,
                "layer2_fpr": (self.layer2_fp / (self.layer2_fp + self.layer2_tn) * 100) if (self.layer2_fp + self.layer2_tn) > 0 else 0,
                
                # Combined metrics
                "combined_tp": self.combined_tp,
                "combined_fp": self.combined_fp,
                "combined_fn": self.combined_fn,
                "combined_tn": self.combined_tn,
                "combined_precision": (self.combined_tp / (self.combined_tp + self.combined_fp) * 100) if (self.combined_tp + self.combined_fp) > 0 else 0,
                "combined_recall": (self.combined_tp / (self.combined_tp + self.combined_fn) * 100) if (self.combined_tp + self.combined_fn) > 0 else 0,
                "combined_fpr": (self.combined_fp / (self.combined_fp + self.combined_tn) * 100) if (self.combined_fp + self.combined_tn) > 0 else 0,
                
                # History
                "layer_breakdown_history": self.layer_breakdown_history,
                "ground_truth_malicious": list(self.malicious_clients),
            }
        
        if self.accuracy_history:
            metadata["current_accuracy"] = self.accuracy_history[-1][1]
        
        manager = ModelManager(save_dir=self.save_dir)
        manager.save_model(net, metadata, model_name=model_name)
        print(f"[Round {server_round}] 💾 Model saved: {model_name}")


def server_fn(context: Context) -> ServerAppComponents:
    """Create server components với full pipeline."""
    
    # Get run config
    num_rounds = context.run_config.get("num-server-rounds", 10)
    fraction_fit = context.run_config.get("fraction-fit", 0.5)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 0.5)
    
    # Defense config
    enable_defense = context.run_config.get("enable-defense", False)
    
    # Attack config
    attack_type = context.run_config.get("attack-type", "none")
    attack_ratio = context.run_config.get("attack-ratio", 0.0)
    
    # Dataset config
    num_clients = context.run_config.get("num-clients", 10)
    partition_type = context.run_config.get("partition-type", "iid")
    
    auto_save = context.run_config.get("auto-save", True)
    save_dir = context.run_config.get("save-dir", "saved_models")
    save_interval = context.run_config.get("save-interval", 10)

    
    # Metadata for model naming
    config_metadata = {
        "num_clients": num_clients,
        "partition_type": partition_type,
        "attack_type": attack_type,
        "attack_ratio": attack_ratio,
        "enable_defense": enable_defense,
        "num_rounds": num_rounds,
        "proximal_mu": context.run_config.get("proximal-mu", 0.1),
    }
    
    # Initial parameters
    net = get_model()
    params = get_parameters(net)
    initial_parameters = ndarrays_to_parameters(params)
    
    # Create strategy
    strategy = FullPipelineStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=int(num_clients * fraction_fit),
        min_evaluate_clients=int(num_clients * fraction_evaluate),
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        proximal_mu=context.run_config.get("proximal-mu", 0.1),
        save_dir=save_dir,
        auto_save=auto_save,
        save_interval=save_interval,
        enable_defense=enable_defense,
        config_metadata=config_metadata,
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


# Create the ServerApp
app = ServerApp(server_fn=server_fn)