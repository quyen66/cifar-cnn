"""
Flower Server - FULL PIPELINE INTEGRATION
==========================================
Tích hợp đầy đủ hệ thống phòng thủ thích ứng lai theo đề cương luận văn.

Pipeline gồm 5 giai đoạn (theo main.pdf):
1. Thu thập và tiền xử lý
2. Phát hiện đa lớp + xử lý Non-IID
3. Tính điểm tin cậy và lọc hai giai đoạn
4. Cập nhật danh tiếng và đánh giá đe dọa
5. Quyết định chế độ và tổng hợp

Author: Week 3 Full Integration
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
from cifar_cnn.defense import Layer1Detector, Layer2Detector, ReputationSystem, aggregate_by_mode, NonIIDHandler, TwoStageFilter, ModeController

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Tính trung bình có trọng số của metrics."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


class FullPipelineStrategy(FedProx):
    """
    FedProx Strategy với FULL DEFENSE PIPELINE.
    
    Tích hợp theo đề cương main.pdf:
    - Layer 1 + Layer 2 Detection
    - Non-IID Handling (Heterogeneity Score + Adaptive Thresholds + Baseline Tracking)
    - Two-Stage Filtering (Hard + Soft)
    - Reputation System (Asymmetric EMA + Floor Lifting)
    - Mode Controller (3 modes + Hysteresis + Reputation Gates)
    - Mode-Adaptive Aggregation (NORMAL/ALERT/DEFENSE)
    """
    
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
        
        # Ground truth for metrics
        self.malicious_clients = self._identify_malicious_clients()
        
        if self.enable_defense:
            print("\n" + "="*70)
            print("🛡️  FULL DEFENSE PIPELINE ENABLED")
            print("="*70)
            print("  ✓ Layer 1: Enhanced DBSCAN Detection")
            print("  ✓ Layer 2: Distance + Direction Detection")
            print("  ✓ Non-IID Handler (H-score + Adaptive + Baseline)")
            print("  ✓ Two-Stage Filtering (Hard + Soft)")
            print("  ✓ Reputation System (Asymmetric EMA + Floor Lifting)")
            print("  ✓ Mode Controller (3 modes + Hysteresis + Rep Gates)")
            print("  ✓ Mode-Adaptive Aggregation (NORMAL/ALERT/DEFENSE)")
            print("="*70)
            if self.malicious_clients:
                print(f"  🎯 Ground Truth: {len(self.malicious_clients)} malicious clients")
                print(f"     IDs: {sorted(list(self.malicious_clients))}")
            print("="*70 + "\n")
            
            # Initialize ALL components
            self.layer1_detector = Layer1Detector()
            self.layer2_detector = Layer2Detector()
            self.noniid_handler = NonIIDHandler()
            self.two_stage_filter = TwoStageFilter()
            self.reputation_system = ReputationSystem()
            self.mode_controller = ModeController()
            
            # Track detection stats
            self.detection_history = []
            self.total_tp = 0
            self.total_fp = 0
            self.total_fn = 0
            self.total_tn = 0
    
    def _identify_malicious_clients(self) -> Set[int]:
        """Identify malicious clients dựa trên attack config."""
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
        Aggregate fit results với FULL DEFENSE PIPELINE (5 giai đoạn).
        """
        
        if not results:
            return None
        
        # Track loss
        if results:
            avg_loss = sum([r.metrics.get("train_loss", 0) for _, r in results]) / len(results)
            self.loss_history.append((server_round, avg_loss))
        
        # ============================================
        # GIAI ĐOẠN 1: THU THẬP VÀ TIỀN XỬ LÝ
        # ============================================
        print(f"\n[Round {server_round}] 🛡️  Full Defense Pipeline")
        print(f"{'='*70}")
        print(f"STAGE 1: COLLECTION & PREPROCESSING")
        print(f"{'='*70}")
        
        # Extract gradients and client IDs
        gradients = []
        client_ids = []
        client_results = []
        
        for client_proxy, fit_res in results:
            # Get client ID (consistent hashing)
            client_id = hash(str(client_proxy)) % self.config_metadata.get('num_clients', 1000)
            client_ids.append(client_id)
            
            # Convert parameters to gradients
            params = parameters_to_ndarrays(fit_res.parameters)
            gradient = np.concatenate([p.flatten() for p in params])
            gradients.append(gradient)
            
            client_results.append((client_proxy, fit_res))
        
        print(f"   ✓ Collected {len(gradients)} gradients (dim={gradients[0].shape[0]})")
        
        # Compute median gradient (reference point)
        grad_matrix = np.vstack([g.flatten() for g in gradients])
        grad_median = np.median(grad_matrix, axis=0)
        print(f"   ✓ Computed median gradient")
        
        if not self.enable_defense:
            # Standard aggregation without defense
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated is not None:
                self.current_parameters, _ = aggregated
            return aggregated
        
        # ============================================
        # GIAI ĐOẠN 2: PHÁT HIỆN ĐA TẦNG + XỬ LÝ NON-IID
        # ============================================
        print(f"\n{'='*70}")
        print(f"STAGE 2: MULTI-LAYER DETECTION + NON-IID HANDLING")
        print(f"{'='*70}")
        
        # Ground truth for evaluation
        ground_truth_list = [cid in self.malicious_clients for cid in client_ids]
        
        # Layer 1 Detection
        print(f"\n   🔍 Layer 1: Enhanced DBSCAN...")
        layer1_results = self.layer1_detector.detect(
            gradients=gradients,
            client_ids=client_ids,
            is_malicious_ground_truth=ground_truth_list,
            current_round=server_round
        )
        
        # Layer 2 Detection (only for clients not flagged by Layer 1)
        print(f"\n   🔍 Layer 2: Distance + Direction...")
        layer2_results = self.layer2_detector.detect(
            gradients=gradients,
            client_ids=client_ids,
            current_round=server_round,
            layer1_flags=layer1_results
        )
        
        # Combine detection results
        combined_flags = {}
        for cid in client_ids:
            combined_flags[cid] = layer1_results.get(cid, False) or layer2_results.get(cid, False)
        
        flagged_clients = [cid for cid, flag in combined_flags.items() if flag]
        
        # Non-IID Handling
        print(f"\n   📊 Non-IID Analysis...")
        
        # Update gradient history
        for i, cid in enumerate(client_ids):
            self.noniid_handler.update_client_gradient(cid, gradients[i])
        
        # Compute heterogeneity score
        H = self.noniid_handler.compute_heterogeneity_score(gradients, client_ids)
        print(f"      Heterogeneity Score: H = {H:.3f}")
        
        # Compute baseline deviations
        baseline_deviations = {}
        for i, cid in enumerate(client_ids):
            delta = self.noniid_handler.compute_baseline_deviation(cid, gradients[i])
            baseline_deviations[cid] = delta
        
        # ============================================
        # GIAI ĐOẠN 3: TÍNH ĐIỂM TIN CẬY VÀ LỌC HAI GIAI ĐOẠN
        # ============================================
        print(f"\n{'='*70}")
        print(f"STAGE 3: CONFIDENCE SCORING + TWO-STAGE FILTERING")
        print(f"{'='*70}")
        
        # Compute confidence scores (theo công thức trong main.pdf)
        confidence_scores = {}
        for i, cid in enumerate(client_ids):
            # Base score from detection layers
            s_base = 1.0 if combined_flags[cid] else 0.0
            
            # Reputation adjustment (old reputation)
            old_rep = self.reputation_system.get_reputation(cid)
            adj_rep = -0.2 if old_rep < 0.3 else 0.0
            
            # Baseline factor
            delta = baseline_deviations.get(cid, 0.0)
            factor_baseline = 0.8 if delta < 0.3 else 1.0
            
            # Combined confidence: c_i = clip((s_base + adj_rep) * factor_baseline)
            confidence = np.clip(
                (s_base + adj_rep) * factor_baseline,
                0.0, 1.0
            )
            confidence_scores[cid] = confidence
        
        print(f"   ✓ Computed confidence scores for {len(confidence_scores)} clients")
        
        # Get current reputations
        all_reputations = self.reputation_system.get_all_reputations()
        
        # Two-Stage Filtering
        print(f"\n   🔒 Two-Stage Filtering...")
        trusted_clients, filtered_clients, filter_stats = self.two_stage_filter.filter_clients(
            client_ids=client_ids,
            confidence_scores=confidence_scores,
            reputations=all_reputations,
            mode=self.mode_controller.current_mode,
            heterogeneity=H
        )
        
        print(f"      Stage 1 (Hard):  {filter_stats['hard_filtered']} filtered")
        print(f"      Stage 2 (Soft):  {filter_stats['soft_filtered']} filtered")
        print(f"      Total Trusted:   {filter_stats['trusted']}/{len(client_ids)} clients")
        
        # ============================================
        # GIAI ĐOẠN 4: CẬP NHẬT DANH TIẾNG VÀ ĐÁNH GIÁ ĐE DỌA
        # ============================================
        print(f"\n{'='*70}")
        print(f"STAGE 4: REPUTATION UPDATE + THREAT ASSESSMENT")
        print(f"{'='*70}")
        
        # Update reputation for all clients (Asymmetric EMA)
        for i, cid in enumerate(client_ids):
            was_flagged = combined_flags[cid]
            gradient = gradients[i]
            
            new_rep = self.reputation_system.update(
                client_id=cid,
                gradient=gradient,
                grad_median=grad_median,
                was_flagged=was_flagged,
                current_round=server_round
            )
        
        print(f"   ✓ Updated reputations (Asymmetric EMA + Floor Lifting)")
        
        # Get updated reputations
        updated_reputations = self.reputation_system.get_all_reputations()
        
        # Compute weighted threat ratio: ρ = Σ(R_i for flagged) / Σ(all R_j)
        flagged_rep_sum = sum([updated_reputations.get(cid, 0) for cid in flagged_clients])
        total_rep_sum = sum(updated_reputations.values())
        
        rho = flagged_rep_sum / total_rep_sum if total_rep_sum > 0 else 0.0
        
        print(f"\n   📈 Threat Assessment:")
        print(f"      Flagged: {len(flagged_clients)}/{len(client_ids)} clients")
        print(f"      Weighted threat ratio (ρ): {rho:.3f}")
        
        # ============================================
        # GIAI ĐOẠN 5: QUYẾT ĐỊNH CHẾ ĐỘ VÀ TỔNG HỢP
        # ============================================
        print(f"\n{'='*70}")
        print(f"STAGE 5: MODE DECISION + AGGREGATION")
        print(f"{'='*70}")
        
        # Update mode using Mode Controller (Hysteresis + Reputation Gates)
        new_mode = self.mode_controller.update_mode(
            threat_ratio=rho,
            flagged_clients=flagged_clients,
            reputations=updated_reputations,
            current_round=server_round
        )
        
        print(f"\n   🎯 Current Mode: {new_mode}")
        
        # Extract trusted gradients for aggregation
        trusted_gradients = []
        trusted_results = []
        
        for i, cid in enumerate(client_ids):
            if cid in trusted_clients:
                trusted_gradients.append(gradients[i])
                trusted_results.append(client_results[i])
        
        print(f"   ✓ Aggregating {len(trusted_gradients)} trusted clients")
        
        # Mode-Adaptive Aggregation
        if len(trusted_gradients) > 0:
            # NORMAL: Weighted Average
            # ALERT: Trimmed Mean 10%
            # DEFENSE: Coordinate Median
            aggregated_gradient = aggregate_by_mode(trusted_gradients, mode=new_mode)
            
            # Reshape back to parameters
            param_shapes = [p.shape for p in parameters_to_ndarrays(results[0][1].parameters)]
            aggregated_params = []
            offset = 0
            for shape in param_shapes:
                size = int(np.prod(shape))
                param = aggregated_gradient[offset:offset+size].reshape(shape)
                aggregated_params.append(param)
                offset += size
            
            # Convert to Flower format
            aggregated_parameters = ndarrays_to_parameters(aggregated_params)
            aggregated = (aggregated_parameters, {
                "mode": new_mode, 
                "threat_ratio": rho,
                "heterogeneity": H
            })
        else:
            # Fallback: No trusted clients
            print(f"   ⚠️  WARNING: No trusted clients! Using standard aggregation.")
            aggregated = super().aggregate_fit(server_round, trusted_results if trusted_results else results, failures)
        
        # ============================================
        # DETECTION METRICS (TP/FP/FN/TN)
        # ============================================
        print(f"\n{'='*70}")
        print(f"DETECTION METRICS")
        print(f"{'='*70}")
        
        # Ground truth
        true_malicious = set([cid for cid in client_ids if cid in self.malicious_clients])
        true_benign = set([cid for cid in client_ids if cid not in self.malicious_clients])
        
        # Detected as malicious
        detected_malicious = set([cid for cid, flag in combined_flags.items() if flag])
        
        # Calculate metrics
        tp = len(true_malicious & detected_malicious)
        fp = len(true_benign & detected_malicious)
        fn = len(true_malicious - detected_malicious)
        tn = len(true_benign - detected_malicious)
        
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
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        # Print detection summary
        print(f"\n   Ground Truth: {total_malicious} malicious, {total_benign} benign")
        print(f"   ✓ TP (True Positive):  {tp:2d}")
        print(f"   ✗ FP (False Positive): {fp:2d}")
        print(f"   ✗ FN (False Negative): {fn:2d}")
        print(f"   ✓ TN (True Negative):  {tn:2d}")
        print(f"\n   📈 Performance:")
        print(f"      Detection Rate: {detection_rate:.1f}%")
        print(f"      FPR: {fpr:.1f}%")
        print(f"      Precision: {precision:.1f}%")
        print(f"      F1-Score: {f1:.1f}%")
        
        # Save to history
        self.detection_history.append({
            'round': server_round,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'detection_rate': detection_rate,
            'fpr': fpr,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mode': new_mode,
            'threat_ratio': rho,
            'heterogeneity': H,
            'detected_ids': list(detected_malicious),
            'ground_truth': list(true_malicious)
        })
        
        print(f"{'='*70}\n")
        
        # Store parameters
        if aggregated is not None:
            self.current_parameters, _ = aggregated
        
        return aggregated
    
    def _generate_model_name(self, server_round, is_final=False):
        """Generate model name với attack info."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_clients = self.config_metadata.get('num_clients', 'unknown')
        attack_type = self.config_metadata.get('attack_type', 'none')
        attack_ratio = self.config_metadata.get('attack_ratio', 0.0)
        
        attack_info = f"{attack_type}{int(attack_ratio*100)}pct" if attack_type != 'none' else "clean"
        defense_info = "fullpipeline" if self.enable_defense else "baseline"
        round_suffix = "FINAL" if is_final else f"r{server_round}"
        
        return f"{num_clients}c_{attack_info}_{defense_info}_{timestamp}_{round_suffix}"
    
    def _save_checkpoint(self, server_round):
        """Save checkpoint with full pipeline metrics."""
        if self.current_parameters is None:
            return
        
        net = get_model()
        params_arrays = parameters_to_ndarrays(self.current_parameters)
        set_parameters(net, params_arrays)
        
        model_name = self._generate_model_name(server_round)
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "training_time": str(datetime.now() - self.start_time),
            "current_round": server_round,
            "best_accuracy": self.best_accuracy,
            "config": self.config_metadata,
            "accuracy_history": self.accuracy_history,
            "loss_history": self.loss_history,
            "algorithm": "FedProx + Full Pipeline",
        }
        
        # Add full defense stats
        if self.enable_defense:
            metadata["defense_stats"] = {
                "total_tp": self.total_tp,
                "total_fp": self.total_fp,
                "total_fn": self.total_fn,
                "total_tn": self.total_tn,
                "detection_history": self.detection_history,
                "ground_truth": list(self.malicious_clients),
                "avg_detection_rate": (self.total_tp / (self.total_tp + self.total_fn) * 100) if (self.total_tp + self.total_fn) > 0 else 0,
                "avg_fpr": (self.total_fp / (self.total_fp + self.total_tn) * 100) if (self.total_fp + self.total_tn) > 0 else 0,
                "mode_stats": self.mode_controller.get_stats(),
                "reputation_stats": self.reputation_system.get_stats(),
                "noniid_stats": self.noniid_handler.get_stats()
            }
        
        manager = ModelManager(save_dir=self.save_dir)
        manager.save_model(net, metadata, model_name=model_name)
        print(f"[Round {server_round}] 💾 Saved: {model_name}\n")


def server_fn(context: Context) -> ServerAppComponents:
    """Create server with Full Pipeline."""
    
    # Get config
    num_rounds = context.run_config.get("num-server-rounds", 50)
    num_clients = context.run_config.get("num-clients", 30)
    fraction_fit = context.run_config.get("fraction-fit", 0.6)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 0.2)
    
    num_fit_clients = max(1, int(num_clients * fraction_fit))
    num_evaluate_clients = max(1, int(num_clients * fraction_evaluate))
    
    min_fit_clients = context.run_config.get("min-fit-clients", num_fit_clients)
    min_evaluate_clients = context.run_config.get("min-evaluate-clients", num_evaluate_clients)
    min_available_clients = context.run_config.get("min-available-clients", num_clients)
    
    proximal_mu = context.run_config.get("proximal-mu", 0.01)
    auto_save = context.run_config.get("auto-save", True)
    save_dir = context.run_config.get("save-dir", "saved_models")
    save_interval = context.run_config.get("save-interval", 10)
    
    # Defense
    enable_defense = context.run_config.get("enable-defense", True)
    
    # Attack
    attack_type = context.run_config.get("attack-type", "byzantine")
    attack_ratio = context.run_config.get("attack-ratio", 0.3)
    partition_type = context.run_config.get("partition-type", "iid")
    
    # Print config
    print(f"\n{'='*70}")
    print("SERVER CONFIG - FULL DEFENSE PIPELINE")
    print(f"{'='*70}")
    print(f"  Clients: {num_clients} (fit={num_fit_clients}/round)")
    print(f"  Rounds: {num_rounds}")
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
    
    # Get model
    net = get_model()
    
    # Strategy
    strategy = FullPipelineStrategy(
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
        start_round=0,
        enable_defense=enable_defense
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)