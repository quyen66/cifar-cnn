"""
Server Application - Trusted Warm-up & Adaptive Defense (V2 - Soft Pipeline)
=============================================================================
FULL PRODUCTION VERSION:
- Logic: Full Model Detection + Historical Momentum.
- Fixes: Metrics Calculation, Momentum Update, ID Mapping.
"""

import numpy as np
import time
import os
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    FitIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedProx
from flwr.common.context import Context

from cifar_cnn.task import get_model, get_parameters, set_parameters
from cifar_cnn.model_manager import ModelManager

# Import defense components
from cifar_cnn.defense import (
    Layer1Detector,
    Layer2Detector,
    NonIIDHandler,
    ConfidenceScorer,    
    TwoStageFilter,
    ReputationSystem,
    ModeController,
    Aggregator           
)
from cifar_cnn.defense.reputation import ClientStatus 

def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate evaluation metrics."""
    if not metrics: return {}
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if sum(examples) == 0: return {"accuracy": 0}
    
    accuracy = sum(accuracies) / sum(examples)
    print(f"   [AGGREGATED] Accuracy: {accuracy:.4f}") 
    return {"accuracy": accuracy}


class FullPipelineStrategy(FedProx):
    
    def __init__(self, *args,
                 auto_save=True,
                 save_dir="saved_models",
                 save_interval=10,
                 config_metadata=None,
                 start_round=0,
                 enable_defense=False,
                 defense_params=None,
                 warmup_rounds=10, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.client_id_to_sequential = {}
        self.sequential_to_client_id = {}
        
        self.auto_save = bool(auto_save)
        self.save_dir = str(save_dir)
        try:
            self.save_interval = int(save_interval)
        except:
            self.save_interval = 10
            
        self.config_metadata = config_metadata or {}
        self.start_time = datetime.now()
        self.detection_history = []
        self.current_parameters = kwargs.get('initial_parameters')
        
        self.enable_defense = bool(enable_defense)
        self.defense_params = defense_params or {}
        self.warmup_rounds = int(warmup_rounds)
        
        # Chúng ta sẽ dùng logic động (Metrics) để xác định malicious, 
        # nhưng vẫn giữ hàm này để xác định trusted cho warmup
        self.malicious_clients = self._identify_malicious_clients()
        self.trusted_clients = self._identify_trusted_clients()

        self.total_tp = 0
        self.total_fp = 0
        
        self.previous_full_grad = None
        
        if self.enable_defense:
            print("\n" + "="*70)
            print("🛡️  HYBRID DEFENSE V2: SOFT PIPELINE ACTIVATED (FULL MODEL MODE)")
            print(f"   Save Interval: {self.save_interval} rounds")
            print(f"   Auto Save: {self.auto_save}")
            print("="*70)
            self._initialize_defense_components()
    
    def _initialize_defense_components(self):
        self.layer1_detector = Layer1Detector(**self.defense_params.get('layer1', {}))
        self.layer2_detector = Layer2Detector(**self.defense_params.get('layer2', {}))
        self.noniid_handler = NonIIDHandler(**self.defense_params.get('noniid', {}))
        self.confidence_scorer = ConfidenceScorer(**self.defense_params.get('confidence', {}))
        self.two_stage_filter = TwoStageFilter(**self.defense_params.get('filtering', {}))
        self.mode_controller = ModeController(**self.defense_params.get('mode', {}))
        
        rep_params = self.defense_params.get('reputation', {})
        self.reputation_system = ReputationSystem(
            ema_alpha_increase=rep_params.get('ema_alpha_increase', 0.15),
            ema_alpha_decrease=rep_params.get('ema_alpha_decrease', 0.5),
            lambda_clean=rep_params.get('lambda_clean', 1.0),
            lambda_suspicious_base=rep_params.get('lambda_suspicious_base', 0.7),
            lambda_suspicious_h_mult=rep_params.get('lambda_suspicious_h_mult', 0.2),
            lambda_rejected_base=rep_params.get('lambda_rejected_base', 0.1),
            lambda_rejected_h_mult=rep_params.get('lambda_rejected_h_mult', 0.4),
            floor_warning_threshold=rep_params.get('floor_warning_threshold', 0.2),
            probation_rounds=rep_params.get('probation_rounds', 5),
            initial_reputation=rep_params.get('initial_reputation', 0.1),
            trusted_reputation=rep_params.get('trusted_reputation', 1.0),
            consistency_p_weight=rep_params.get('consistency_p_weight', 0.6),
            consistency_history_weight=rep_params.get('consistency_history_weight', 0.4),
            raw_c_weight=rep_params.get('raw_c_weight', 0.5),
            raw_p_weight=rep_params.get('raw_p_weight', 0.5),
            history_window_size=rep_params.get('history_window_size', 5)
        )
        
        self.aggregator = Aggregator(**self.defense_params.get('aggregation', {}))
    
    def _identify_malicious_clients(self) -> Set[int]:
        # Logic này chỉ mang tính ước lượng ban đầu
        # Ground Truth chính xác sẽ được lấy từ Metrics của Client gửi về
        return set()

    def _identify_trusted_clients(self) -> Set[int]:
        # Logic chọn Trusted Clients cho Warmup Phase
        num_clients = self.config_metadata.get('num_clients', 40)
        attack_ratio = self.config_metadata.get('attack_ratio', 0.0)
        num_malicious = int(num_clients * attack_ratio)
        
        # Trong Simulation, partition_id < num_malicious là attacker.
        # Nên Trusted Clients an toàn là các ID > num_malicious.
        # Lưu ý: Đây là giả định Partition ID khớp với thứ tự client kết nối.
        # Nếu Mapping sai, có thể lọt attacker vào warmup (nhưng xác suất thấp nếu dùng logic mới).
        all_clients = list(range(num_clients))
        benign_candidates = all_clients[num_malicious:]
        
        target_trusted = 24
        num_trusted = min(target_trusted, len(benign_candidates))
        return set(benign_candidates[:num_trusted])

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        # --- GIAI ĐOẠN 1: FORCE TRUSTED WARM-UP ---
        if self.enable_defense and server_round <= self.warmup_rounds:
            print(f"\n⚡ [WARM-UP CONFIG] Round {server_round}: Forcing Trusted Clients.")
            
            all_clients_dict = client_manager.all()
            all_real_cids = sorted(list(all_clients_dict.keys()))
            
            print(f"   🔍 [DEBUG] Connected Clients: {len(all_real_cids)}")
            if not all_real_cids:
                return [] 
            
            config = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)
            fit_ins = FitIns(parameters, config)

            target_clients = []
            
            for logical_idx in self.trusted_clients:
                if logical_idx < len(all_real_cids):
                    real_cid = all_real_cids[logical_idx]
                    client_proxy = all_clients_dict[real_cid]
                    target_clients.append((client_proxy, fit_ins))
            
            if not target_clients:
                print(f"   ⚠️  Warning: Mapping thất bại. Activating Fallback.")
                fallback_count = min(10, len(all_real_cids))
                for i in range(fallback_count):
                    real_cid = all_real_cids[i]
                    client_proxy = all_clients_dict[real_cid]
                    target_clients.append((client_proxy, fit_ins))
            
            print(f"   ✅ Đã chọn {len(target_clients)} clients cho Warm-up.")
            return target_clients

        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[any, FitRes]],
        failures: List[Tuple[any, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]] | None:
        
        # --- BƯỚC 1: XÂY DỰNG MAP ID ---
        if not self.client_id_to_sequential:
            all_real_ids_strs = sorted([c.cid for c, _ in results])
            for idx, cid_str in enumerate(all_real_ids_strs):
                self.client_id_to_sequential[cid_str] = idx
                self.sequential_to_client_id[idx] = cid_str
            print(f"   🗺️  ID Mapping Created for {len(all_real_ids_strs)} clients.")

        if not results: return None
        should_save = self.auto_save and (server_round % self.save_interval == 0)

        if not self.enable_defense: 
            res = super().aggregate_fit(server_round, results, failures)
            if should_save: self._save_checkpoint(server_round)
            return res

        start_time = time.time()

        # ================= PHASE 1: TRUSTED WARM-UP =================
        if server_round <= self.warmup_rounds:
            print(f"\n{'='*70}")
            print(f"ROUND {server_round} - PHASE 1: TRUSTED WARM-UP")
            print(f"{'='*70}")
            
            trusted_grads = []
            trusted_reps = []
            
            for client, res in results:
                real_cid_str = client.cid
                seq_id = self.client_id_to_sequential.get(real_cid_str, -1)
                
                if seq_id in self.trusted_clients:
                    p = parameters_to_ndarrays(res.parameters)
                    grad = np.concatenate([x.flatten() for x in p])
                    trusted_grads.append(grad)
                    
                    self.noniid_handler.update_client_gradient(seq_id, grad) 
                    self.reputation_system.initialize_client(seq_id, is_trusted=True)
                    trusted_reps.append(1.0)

            if not trusted_grads:
                print("   ⚠️  No trusted clients sampled in Warm-up. Skipping update.")
                if should_save: self._save_checkpoint(server_round) 
                return self.current_parameters, {}

            agg_grad = self.aggregator.weighted_average(trusted_grads, trusted_reps)
            self._update_parameters(agg_grad, results[0][1].parameters)
            
            # Save Momentum from Warmup
            try:
                if agg_grad is not None:
                    if isinstance(agg_grad, list):
                         self.previous_full_grad = np.concatenate([x.flatten() for x in agg_grad])
                    else:
                         self.previous_full_grad = agg_grad.flatten()
            except Exception as e:
                print(f"   ⚠️ Warning: Momentum update failed in Warmup: {e}")
            
            print(f"   Heterogeneity Score H = 0.000 (Warmup)") 
            # Dummy metrics call to populate CSV
            self._calculate_metrics(server_round, [], [], [], 0.0, 0.0, 0.0)
            if should_save: self._save_checkpoint(server_round)
            return self.current_parameters, {}

        # ================= PHASE 2: FULL ADAPTIVE DEFENSE =================
        print(f"\n{'='*70}")
        print(f"ROUND {server_round} - PHASE 2: SOFT PIPELINE DEFENSE (FULL MODEL)")
        print(f"{'='*70}")

        full_gradients = []       
        seq_cids = []
        gt_malicious_batch = []
        
        for c, res in results:
            p = parameters_to_ndarrays(res.parameters)
            full_flat = np.concatenate([x.flatten() for x in p])
            full_gradients.append(full_flat)
            
            real_cid_str = c.cid
            seq_id = self.client_id_to_sequential.get(real_cid_str, -1)
            
            if seq_id == -1:
                seq_id = len(self.client_id_to_sequential)
                self.client_id_to_sequential[real_cid_str] = seq_id
            
            seq_cids.append(seq_id)
            
            # --- [METRICS FIX] Get flag from Client Metrics ---
            is_mal = bool(res.metrics.get("is_malicious", 0))
            gt_malicious_batch.append(is_mal)
            # --------------------------------------------------

        # Debug GT
        num_mal = sum(gt_malicious_batch)
        print(f"   🔍 [DEBUG] Ground Truth from Metrics: {num_mal} attackers in batch.")

        if self.previous_full_grad is not None:
            reference_vector = self.previous_full_grad
            print("   🛡️  Using Historical Momentum (Full Model) as Reference")
        else:
            reference_vector = np.median(np.vstack(full_gradients), axis=0)
            print("   ℹ️  No History. Fallback to Current Round Median (Full Model).")
        
        # 1. Detection Layers
        l1_res = self.layer1_detector.detect(full_gradients, seq_cids, current_round=server_round, is_malicious_ground_truth=gt_malicious_batch)
        
        l2_status, l2_suspicion = self.layer2_detector.detect(
            full_gradients, 
            seq_cids, 
            l1_res, 
            current_round=server_round, 
            is_malicious_ground_truth=gt_malicious_batch, 
            external_reference=reference_vector
        )

        early_rejected = set()
        candidates = []
        for cid in seq_cids:
            if l1_res.get(cid) == "REJECTED" or l2_status.get(cid) == "REJECTED":
                early_rejected.add(cid)
            else:
                candidates.append(cid)

        # 2. Non-IID Analysis
        for i, cid in enumerate(seq_cids):
            self.noniid_handler.update_client_gradient(cid, full_gradients[i])
        
        H = self.noniid_handler.compute_heterogeneity_score(full_gradients, seq_cids)
        print(f"   Heterogeneity Score H = {H:.3f}")
        
        baseline_deviations = {}
        for i, cid in enumerate(seq_cids):
            devs = self.noniid_handler.compute_baseline_deviation_detailed(cid, full_gradients[i], reference_vector)
            baseline_deviations[cid] = devs['delta_combined']

        # 3. Confidence Scoring
        if candidates:
            conf_scores = self.confidence_scorer.calculate_scores(
                client_ids=candidates,
                layer1_results=l1_res,
                suspicion_levels=l2_suspicion,
                baseline_deviations=baseline_deviations
            )
        else:
            conf_scores = {}

        # 4. Mode Control
        detected_cids = list(early_rejected)
        rho_est = len(detected_cids) / len(seq_cids) if seq_cids else 0.0
        current_reps = self.reputation_system.get_all_reputations()
        mode = self.mode_controller.update_mode(rho_est, detected_cids, current_reps, server_round)
        
        print(f"   Mode: {mode} (Est. Threat Ratio: {rho_est:.2%})")

        # 5. Filtering
        if candidates:
            trusted, soft_filtered, f_stats = self.two_stage_filter.filter_clients(
                client_ids=candidates,
                confidence_scores=conf_scores,
                reputations=current_reps,
                mode=mode,
                H=H,
                noniid_handler=self.noniid_handler
            )
        else:
            trusted, soft_filtered = set(), set()
            
        total_filtered = early_rejected | soft_filtered
        
        print(f"   Trusted: {len(trusted)}")
        print(f"   Filtered Total: {len(total_filtered)} (Early: {len(early_rejected)} + Soft: {len(soft_filtered)})")

        # 6. Reputation Update
        for i, cid in enumerate(seq_cids):
            status = ClientStatus.CLEAN
            if cid in early_rejected:
                status = ClientStatus.REJECTED 
            elif cid in soft_filtered:
                status = ClientStatus.REJECTED 
            elif l2_suspicion.get(cid) == "suspicious":
                status = ClientStatus.SUSPICIOUS
            
            self.reputation_system.update(
                client_id=cid,
                gradient=full_gradients[i],
                grad_median=reference_vector,
                status=status,
                heterogeneity_score=H,
                current_round=server_round
            )

        # 7. Aggregation
        trusted_idxs = [i for i, cid in enumerate(seq_cids) if cid in trusted]
        final_grads = [full_gradients[i] for i in trusted_idxs]
        
        rho_actual = len(total_filtered) / len(seq_cids) if seq_cids else 0.0
        
        if not final_grads:
            print("   ⚠️  ALL CLIENTS FILTERED. Fallback to Reputation-Weighted Average.")
            final_grads = full_gradients
            final_reps = [self.reputation_system.get_reputation(cid) for cid in seq_cids]
            agg_grad = self.aggregator.aggregate_by_mode(final_grads, "NORMAL", final_reps)
        else:
            final_reps = [self.reputation_system.get_reputation(seq_cids[i]) for i in trusted_idxs]
            agg_grad = self.aggregator.aggregate_by_mode(
                gradients=final_grads,
                mode=mode,
                reputations=final_reps,
                threat_ratio=rho_actual
            )

        try:
            if agg_grad is not None:
                if isinstance(agg_grad, list):
                     self.previous_full_grad = np.concatenate([x.flatten() for x in agg_grad])
                else:
                     self.previous_full_grad = agg_grad.flatten()
        except Exception as e:
            print(f"   ⚠️ Warning: Momentum update failed (Ignored): {e}")
        
        self._update_parameters(agg_grad, results[0][1].parameters)
        
        # Calculate final metrics for CSV
        self._calculate_metrics(server_round, total_filtered, seq_cids, gt_malicious_batch, mode, rho_actual, H)
        
        if should_save: self._save_checkpoint(server_round)
        
        print(f"⏱️  Defense Duration: {time.time() - start_time:.4f}s")
        return self.current_parameters, {}

    def _update_parameters(self, agg_grad, template_params):
        agg_params = []
        offset = 0
        template = parameters_to_ndarrays(template_params)
        
        if isinstance(agg_grad, np.ndarray):
             for p in template:
                size = p.size
                agg_params.append(agg_grad[offset:offset+size].reshape(p.shape))
                offset += size
        else:
            agg_params = agg_grad

        self.current_parameters = ndarrays_to_parameters(agg_params)

    def _calculate_metrics(self, server_round, filtered_cids, batch_seq_ids, batch_gt, new_mode, rho, H):
        if not batch_seq_ids:
            return

        tp, fp, fn, tn = 0, 0, 0, 0
        
        for i, seq_id in enumerate(batch_seq_ids):
            is_attacker = batch_gt[i]
            is_filtered = seq_id in filtered_cids
            
            if is_attacker and is_filtered: tp += 1
            elif not is_attacker and is_filtered: fp += 1
            elif is_attacker and not is_filtered: fn += 1
            elif not is_attacker and not is_filtered: tn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # FPR = FP / (FP + TN)
        benign_count = fp + tn
        fpr = fp / benign_count if benign_count > 0 else 0.0
        
        self.total_tp += tp
        self.total_fp += fp
        
        metric_record = {
            "round": int(server_round),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(precision), "recall": float(recall), "f1": float(f1), "fpr": float(fpr),
            "mode": str(new_mode), "rho": float(rho), "H": float(H)
        }
        self.detection_history.append(metric_record)

        print(f"\n📊 METRICS ROUND {server_round}")
        print(f"   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"   Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, FPR={fpr:.2f}")

    def _save_checkpoint(self, server_round):
        if not self.current_parameters: return
        try:
            print(f"💾 Saving checkpoint for Round {server_round}...")
            net = get_model()
            set_parameters(net, parameters_to_ndarrays(self.current_parameters))
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "round": server_round,
                "config": self.config_metadata,
                "metrics_history": self.detection_history
            }
            ModelManager(save_dir=self.save_dir).save_model(
                net, metadata, model_name=f"model_r{server_round}"
            )
            print(f"✅ Checkpoint saved.")
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = int(context.run_config.get("num-server-rounds", 50))
    num_clients = int(context.run_config.get("num-clients", 40))
    fraction_fit = float(context.run_config.get("fraction-fit", 0.6))
    fraction_evaluate = float(context.run_config.get("fraction-evaluate", 0.2))
    min_fit_clients = int(context.run_config.get("min-fit-clients", 23))
    min_evaluate_clients = int(context.run_config.get("min-evaluate-clients", 6))
    min_available_clients = int(context.run_config.get("min-available-clients", 40))
    proximal_mu = float(context.run_config.get("proximal-mu", 0.01))
    
    auto_save = context.run_config.get("auto-save", True) 
    if isinstance(auto_save, str): auto_save = auto_save.lower() == 'true'
        
    save_dir = context.run_config.get("save-dir", "saved_models")
    save_interval = int(context.run_config.get("save-interval", 10))
    resume_from = context.run_config.get("resume-from", "")

    enable_defense = context.run_config.get("enable-defense", True)
    if isinstance(enable_defense, str): enable_defense = enable_defense.lower() == 'true'
        
    warmup_rounds = int(context.run_config.get("warmup-rounds", 10))
    attack_type = context.run_config.get("attack-type", "none")
    attack_ratio = float(context.run_config.get("attack-ratio", 0.0))
    partition_type = context.run_config.get("partition-type", "iid")
    alpha = float(context.run_config.get("alpha", 0.5))
    
    defense_params = {}
    if enable_defense:
        defense_params['layer1'] = {
            'pca_dims': int(context.run_config.get("defense.layer1.pca-dims", 20)),
            'dbscan_min_samples': int(context.run_config.get("defense.layer1.dbscan-min-samples", 3)),
            'dbscan_eps_multiplier': float(context.run_config.get("defense.layer1.dbscan-eps-multiplier", 0.5)),
            'mad_k_reject': float(context.run_config.get("defense.layer1.mad-k-reject", 15.0)),
            'mad_k_flag': float(context.run_config.get("defense.layer1.mad-k-flag", 4.0)),
        }
        defense_params['layer2'] = {
            'distance_multiplier': float(context.run_config.get("defense.layer2.distance-multiplier", 1.5)),
            'cosine_threshold': float(context.run_config.get("defense.layer2.cosine-threshold", 0.3))
        }
        defense_params['noniid'] = {
            'weight_cv': float(context.run_config.get("defense.noniid.weight-cv", 0.4)),
            'weight_sim': float(context.run_config.get("defense.noniid.weight-sim", 0.4)),
            'weight_cluster': float(context.run_config.get("defense.noniid.weight-cluster", 0.2)),
            'adjustment_factor': float(context.run_config.get("defense.noniid.adjustment-factor", 0.4)),
            'theta_adj_clip_min': float(context.run_config.get("defense.noniid.theta-adj-clip-min", 0.5)),
            'theta_adj_clip_max': float(context.run_config.get("defense.noniid.theta-adj-clip-max", 0.9)),
            'baseline_window_size': int(context.run_config.get("defense.noniid.baseline-window-size", 10)),
            'delta_norm_weight': float(context.run_config.get("defense.noniid.delta-norm-weight", 0.5)),
            'delta_direction_weight': float(context.run_config.get("defense.noniid.delta-direction-weight", 0.5))
        }
        defense_params['confidence'] = {
            'weight_flagged': float(context.run_config.get("defense.confidence.weight-flagged", 0.4)),
            'weight_euclidean': float(context.run_config.get("defense.confidence.weight-euclidean", 0.3)),
            'weight_baseline': float(context.run_config.get("defense.confidence.weight-baseline", 0.3)),
            'baseline_suspicious_threshold': float(context.run_config.get("defense.confidence.baseline-suspicious-threshold", 0.3)),
            'baseline_factor_suspicious': float(context.run_config.get("defense.confidence.baseline-factor-suspicious", 0.8))
        }
        defense_params['filtering'] = {
            'hard_threshold_base': float(context.run_config.get("defense.filtering.hard-threshold-base", 0.85)),
            'soft_threshold_base': float(context.run_config.get("defense.filtering.soft-threshold-base", 0.65)),
            'soft_rep_threshold_normal': float(context.run_config.get("defense.filtering.soft-rep-threshold-normal", 0.2)),
            'soft_rep_threshold_alert': float(context.run_config.get("defense.filtering.soft-rep-threshold-alert", 0.4)),
            'soft_rep_threshold_defense': float(context.run_config.get("defense.filtering.soft-rep-threshold-defense", 0.6))
        }
        defense_params['reputation'] = {
            'ema_alpha_increase': float(context.run_config.get("defense.reputation.ema-alpha-increase", 0.15)),
            'ema_alpha_decrease': float(context.run_config.get("defense.reputation.ema-alpha-decrease", 0.5)),
            'lambda_clean': float(context.run_config.get("defense.reputation.lambda-clean", 1.0)),
            'lambda_suspicious_base': float(context.run_config.get("defense.reputation.lambda-suspicious-base", 0.7)),
            'lambda_suspicious_h_mult': float(context.run_config.get("defense.reputation.lambda-suspicious-h-mult", 0.2)),
            'lambda_rejected_base': float(context.run_config.get("defense.reputation.lambda-rejected-base", 0.1)),
            'lambda_rejected_h_mult': float(context.run_config.get("defense.reputation.lambda-rejected-h-mult", 0.4)),
            'floor_warning_threshold': float(context.run_config.get("defense.reputation.floor-warning-threshold", 0.2)),
            'probation_rounds': int(context.run_config.get("defense.reputation.floor-probation-rounds", 5)),
            'initial_reputation': float(context.run_config.get("defense.reputation.initial-reputation", 0.1)),
            'trusted_reputation': float(context.run_config.get("defense.reputation.trusted-reputation", 1.0)),
            'consistency_p_weight': float(context.run_config.get("defense.reputation.consistency-p-weight", 0.6)),
            'consistency_history_weight': float(context.run_config.get("defense.reputation.consistency-history-weight", 0.4)),
            'raw_c_weight': float(context.run_config.get("defense.reputation.raw-c-weight", 0.5)),
            'raw_p_weight': float(context.run_config.get("defense.reputation.raw-p-weight", 0.5)),
            'history_window_size': int(context.run_config.get("defense.reputation.history-window-size", 5))
        }
        defense_params['mode'] = {
            'threshold_normal_to_alert': float(context.run_config.get("defense.mode.threshold-normal-to-alert", 0.15)),
            'threshold_alert_to_defense': float(context.run_config.get("defense.mode.threshold-alert-to-defense", 0.30)),
            'hysteresis_rounds': int(context.run_config.get("defense.mode.hysteresis-rounds", 2)),
            'trust_breach_count': int(context.run_config.get("defense.mode.trust-breach-count", 3)),
            'trust_breach_threshold': float(context.run_config.get("defense.mode.trust-breach-threshold", 0.85)),
            'rep_drop_threshold': float(context.run_config.get("defense.mode.rep-drop-threshold", 0.10)),
            'initial_mode': context.run_config.get("defense.mode.initial-mode", "NORMAL"),
            'warmup_rounds': warmup_rounds,
            'safe_weight_epsilon': float(context.run_config.get("defense.aggregation.safe-weight-epsilon", 1e-6))
        }
        defense_params['aggregation'] = {
            'safe_weight_epsilon': float(context.run_config.get("defense.aggregation.safe-weight-epsilon", 1e-6)),
            'trim_ratio_min': float(context.run_config.get("defense.aggregation.trim-ratio-min", 0.1)),
            'trim_ratio_max': float(context.run_config.get("defense.aggregation.trim-ratio-max", 0.4)),
            'use_reputation_weights': context.run_config.get("defense.aggregation.use-reputation-weights", True),
            'uniform_weight_fallback': context.run_config.get("defense.aggregation.uniform-weight-fallback", True)
        }
    
    config_metadata = {
        'num_clients': num_clients,
        'partition_type': partition_type,
        'alpha': alpha,
        'attack_type': attack_type,
        'attack_ratio': attack_ratio,
        'enable_defense': enable_defense,
        'proximal_mu': proximal_mu,
    }
    
    net = get_model()
    initial_params = None
    
    if resume_from and os.path.exists(resume_from):
        print(f"\n{'='*70}")
        print(f"🔄 RESUMING FROM CHECKPOINT: {resume_from}")
        print(f"{'='*70}")
        try:
            state_dict = torch.load(resume_from, map_location='cpu')
            net.load_state_dict(state_dict)
            initial_params = ndarrays_to_parameters(get_parameters(net))
            print("✅ Weights loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("⚠️  Starting with fresh weights.")
            initial_params = ndarrays_to_parameters(get_parameters(net))
    else:
        initial_params = ndarrays_to_parameters(get_parameters(net))
    
    strategy = FullPipelineStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_params,
        proximal_mu=proximal_mu,
        auto_save=auto_save,
        save_dir=save_dir,
        save_interval=save_interval,
        config_metadata=config_metadata,
        start_round=0,
        enable_defense=enable_defense,
        defense_params=defense_params,
        warmup_rounds=warmup_rounds
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)
    
app = ServerApp(server_fn=server_fn)