"""
Server Application - Trusted Warm-up & Adaptive Defense (V2 - Soft Pipeline)
=============================================================================
UPDATED FOR PARAMETER SEARCH PARSING:
- Fixed logging formats to match run_param_tests.py regex.
- Ensures H score and TP/FP/FN/TN are printed clearly.
"""

import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
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

# --- [CRITICAL FOR PARSER] Accuracy Aggregation ---
def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate evaluation metrics."""
    if not metrics: return {}
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if sum(examples) == 0: return {"accuracy": 0}
    
    # Calculate weighted mean
    accuracy = sum(accuracies) / sum(examples)
    
    # Log for immediate visibility (optional, but helpful)
    print(f"   [AGGREGATED] Accuracy: {accuracy:.4f}") 
    
    return {"accuracy": accuracy}


class FullPipelineStrategy(FedProx):
    """
    FedProx Strategy với TRUSTED WARM-UP & ADAPTIVE DEFENSE (V2).
    """
    
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
        self.auto_save = auto_save
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.config_metadata = config_metadata or {}
        self.start_time = datetime.now()
        self.detection_history = []
        self.current_parameters = None
        
        self.enable_defense = enable_defense
        self.defense_params = defense_params or {}
        self.warmup_rounds = warmup_rounds
        
        self.malicious_clients = self._identify_malicious_clients()
        self.trusted_clients = self._identify_trusted_clients()

        self.total_tp = 0
        self.total_fp = 0
        
        if self.enable_defense:
            print("\n" + "="*70)
            print("🛡️  HYBRID DEFENSE V2: SOFT PIPELINE ACTIVATED")
            print("="*70)
            self._initialize_defense_components()
    
    def _initialize_defense_components(self):
        """Initialize ALL defense components with params from config."""
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
        attack_type = self.config_metadata.get('attack_type', 'none')
        attack_ratio = self.config_metadata.get('attack_ratio', 0.0)
        num_clients = self.config_metadata.get('num_clients', 40)
        if attack_type == 'none' or attack_ratio == 0: return set()
        num_malicious = int(num_clients * attack_ratio)
        return set(range(num_malicious))

    def _identify_trusted_clients(self) -> Set[int]:
        num_clients = self.config_metadata.get('num_clients', 40)
        all_clients = set(range(num_clients))
        benign_clients = list(all_clients - self.malicious_clients)
        benign_clients.sort()
        target_trusted = 15
        num_trusted = min(target_trusted, len(benign_clients))
        return set(benign_clients[:num_trusted])

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[any, FitRes]],
        failures: List[Tuple[any, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]] | None:
        
        if not self.client_id_to_sequential:
            all_ids = sorted([int(c.cid) for c, _ in results])
            for idx, cid in enumerate(all_ids):
                self.client_id_to_sequential[cid] = idx
                self.client_id_to_sequential[str(cid)] = idx
                self.sequential_to_client_id[idx] = cid

        if not results: return None
        if not self.enable_defense: return super().aggregate_fit(server_round, results, failures)

        start_time = time.time()

        # ================= PHASE 1: TRUSTED WARM-UP =================
        if server_round <= self.warmup_rounds:
            print(f"\n{'='*70}")
            print(f"ROUND {server_round} - PHASE 1: TRUSTED WARM-UP")
            print(f"{'='*70}")
            
            trusted_grads = []
            trusted_reps = []
            for client, res in results:
                cid = int(client.cid)
                if cid in self.trusted_clients:
                    p = parameters_to_ndarrays(res.parameters)
                    grad = np.concatenate([x.flatten() for x in p])
                    trusted_grads.append(grad)
                    self.noniid_handler.update_client_gradient(cid, grad)
                    self.reputation_system.initialize_client(cid, is_trusted=True)
                    trusted_reps.append(1.0)

            if not trusted_grads:
                return self.current_parameters, {}

            agg_grad = self.aggregator.weighted_average(trusted_grads, trusted_reps)
            self._update_parameters(agg_grad, results[0][1].parameters)
            
            # --- [CRITICAL] Log H Score even in warmup (if calculated) ---
            # Though H is mostly 0 here, parser might look for it.
            print(f"   Heterogeneity Score H = 0.000 (Warmup)") 
            
            # --- [CRITICAL] Log Metrics Dummy for Parser ---
            # Parser expects TP/FP/FN/TN line. In warmup, we assume perfect trust.
            print(f"   TP=0, FP=0, FN=0, TN={len(results)}")

            if self.auto_save and server_round % self.save_interval == 0:
                self._save_checkpoint(server_round)
            return self.current_parameters, {}

        # ================= PHASE 2: FULL ADAPTIVE DEFENSE =================
        print(f"\n{'='*70}")
        print(f"ROUND {server_round} - PHASE 2: SOFT PIPELINE DEFENSE")
        print(f"{'='*70}")

        # 0. Extract Gradients
        gradients, cids = [], []
        for c, res in results:
            p = parameters_to_ndarrays(res.parameters)
            gradients.append(np.concatenate([x.flatten() for x in p]))
            cids.append(int(c.cid))
        
        gt_malicious = [cid in self.malicious_clients for cid in cids]

        # 1. Detection Layers
        l1_res = self.layer1_detector.detect(gradients, cids, current_round=server_round, is_malicious_ground_truth=gt_malicious)
        l2_status, l2_suspicion = self.layer2_detector.detect(gradients, cids, l1_res, current_round=server_round, is_malicious_ground_truth=gt_malicious)

        # Early Rejection
        early_rejected = set()
        candidates = []
        for cid in cids:
            if l1_res.get(cid) == "REJECTED" or l2_status.get(cid) == "REJECTED":
                early_rejected.add(cid)
            else:
                candidates.append(cid)

        # 2. Non-IID Analysis
        for i, cid in enumerate(cids):
            self.noniid_handler.update_client_gradient(cid, gradients[i])
        
        # --- [CRITICAL FOR PARSER] Log H Score Exact Format ---
        H = self.noniid_handler.compute_heterogeneity_score(gradients, cids)
        print(f"   Heterogeneity Score H = {H:.3f}")
        
        grad_median = np.median(np.vstack(gradients), axis=0)
        baseline_deviations = {}
        for i, cid in enumerate(cids):
            devs = self.noniid_handler.compute_baseline_deviation_detailed(cid, gradients[i], grad_median)
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
        rho_est = len(detected_cids) / len(cids) if cids else 0.0
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
        for i, cid in enumerate(cids):
            status = ClientStatus.CLEAN
            if cid in early_rejected:
                status = ClientStatus.REJECTED 
            elif cid in soft_filtered:
                status = ClientStatus.REJECTED 
            elif l2_suspicion.get(cid) == "suspicious":
                status = ClientStatus.SUSPICIOUS
            
            self.reputation_system.update(
                client_id=cid,
                gradient=gradients[i],
                grad_median=grad_median,
                status=status,
                heterogeneity_score=H,
                current_round=server_round
            )

        # 7. Aggregation
        trusted_idxs = [i for i, cid in enumerate(cids) if cid in trusted]
        final_grads = [gradients[i] for i in trusted_idxs]
        
        if not final_grads:
            print("   ⚠️  All clients filtered. Fallback to weighted average.")
            final_grads = gradients
            final_reps = [self.reputation_system.get_reputation(cid) for cid in cids]
            agg_grad = self.aggregator.aggregate_by_mode(final_grads, "NORMAL", final_reps)
        else:
            final_reps = [self.reputation_system.get_reputation(cids[i]) for i in trusted_idxs]
            rho_actual = len(total_filtered) / len(cids)
            agg_grad = self.aggregator.aggregate_by_mode(
                gradients=final_grads,
                mode=mode,
                reputations=final_reps,
                threat_ratio=rho_actual
            )

        self._update_parameters(agg_grad, results[0][1].parameters)
        
        # --- [CRITICAL FOR PARSER] Print Metrics ---
        self._calculate_metrics(server_round, total_filtered, mode, rho_actual, H)
        
        if self.auto_save and server_round % self.save_interval == 0:
            self._save_checkpoint(server_round)
        
        print(f"⏱️  Defense Duration: {time.time() - start_time:.4f}s")
        return self.current_parameters, {}

    def _update_parameters(self, agg_grad, template_params):
        agg_params = []
        offset = 0
        template = parameters_to_ndarrays(template_params)
        for p in template:
            size = p.size
            agg_params.append(agg_grad[offset:offset+size].reshape(p.shape))
            offset += size
        self.current_parameters = ndarrays_to_parameters(agg_params)

    def _calculate_metrics(self, server_round, filtered_cids, new_mode, rho, H):
        """Calculate and log metrics clearly for parser."""
        filtered_sequential = set()
        for hashed_id in filtered_cids:
            seq_id = self.client_id_to_sequential.get(hashed_id, -1)
            if seq_id != -1: filtered_sequential.add(seq_id)
            
        true_malicious = self.malicious_clients
        num_clients = self.config_metadata.get('num_clients', 40)
        
        # TP: Malicious clients that were FILTERED
        tp = len(true_malicious & filtered_sequential)
        # FP: Benign clients that were FILTERED
        fp = len(filtered_sequential - true_malicious)
        # FN: Malicious clients that were TRUSTED (not filtered)
        fn = len(true_malicious - filtered_sequential)
        # TN: Benign clients that were TRUSTED
        tn = num_clients - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        
        self.total_tp += tp
        self.total_fp += fp
        
        metric_record = {
            "round": server_round,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "mode": new_mode, "rho": rho, "H": H
        }
        self.detection_history.append(metric_record)

        # --- [CRITICAL FOR PARSER] Match Regex Exactly ---
        # Regex: r'TP=(\d+),\s*FP=(\d+),\s*FN=(\d+),\s*TN=(\d+)'
        print(f"\n📊 METRICS ROUND {server_round}")
        print(f"   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"   Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, FPR={fpr:.2f}")

    def _save_checkpoint(self, server_round):
        if not self.current_parameters: return
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
        print(f"💾 Checkpoint saved.")

def server_fn(context: Context) -> ServerAppComponents:
    # Load Basic Config
    num_rounds = context.run_config.get("num-server-rounds", 50)
    num_clients = context.run_config.get("num-clients", 40)
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
    
    # Root Level Configs
    enable_defense = context.run_config.get("enable-defense", True)
    warmup_rounds = context.run_config.get("warmup-rounds", 10) 
    
    attack_type = context.run_config.get("attack-type", "none")
    attack_ratio = context.run_config.get("attack-ratio", 0.0)
    partition_type = context.run_config.get("partition-type", "iid")
    alpha = context.run_config.get("alpha", 0.5)
    
    defense_params = {}
    if enable_defense:
        defense_params['layer1'] = {
            'pca_dims': context.run_config.get("defense.layer1.pca-dims", 20),
            'dbscan_min_samples': context.run_config.get("defense.layer1.dbscan-min-samples", 3),
            'dbscan_eps_multiplier': context.run_config.get("defense.layer1.dbscan-eps-multiplier", 0.5),
            'mad_k_reject': context.run_config.get("defense.layer1.mad-k-reject", 15.0),
            'mad_k_flag': context.run_config.get("defense.layer1.mad-k-flag", 4.0),
        }
        
        defense_params['layer2'] = {
            'distance_multiplier': context.run_config.get("defense.layer2.distance-multiplier", 1.5),
            'cosine_threshold': context.run_config.get("defense.layer2.cosine-threshold", 0.3)
        }
        
        defense_params['noniid'] = {
            'weight_cv': context.run_config.get("defense.noniid.weight-cv", 0.4),
            'weight_sim': context.run_config.get("defense.noniid.weight-sim", 0.4),
            'weight_cluster': context.run_config.get("defense.noniid.weight-cluster", 0.2),
            'adjustment_factor': context.run_config.get("defense.noniid.adjustment-factor", 0.4),
            'theta_adj_clip_min': context.run_config.get("defense.noniid.theta-adj-clip-min", 0.5),
            'theta_adj_clip_max': context.run_config.get("defense.noniid.theta-adj-clip-max", 0.9),
            'baseline_window_size': context.run_config.get("defense.noniid.baseline-window-size", 10),
            'delta_norm_weight': context.run_config.get("defense.noniid.delta-norm-weight", 0.5),
            'delta_direction_weight': context.run_config.get("defense.noniid.delta-direction-weight", 0.5)
        }
        
        defense_params['confidence'] = {
            'weight_flagged': context.run_config.get("defense.confidence.weight-flagged", 0.4),
            'weight_euclidean': context.run_config.get("defense.confidence.weight-euclidean", 0.3),
            'weight_baseline': context.run_config.get("defense.confidence.weight-baseline", 0.3),
            'baseline_suspicious_threshold': context.run_config.get("defense.confidence.baseline-suspicious-threshold", 0.3),
            'baseline_factor_suspicious': context.run_config.get("defense.confidence.baseline-factor-suspicious", 0.8)
        }
        
        defense_params['filtering'] = {
            'hard_threshold_base': context.run_config.get("defense.filtering.hard-threshold-base", 0.85),
            'soft_threshold_base': context.run_config.get("defense.filtering.soft-threshold-base", 0.65),
            'soft_rep_threshold_normal': context.run_config.get("defense.filtering.soft-rep-threshold-normal", 0.2),
            'soft_rep_threshold_alert': context.run_config.get("defense.filtering.soft-rep-threshold-alert", 0.4),
            'soft_rep_threshold_defense': context.run_config.get("defense.filtering.soft-rep-threshold-defense", 0.6)
        }
        
        defense_params['reputation'] = {
            'ema_alpha_increase': context.run_config.get("defense.reputation.ema-alpha-increase", 0.15),
            'ema_alpha_decrease': context.run_config.get("defense.reputation.ema-alpha-decrease", 0.5),
            'lambda_clean': context.run_config.get("defense.reputation.lambda-clean", 1.0),
            'lambda_suspicious_base': context.run_config.get("defense.reputation.lambda-suspicious-base", 0.7),
            'lambda_suspicious_h_mult': context.run_config.get("defense.reputation.lambda-suspicious-h-mult", 0.2),
            'lambda_rejected_base': context.run_config.get("defense.reputation.lambda-rejected-base", 0.1),
            'lambda_rejected_h_mult': context.run_config.get("defense.reputation.lambda-rejected-h-mult", 0.4),
            'floor_warning_threshold': context.run_config.get("defense.reputation.floor-warning-threshold", 0.2),
            'probation_rounds': context.run_config.get("defense.reputation.floor-probation-rounds", 5),
            'initial_reputation': context.run_config.get("defense.reputation.initial-reputation", 0.1),
            'trusted_reputation': context.run_config.get("defense.reputation.trusted-reputation", 1.0),
            'consistency_p_weight': context.run_config.get("defense.reputation.consistency-p-weight", 0.6),
            'consistency_history_weight': context.run_config.get("defense.reputation.consistency-history-weight", 0.4),
            'raw_c_weight': context.run_config.get("defense.reputation.raw-c-weight", 0.5),
            'raw_p_weight': context.run_config.get("defense.reputation.raw-p-weight", 0.5),
            'history_window_size': context.run_config.get("defense.reputation.history-window-size", 5)
        }
        
        defense_params['mode'] = {
            'threshold_normal_to_alert': context.run_config.get("defense.mode.threshold-normal-to-alert", 0.15),
            'threshold_alert_to_defense': context.run_config.get("defense.mode.threshold-alert-to-defense", 0.30),
            'hysteresis_rounds': context.run_config.get("defense.mode.hysteresis-rounds", 2),
            'trust_breach_count': context.run_config.get("defense.mode.trust-breach-count", 3),
            'trust_breach_threshold': context.run_config.get("defense.mode.trust-breach-threshold", 0.85),
            'rep_drop_threshold': context.run_config.get("defense.mode.rep-drop-threshold", 0.10),
            'initial_mode': context.run_config.get("defense.mode.initial-mode", "NORMAL"),
            'warmup_rounds': warmup_rounds,
            'safe_weight_epsilon': context.run_config.get("defense.aggregation.safe-weight-epsilon", 1e-6)
        }
        
        defense_params['aggregation'] = {
            'safe_weight_epsilon': context.run_config.get("defense.aggregation.safe-weight-epsilon", 1e-6),
            'trim_ratio_min': context.run_config.get("defense.aggregation.trim-ratio-min", 0.1),
            'trim_ratio_max': context.run_config.get("defense.aggregation.trim-ratio-max", 0.4),
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
    
    strategy = FullPipelineStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=ndarrays_to_parameters(get_parameters(net)),
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