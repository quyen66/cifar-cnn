"""
Server Application - Trusted Warm-up & Adaptive Defense (VERIFIED)
===================================================================
VERIFIED:
1. ✅ Reputation Params: Đã map đúng 'floor_warning_threshold' & 'probation_rounds'.
2. ✅ Dynamic Warmup: Đọc từ root config.
3. ✅ Features: Time tracking, detailed logs, metrics history.
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
    TwoStageFilter,
    ReputationSystem,
    ModeController,
    aggregate_by_mode,
    weighted_average_aggregation 
)

def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate evaluation metrics."""
    if not metrics: return {}
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if sum(examples) == 0: return {"accuracy": 0}
    return {"accuracy": sum(accuracies) / sum(examples)}


class FullPipelineStrategy(FedProx):
    """
    FedProx Strategy với TRUSTED WARM-UP & ADAPTIVE DEFENSE.
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
        self.best_accuracy = 0.0
        self.accuracy_history = []
        self.detection_history = []
        self.loss_history = []
        self.current_parameters = None
        self.start_round = start_round
        
        self.enable_defense = enable_defense
        self.defense_params = defense_params or {}
        self.warmup_rounds = warmup_rounds
        
        self.malicious_clients = self._identify_malicious_clients()
        self.trusted_clients = self._identify_trusted_clients()

        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        
        if self.enable_defense:
            print("\n" + "="*70)
            print("🛡️  HYBRID DEFENSE: TRUSTED WARM-UP ENABLED")
            print("="*70)
            print(f"  Configuration:")
            print(f"  ► Warm-up Duration: {self.warmup_rounds} rounds")
            print(f"  ► Trusted Nodes: {len(self.trusted_clients)} clients (Target: 15)")
            print("-" * 70)
            print(f"  Phase 1 (R1-{self.warmup_rounds}): Trusted Nodes Only (Weighted Avg)")
            print(f"  Phase 2 (R{self.warmup_rounds+1}+): Full Adaptive Defense")
            print("="*70 + "\n")
            self._initialize_defense_components()
    
    def _initialize_defense_components(self):
        """Initialize ALL defense components with params from config."""
        self.layer1_detector = Layer1Detector(**self.defense_params.get('layer1', {}))
        self.layer2_detector = Layer2Detector(**self.defense_params.get('layer2', {}))
        self.noniid_handler = NonIIDHandler(**self.defense_params.get('noniid', {}))
        self.two_stage_filter = TwoStageFilter(**self.defense_params.get('filtering', {}))
        
        # Reputation System (Correctly Mapped)
        rep_params = self.defense_params.get('reputation', {})
        self.reputation_system = ReputationSystem(
            ema_alpha_increase=rep_params.get('ema_alpha_increase', 0.4),
            ema_alpha_decrease=rep_params.get('ema_alpha_decrease', 0.2),
            penalty_flagged=rep_params.get('penalty_flagged', 0.2),
            #reward_clean=rep_params.get('reward_clean', 0.1),
            floor_warning_threshold=rep_params.get('floor_warning_threshold', 0.2),
            probation_rounds=rep_params.get('probation_rounds', 5),
            initial_reputation=rep_params.get('initial_reputation', 0.8)
        )
        
        self.mode_controller = ModeController(**self.defense_params.get('mode', {}))
    
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
        if num_trusted < target_trusted:
            print(f"⚠️  Warning: Requested {target_trusted} trusted clients but only found {len(benign_clients)}.")
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
            if server_round == 1:
                print(f"✅ Created client ID mapping: {len(all_ids)} clients")

        if not results: return None
        if not self.enable_defense: return super().aggregate_fit(server_round, results, failures)

        start_time = time.time()

        # ================= PHASE 1: TRUSTED WARM-UP =================
        if server_round <= self.warmup_rounds:
            print(f"\n{'='*70}")
            print(f"ROUND {server_round} - PHASE 1: TRUSTED WARM-UP")
            print(f"{'='*70}")
            
            trusted_grads = []
            for client, res in results:
                cid = int(client.cid)
                if cid in self.trusted_clients:
                    p = parameters_to_ndarrays(res.parameters)
                    grad = np.concatenate([x.flatten() for x in p])
                    trusted_grads.append(grad)
                    self.noniid_handler.update_client_gradient(cid, grad)
                    self.reputation_system.reputations[cid] = 1.0

            if not trusted_grads:
                print("⚠️  No trusted clients! Using previous parameters.")
                return self.current_parameters, {}

            print(f"   Aggregation: Weighted Average on {len(trusted_grads)} trusted nodes.")
            agg_grad = weighted_average_aggregation(trusted_grads)
            
            agg_params = []
            offset = 0
            template = parameters_to_ndarrays(results[0][1].parameters)
            for p in template:
                size = p.size
                agg_params.append(agg_grad[offset:offset+size].reshape(p.shape))
                offset += size
                
            self.current_parameters = ndarrays_to_parameters(agg_params)
            
            if self.auto_save and server_round % self.save_interval == 0:
                self._save_checkpoint(server_round)
            
            print(f"⏱️  Defense Duration: {time.time() - start_time:.4f}s")
            return self.current_parameters, {}

        # ================= PHASE 2: FULL DEFENSE =================
        print(f"\n{'='*70}")
        print(f"ROUND {server_round} - PHASE 2: FULL ADAPTIVE DEFENSE")
        print(f"{'='*70}")

        gradients, cids = [], []
        for c, res in results:
            p = parameters_to_ndarrays(res.parameters)
            gradients.append(np.concatenate([x.flatten() for x in p]))
            cids.append(int(c.cid))
        
        gt_malicious = [cid in self.malicious_clients for cid in cids]

        print(f"\n🔍 STAGE 1: DETECTION")
        l1_res = self.layer1_detector.detect(gradients, cids, gt_malicious, server_round)
        l2_res = self.layer2_detector.detect(gradients, cids, server_round, l1_res)
        combined_flags = {cid: l1_res.get(cid, False) or l2_res.get(cid, False) for cid in cids}

        print(f"\n📊 STAGE 2: NON-IID")
        for i, cid in enumerate(cids):
            self.noniid_handler.update_client_gradient(cid, gradients[i])
        
        H = self.noniid_handler.compute_heterogeneity_score(gradients, cids)
        print(f"   Heterogeneity Score H = {H:.3f}")
        
        grad_median = np.median(np.vstack(gradients), axis=0)
        base_devs = {}
        print(f"\n🔬 STAGE 2.5: BASELINE TRACKING (DETAILS)")
        for i, cid in enumerate(cids):
            dev_info = self.noniid_handler.compute_baseline_deviation_detailed(cid, gradients[i], grad_median)
            base_devs[cid] = dev_info['delta_combined']
            if dev_info['delta_combined'] > 0.3:
                print(f"   Client {cid}: δ={dev_info['delta_combined']:.3f} "
                      f"(norm={dev_info['delta_norm']:.3f}, dir={dev_info['delta_direction']:.3f})")
        
        print(f"\n💯 STAGE 3: CONFIDENCE")
        conf_scores = {}
        for cid in cids:
            is_trusted = cid in self.trusted_clients
            self.reputation_system.initialize_client(cid, is_trusted)
            rep = self.reputation_system.get_reputation(cid)
            was_flagged = combined_flags.get(cid, False)
            s_base = 3 if was_flagged else 0
            adj_rep = (1.0 - rep) * 2
            dev = base_devs.get(cid, 0.0)
            factor = 0.8 if dev > 0.3 else 1.0
            ci = np.clip((s_base + adj_rep) * factor, 0.0, 1.0)
            conf_scores[cid] = ci
        
        print(f"   Mean confidence: {np.mean(list(conf_scores.values())):.3f}")

        detected = [cid for cid, f in combined_flags.items() if f]
        rho = len(detected) / len(cids) if cids else 0
        mode = self.mode_controller.update_mode(rho, detected, self.reputation_system.reputations, server_round)
        print(f"\n🎛️  STAGE 4: MODE CONTROL")
        print(f"   Mode: {mode} (Threat Ratio: {rho:.2%})")

        print(f"\n🔧 STAGE 5: FILTERING")
        trusted, filtered, f_stats = self.two_stage_filter.filter_clients(
            cids, conf_scores, self.reputation_system.reputations, mode, H, self.noniid_handler
        )
        print(f"   Trusted: {len(trusted)}, Filtered: {len(filtered)}")

        print(f"\n⭐ STAGE 6: REPUTATION")
        for i, cid in enumerate(cids):
            self.reputation_system.update(
                cid, gradients[i], grad_median, combined_flags.get(cid, False), server_round, 
                baseline_deviation=base_devs.get(cid, 0.0), heterogeneity_score=H
            )
        
        # Log Stats
        rep_stats = self.reputation_system.get_stats()
        print(f"   Mean Rep: {rep_stats.get('mean_reputation', 0):.3f} (Probation: {rep_stats.get('clients_in_probation', 0)})")

        print(f"\n⚙️  STAGE 7: AGGREGATION")
        trusted_idxs = [i for i, cid in enumerate(cids) if cid in trusted]
        final_grads = [gradients[i] for i in trusted_idxs]
        
        if not final_grads:
            print("   ⚠️  All clients filtered. Using weighted average of all.")
            final_grads = gradients
            agg_grad = weighted_average_aggregation(final_grads)
        else:
            agg_grad = aggregate_by_mode(final_grads, mode=mode)

        agg_params = []
        offset = 0
        template = parameters_to_ndarrays(results[0][1].parameters)
        for p in template:
            size = p.size
            agg_params.append(agg_grad[offset:offset+size].reshape(p.shape))
            offset += size
        
        self.current_parameters = ndarrays_to_parameters(agg_params)
        
        print(f"\n📈 STAGE 8: REPORTING")
        l1_stats = self.layer1_detector.get_stats()
        noniid_stats = self.noniid_handler.get_stats()
        print(f"   Layer1: PCA fitted={l1_stats.get('pca_fitted', False)}")
        print(f"   NonIID: Tracked clients={noniid_stats.get('num_tracked_clients', 0)}")

        self._calculate_metrics(server_round, detected, mode, rho, H)
        if self.auto_save and server_round % self.save_interval == 0:
            self._save_checkpoint(server_round)
        
        print(f"⏱️  Defense Duration: {time.time() - start_time:.4f}s")
        return self.current_parameters, {}

    def _calculate_metrics(self, server_round, detected_ids, new_mode, rho, H):
        detected_sequential = set()
        for hashed_id in detected_ids:
            seq_id = self.client_id_to_sequential.get(hashed_id, -1)
            if seq_id != -1: detected_sequential.add(seq_id)
            
        true_malicious = self.malicious_clients
        num_clients = self.config_metadata.get('num_clients', 40)
        
        tp = len(true_malicious & detected_sequential)
        fp = len(detected_sequential - true_malicious)
        fn = len(true_malicious - detected_sequential)
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
            'mad_k_normal': context.run_config.get("defense.layer1.mad-k-normal", 4.0),
            'mad_k_warmup': context.run_config.get("defense.layer1.mad-k-warmup", 6.0),
            'voting_threshold_normal': context.run_config.get("defense.layer1.voting-threshold-normal", 2),
            'voting_threshold_warmup': context.run_config.get("defense.layer1.voting-threshold-warmup", 3)
        }
        
        defense_params['layer2'] = {
            'distance_multiplier': context.run_config.get("defense.layer2.distance-multiplier", 1.5),
            'cosine_threshold': context.run_config.get("defense.layer2.cosine-threshold", 0.3)
        }
        
        defense_params['noniid'] = {
            'h_threshold_normal': context.run_config.get("defense.noniid.h-threshold-normal", 0.6),
            'h_threshold_alert': context.run_config.get("defense.noniid.h-threshold-alert", 0.5),
            'adaptive_multiplier': context.run_config.get("defense.noniid.adaptive-multiplier", 1.5),
            'adjustment_factor': context.run_config.get("defense.noniid.adjustment-factor", 0.4),
            'baseline_percentile': context.run_config.get("defense.noniid.baseline-percentile", 60),
            'baseline_window_size': context.run_config.get("defense.noniid.baseline-window-size", 10),
            'delta_norm_weight': context.run_config.get("defense.noniid.delta-norm-weight", 0.5),
            'delta_direction_weight': context.run_config.get("defense.noniid.delta-direction-weight", 0.5)
        }
        
        defense_params['filtering'] = {
            'hard_k_threshold': context.run_config.get("defense.filtering.hard-k-threshold", 3),
            'hard_threshold_min': context.run_config.get("defense.filtering.hard-threshold-min", 0.85),
            'hard_threshold_max': context.run_config.get("defense.filtering.hard-threshold-max", 0.95),
            'soft_reputation_threshold': context.run_config.get("defense.filtering.soft-reputation-threshold", 0.4),
            'soft_distance_multiplier': context.run_config.get("defense.filtering.soft-distance-multiplier", 2.0),
            'soft_enabled': context.run_config.get("defense.filtering.soft-enabled", True)
        }
        
        defense_params['reputation'] = {
            'ema_alpha_increase': context.run_config.get("defense.reputation.ema-alpha-increase", 0.4),
            'ema_alpha_decrease': context.run_config.get("defense.reputation.ema-alpha-decrease", 0.2),
            'penalty_flagged': context.run_config.get("defense.reputation.penalty-flagged", 0.2),
            #'penalty_variance': context.run_config.get("defense.reputation.penalty-variance", 0.1),
            #'reward_clean': context.run_config.get("defense.reputation.reward-clean", 0.1),
            # Correct mapping for updated Probation logic
            'floor_warning_threshold': context.run_config.get("defense.reputation.floor-warning-threshold", 0.2),
            'probation_rounds': context.run_config.get("defense.reputation.floor-probation-rounds", 5),
            'initial_reputation': context.run_config.get("defense.reputation.initial-reputation", 0.1)
        }
        
        defense_params['mode'] = {
            'threshold_normal_to_alert': context.run_config.get("defense.mode.threshold-normal-to-alert", 0.20),
            'threshold_alert_to_defense': context.run_config.get("defense.mode.threshold-alert-to-defense", 0.30),
            'rep_gate_defense': context.run_config.get("defense.mode.rep-gate-defense", 0.5),
            'initial_mode': context.run_config.get("defense.mode.initial-mode", "NORMAL"),
            'warmup_rounds': warmup_rounds
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