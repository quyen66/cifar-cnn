"""
Server Application - Full Defense Pipeline
===========================================
Load ALL parameters from pyproject.toml và pass xuống defense components.

KHÔNG CÓ HARDCODED VALUES - TẤT CẢ LOAD TỪ CONFIG!
"""

import numpy as np
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
#from cifar_cnn.utils import ModelManager

# Import ALL defense components
from cifar_cnn.defense import (
    Layer1Detector,
    Layer2Detector,
    NonIIDHandler,
    TwoStageFilter,
    ReputationSystem,
    ModeController,
    aggregate_by_mode
)


def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate evaluation metrics."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


class FullPipelineStrategy(FedProx):
    """
    FedProx Strategy với FULL DEFENSE PIPELINE.
    
    TẤT CẢ PARAMETERS ĐƯỢC LOAD TỪ pyproject.toml!
    """
    
    def __init__(self, *args,
                 # Model saving
                 auto_save=True,
                 save_dir="saved_models",
                 save_interval=10,
                 config_metadata=None,
                 start_round=0,
                 # Defense
                 enable_defense=False,
                 defense_params=None,  # NEW: Dict chứa ALL defense params
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
        self.defense_params = defense_params or {}
        
        # Ground truth for metrics
        self.malicious_clients = self._identify_malicious_clients()
        
        if self.enable_defense:
            print("\n" + "="*70)
            print("🛡️  FULL DEFENSE PIPELINE ENABLED")
            print("="*70)
            print("  ✓ Layer 1: Enhanced DBSCAN Detection")
            print("  ✓ Layer 2: Distance + Direction Detection")
            print("  ✓ Non-IID Handler")
            print("  ✓ Two-Stage Filtering")
            print("  ✓ Reputation System")
            print("  ✓ Mode Controller")
            print("  ✓ Mode-Adaptive Aggregation")
            print("="*70)
            if self.malicious_clients:
                print(f"  🎯 Ground Truth: {len(self.malicious_clients)} malicious clients")
                print(f"     IDs: {sorted(list(self.malicious_clients))}")
            print("="*70 + "\n")
            
            # Initialize ALL components với params từ config
            self._initialize_defense_components()
            
            # Track detection stats
            self.detection_history = []
            self.total_tp = 0
            self.total_fp = 0
            self.total_fn = 0
            self.total_tn = 0
    
    def _initialize_defense_components(self):
        """Initialize ALL defense components với params từ config."""
        
        # Layer 1 params
        layer1_params = self.defense_params.get('layer1', {})
        self.layer1_detector = Layer1Detector(
            pca_dims=layer1_params.get('pca_dims', 20),
            dbscan_min_samples=layer1_params.get('dbscan_min_samples', 3),
            dbscan_eps_multiplier=layer1_params.get('dbscan_eps_multiplier', 0.5),
            mad_k_normal=layer1_params.get('mad_k_normal', 4.0),
            mad_k_warmup=layer1_params.get('mad_k_warmup', 6.0),
            voting_threshold_normal=layer1_params.get('voting_threshold_normal', 2),
            voting_threshold_warmup=layer1_params.get('voting_threshold_warmup', 3),
            warmup_rounds=layer1_params.get('warmup_rounds', 10)
        )
        
        # Layer 2 params
        layer2_params = self.defense_params.get('layer2', {})
        self.layer2_detector = Layer2Detector(
            distance_multiplier=layer2_params.get('distance_multiplier', 1.5),
            cosine_threshold=layer2_params.get('cosine_threshold', 0.3),
            warmup_rounds=layer2_params.get('warmup_rounds', 15)
        )
        
        # Non-IID params
        noniid_params = self.defense_params.get('noniid', {})
        self.noniid_handler = NonIIDHandler(
            h_threshold_normal=noniid_params.get('h_threshold_normal', 0.6),
            h_threshold_alert=noniid_params.get('h_threshold_alert', 0.5),
            adaptive_multiplier=noniid_params.get('adaptive_multiplier', 1.5),
            baseline_percentile=noniid_params.get('baseline_percentile', 60),
            baseline_window_size=noniid_params.get('baseline_window_size', 10)
        )
        
        # Filtering params
        filtering_params = self.defense_params.get('filtering', {})
        self.two_stage_filter = TwoStageFilter(
            hard_k_threshold=filtering_params.get('hard_k_threshold', 3),
            soft_reputation_threshold=filtering_params.get('soft_reputation_threshold', 0.4),
            soft_distance_multiplier=filtering_params.get('soft_distance_multiplier', 2.0),
            soft_enabled=filtering_params.get('soft_enabled', True)
        )
        
        # Reputation params
        reputation_params = self.defense_params.get('reputation', {})
        self.reputation_system = ReputationSystem(
            ema_alpha_increase=reputation_params.get('ema_alpha_increase', 0.4),
            ema_alpha_decrease=reputation_params.get('ema_alpha_decrease', 0.2),
            penalty_flagged=reputation_params.get('penalty_flagged', 0.2),
            penalty_variance=reputation_params.get('penalty_variance', 0.1),
            reward_clean=reputation_params.get('reward_clean', 0.1),
            floor_lift_threshold=reputation_params.get('floor_lift_threshold', 0.4),
            floor_lift_amount=reputation_params.get('floor_lift_amount', 0.2),
            initial_reputation=reputation_params.get('initial_reputation', 0.8)
        )
        
        # Mode controller params
        mode_params = self.defense_params.get('mode', {})
        self.mode_controller = ModeController(
            threshold_normal_to_alert=mode_params.get('threshold_normal_to_alert', 0.20),
            threshold_alert_to_defense=mode_params.get('threshold_alert_to_defense', 0.30),
            hysteresis_normal=mode_params.get('hysteresis_normal', 0.10),
            hysteresis_defense=mode_params.get('hysteresis_defense', 0.15),
            rep_gate_defense=mode_params.get('rep_gate_defense', 0.5),
            initial_mode=mode_params.get('initial_mode', 'NORMAL')
        )
    
    def _identify_malicious_clients(self) -> Set[int]:
        """Identify malicious clients từ attack config."""
        attack_type = self.config_metadata.get('attack_type', 'none')
        attack_ratio = self.config_metadata.get('attack_ratio', 0.0)
        num_clients = self.config_metadata.get('num_clients', 40)
        
        if attack_type == 'none' or attack_ratio == 0:
            return set()
        
        num_malicious = int(num_clients * attack_ratio)
        return set(range(num_malicious))
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[any, FitRes]],
        failures: List[Tuple[any, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]] | None:
        """Aggregate fit results với FULL DEFENSE PIPELINE."""
        
        if not results:
            return None
        
        # Baseline aggregation (no defense)
        if not self.enable_defense:
            return super().aggregate_fit(server_round, results, failures)
        
        # === FULL DEFENSE PIPELINE ===
        print(f"\n{'='*70}")
        print(f"ROUND {server_round} - FULL DEFENSE PIPELINE")
        print(f"{'='*70}")
        
        # Extract gradients and client IDs
        gradients = []
        client_ids = []
        
        for client_proxy, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            gradient = np.concatenate([p.flatten() for p in params])
            gradients.append(gradient)
            client_ids.append(int(client_proxy.cid))
        
        # Ground truth
        ground_truth_list = [cid in self.malicious_clients for cid in client_ids]
        
        # === STAGE 1: Multi-Layer Detection ===
        print(f"\n🔍 STAGE 1: MULTI-LAYER DETECTION")
        print("─"*70)
        
        # Layer 1
        layer1_results = self.layer1_detector.detect(
            gradients=gradients,
            client_ids=client_ids,
            is_malicious_ground_truth=ground_truth_list,
            current_round=server_round
        )
        
        # Layer 2
        layer2_results = self.layer2_detector.detect(
            gradients=gradients,
            client_ids=client_ids,
            current_round=server_round,
            layer1_flags=layer1_results
        )
        
        # Combine
        combined_flags = {
            cid: layer1_results.get(cid, False) or layer2_results.get(cid, False)
            for cid in client_ids
        }
        
        # === STAGE 2: Non-IID Handling ===
        print(f"\n📊 STAGE 2: NON-IID HANDLING")
        print("─"*70)
        
        # Update gradient history
        for i, cid in enumerate(client_ids):
            self.noniid_handler.update_client_gradient(cid, gradients[i])
        
        # Compute H score
        H = self.noniid_handler.compute_heterogeneity_score(gradients, client_ids)
        print(f"   Heterogeneity: H = {H:.3f}")
        
        # === STAGE 3: Reputation Update ===
        print(f"\n⭐ STAGE 3: REPUTATION UPDATE")
        print("─"*70)
        
        grad_median = np.median(np.vstack(gradients), axis=0)
        
        reputations = {}
        for i, cid in enumerate(client_ids):
            if cid not in self.reputation_system.reputations:
                self.reputation_system.initialize_client(cid)
            
            was_flagged = combined_flags.get(cid, False)
            rep = self.reputation_system.update(
                cid, gradients[i], grad_median, was_flagged, server_round
            )
            reputations[cid] = rep
        
        rep_mean = np.mean(list(reputations.values()))
        print(f"   Mean reputation: {rep_mean:.3f}")
        
        # === STAGE 4: Mode Control ===
        print(f"\n🎛️  STAGE 4: MODE CONTROL")
        print("─"*70)
        
        detected_ids = [cid for cid, flag in combined_flags.items() if flag]
        rho = len(detected_ids) / len(client_ids)
        
        new_mode = self.mode_controller.update_mode(
            threat_ratio=rho,
            detected_clients=detected_ids,
            reputations=reputations,
            current_round=server_round
        )
        
        print(f"   Threat ratio: ρ = {rho:.2%}")
        print(f"   Mode: {new_mode}")
        
        # === STAGE 5: Two-Stage Filtering ===
        print(f"\n🔧 STAGE 5: TWO-STAGE FILTERING")
        print("─"*70)
        
        confidence_scores = {cid: int(combined_flags.get(cid, False)) * 3 for cid in client_ids}
        
        trusted, filtered, filter_stats = self.two_stage_filter.filter_clients(
            client_ids=client_ids,
            confidence_scores=confidence_scores,
            reputations=reputations,
            mode=new_mode,
            heterogeneity=H
        )
        
        print(f"   Trusted: {len(trusted)}, Filtered: {len(filtered)}")
        
        # === STAGE 6: Mode-Adaptive Aggregation ===
        print(f"\n⚙️  STAGE 6: MODE-ADAPTIVE AGGREGATION")
        print("─"*70)
        print(f"   Aggregation method: {new_mode}")
        
        # Get trusted gradients
        trusted_indices = [i for i, cid in enumerate(client_ids) if cid in trusted]
        trusted_gradients = [gradients[i] for i in trusted_indices]
        
        if not trusted_gradients:
            print("   ⚠️  No trusted clients! Using all gradients.")
            trusted_gradients = gradients
        
        # Aggregate by mode
        aggregated_gradient = aggregate_by_mode(trusted_gradients, mode=new_mode)
        
        # Reshape back to parameters
        aggregated_params = []
        offset = 0
        for client_proxy, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            client_params = []
            for p in params:
                size = p.size
                client_params.append(aggregated_gradient[offset:offset+size].reshape(p.shape))
                offset += size
            break  # Use first client's structure
        
        aggregated_parameters = ndarrays_to_parameters(client_params)
        
        # === Metrics Calculation ===
        self._calculate_metrics(server_round, detected_ids, new_mode, rho, H)
        
        # Save checkpoint
        self.current_parameters = aggregated_parameters
        if self.auto_save and server_round % self.save_interval == 0:
            self._save_checkpoint(server_round)
        
        return aggregated_parameters, {}
    
    def _calculate_metrics(self, server_round, detected_ids, new_mode, rho, H):
        """Calculate detection metrics."""
        true_malicious = self.malicious_clients
        detected_malicious = set(detected_ids)
        
        # Confusion matrix
        tp = len(true_malicious & detected_malicious)
        fp = len(detected_malicious - true_malicious)
        fn = len(true_malicious - detected_malicious)
        tn = len([cid for cid in range(self.config_metadata.get('num_clients', 40))
                  if cid not in true_malicious and cid not in detected_malicious])
        
        # Update totals
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_tn += tn
        
        # Metrics
        detection_rate = (tp / len(true_malicious) * 100) if len(true_malicious) > 0 else 0
        fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        recall = detection_rate
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        print(f"\n📈 DETECTION METRICS")
        print("─"*70)
        print(f"   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"   Detection: {detection_rate:.1f}%")
        print(f"   FPR: {fpr:.1f}%")
        print(f"   Precision: {precision:.1f}%")
        print(f"   F1: {f1:.1f}%")
        
        # Save to history
        self.detection_history.append({
            'round': server_round,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'detection_rate': detection_rate,
            'fpr': fpr,
            'precision': precision,
            'f1': f1,
            'mode': new_mode,
            'threat_ratio': rho,
            'heterogeneity': H,
            'detected_ids': list(detected_malicious),
            'ground_truth': list(true_malicious)
        })
        
        print(f"{'='*70}\n")
    
    def _generate_model_name(self, server_round):
        """Generate model name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_clients = self.config_metadata.get('num_clients', 'unknown')
        attack_type = self.config_metadata.get('attack_type', 'none')
        attack_ratio = self.config_metadata.get('attack_ratio', 0.0)
        
        attack_info = f"{attack_type}{int(attack_ratio*100)}pct" if attack_type != 'none' else "clean"
        defense_info = "fullpipeline" if self.enable_defense else "baseline"
        
        return f"{num_clients}c_{attack_info}_{defense_info}_{timestamp}_r{server_round}"
    
    def _save_checkpoint(self, server_round):
        """Save checkpoint."""
        if self.current_parameters is None:
            return
        
        net = get_model()
        params_arrays = parameters_to_ndarrays(self.current_parameters)
        set_parameters(net, params_arrays)
        
        model_name = self._generate_model_name(server_round)
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "current_round": server_round,
            "config": self.config_metadata,
            "defense_params": self.defense_params,  # Save defense params
            "accuracy_history": self.accuracy_history,
            "detection_history": self.detection_history[-10:] if self.detection_history else []
        }
        
        manager = ModelManager(save_dir=self.save_dir)
        manager.save_model(net, metadata, model_name=model_name)
        print(f"💾 Saved: {model_name}\n")


def server_fn(context: Context) -> ServerAppComponents:
    """Create server với ALL params từ config."""
    
    # === Load Basic FL Config ===
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
    
    # Defense
    enable_defense = context.run_config.get("enable-defense", True)
    
    # Attack
    attack_type = context.run_config.get("attack-type", "none")
    attack_ratio = context.run_config.get("attack-ratio", 0.0)
    partition_type = context.run_config.get("partition-type", "iid")
    
    # === Load ALL Defense Params từ Config ===
    defense_params = {}
    
    if enable_defense:
        # Layer 1 params
        defense_params['layer1'] = {
            'pca_dims': context.run_config.get("defense.layer1.pca-dims", 20),
            'dbscan_min_samples': context.run_config.get("defense.layer1.dbscan-min-samples", 3),
            'dbscan_eps_multiplier': context.run_config.get("defense.layer1.dbscan-eps-multiplier", 0.5),
            'mad_k_normal': context.run_config.get("defense.layer1.mad-k-normal", 4.0),
            'mad_k_warmup': context.run_config.get("defense.layer1.mad-k-warmup", 6.0),
            'voting_threshold_normal': context.run_config.get("defense.layer1.voting-threshold-normal", 2),
            'voting_threshold_warmup': context.run_config.get("defense.layer1.voting-threshold-warmup", 3),
            'warmup_rounds': context.run_config.get("defense.layer1.warmup-rounds", 10)
        }
        
        # Layer 2 params
        defense_params['layer2'] = {
            'distance_multiplier': context.run_config.get("defense.layer2.distance-multiplier", 1.5),
            'cosine_threshold': context.run_config.get("defense.layer2.cosine-threshold", 0.3),
            'warmup_rounds': context.run_config.get("defense.layer2.warmup-rounds", 15)
        }
        
        # Non-IID params
        defense_params['noniid'] = {
            'h_threshold_normal': context.run_config.get("defense.noniid.h-threshold-normal", 0.6),
            'h_threshold_alert': context.run_config.get("defense.noniid.h-threshold-alert", 0.5),
            'adaptive_multiplier': context.run_config.get("defense.noniid.adaptive-multiplier", 1.5),
            'baseline_percentile': context.run_config.get("defense.noniid.baseline-percentile", 60),
            'baseline_window_size': context.run_config.get("defense.noniid.baseline-window-size", 10)
        }
        
        # Filtering params
        defense_params['filtering'] = {
            'hard_k_threshold': context.run_config.get("defense.filtering.hard-k-threshold", 3),
            'soft_reputation_threshold': context.run_config.get("defense.filtering.soft-reputation-threshold", 0.4),
            'soft_distance_multiplier': context.run_config.get("defense.filtering.soft-distance-multiplier", 2.0),
            'soft_enabled': context.run_config.get("defense.filtering.soft-enabled", True)
        }
        
        # Reputation params
        defense_params['reputation'] = {
            'ema_alpha_increase': context.run_config.get("defense.reputation.ema-alpha-increase", 0.4),
            'ema_alpha_decrease': context.run_config.get("defense.reputation.ema-alpha-decrease", 0.2),
            'penalty_flagged': context.run_config.get("defense.reputation.penalty-flagged", 0.2),
            'penalty_variance': context.run_config.get("defense.reputation.penalty-variance", 0.1),
            'reward_clean': context.run_config.get("defense.reputation.reward-clean", 0.1),
            'floor_lift_threshold': context.run_config.get("defense.reputation.floor-lift-threshold", 0.4),
            'floor_lift_amount': context.run_config.get("defense.reputation.floor-lift-amount", 0.2),
            'initial_reputation': context.run_config.get("defense.reputation.initial-reputation", 0.8)
        }
        
        # Mode params
        defense_params['mode'] = {
            'threshold_normal_to_alert': context.run_config.get("defense.mode.threshold-normal-to-alert", 0.20),
            'threshold_alert_to_defense': context.run_config.get("defense.mode.threshold-alert-to-defense", 0.30),
            'hysteresis_normal': context.run_config.get("defense.mode.hysteresis-normal", 0.10),
            'hysteresis_defense': context.run_config.get("defense.mode.hysteresis-defense", 0.15),
            'rep_gate_defense': context.run_config.get("defense.mode.rep-gate-defense", 0.5),
            'initial_mode': context.run_config.get("defense.mode.initial-mode", "NORMAL")
        }
    
    # Print config
    print(f"\n{'='*70}")
    print("SERVER CONFIGURATION")
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
        defense_params=defense_params  # Pass ALL defense params!
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)