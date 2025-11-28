"""
Server Application - Full Defense Pipeline (FIXED VERSION)
===========================================================
FIX: Đã khắc phục các vấn đề:
1. ✅ Baseline Tracking được kích hoạt
2. ✅ Adaptive Thresholds hoạt động
3. ✅ Confidence Scoring đúng công thức: ci = (s_base + adj) × factor_baseline
4. ✅ Reputation Update có baseline penalty
5. ✅ Reporting/Stats được thu thập

Load ALL parameters from pyproject.toml và pass xuống defense components.
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
from cifar_cnn.model_manager import ModelManager

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
    FedProx Strategy với FULL DEFENSE PIPELINE - FIXED VERSION.
    
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
                 defense_params=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        # Client ID mappings
        self.client_id_to_sequential = {}
        self.sequential_to_client_id = {}

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
            print("🛡️  FULL DEFENSE PIPELINE ENABLED (FIXED VERSION)")
            print("="*70)
            print("  ✓ Layer 1: Enhanced DBSCAN Detection")
            print("  ✓ Layer 2: Distance + Direction Detection")
            print("  ✓ Non-IID Handler (WITH Baseline Tracking)")
            print("  ✓ Two-Stage Filtering (WITH Adaptive Thresholds)")
            print("  ✓ Reputation System (WITH Baseline Penalty)")
            print("  ✓ Mode Controller")
            print("  ✓ Mode-Adaptive Aggregation")
            print("="*70)
            if self.malicious_clients:
                print(f"  🎯 Ground Truth: {len(self.malicious_clients)} malicious clients")
                print(f"     IDs: {sorted(list(self.malicious_clients))}")
            print("="*70 + "\n")
            
            # Initialize ALL components
            self._initialize_defense_components()
            
            # Track detection stats
            self.detection_history = []
            self.total_tp = 0
            self.total_fp = 0
            self.total_fn = 0
            self.total_tn = 0
    
    def _initialize_defense_components(self):
        """Initialize ALL defense components."""
        
        # Layer 1
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
        
        # Layer 2
        layer2_params = self.defense_params.get('layer2', {})
        self.layer2_detector = Layer2Detector(
            distance_multiplier=layer2_params.get('distance_multiplier', 1.5),
            cosine_threshold=layer2_params.get('cosine_threshold', 0.3),
            warmup_rounds=layer2_params.get('warmup_rounds', 15)
        )
        
        # Non-IID Handler
        noniid_params = self.defense_params.get('noniid', {})
        self.noniid_handler = NonIIDHandler(
            h_threshold_normal=noniid_params.get('h_threshold_normal', 0.6),
            h_threshold_alert=noniid_params.get('h_threshold_alert', 0.5),
            adaptive_multiplier=noniid_params.get('adaptive_multiplier', 1.5),
            adjustment_factor=noniid_params.get('adjustment_factor', 0.4),  # NEW
            baseline_percentile=noniid_params.get('baseline_percentile', 60),
            baseline_window_size=noniid_params.get('baseline_window_size', 10),
            delta_norm_weight=noniid_params.get('delta_norm_weight', 0.5),  # NEW
            delta_direction_weight=noniid_params.get('delta_direction_weight', 0.5)  # NEW
        )
        
        # Two-Stage Filter
        filtering_params = self.defense_params.get('filtering', {})
        self.two_stage_filter = TwoStageFilter(
            hard_k_threshold=filtering_params.get('hard_k_threshold', 3),
            hard_threshold_min=filtering_params.get('hard_threshold_min', 0.85),  # NEW
            hard_threshold_max=filtering_params.get('hard_threshold_max', 0.95),  # NEW
            soft_reputation_threshold=filtering_params.get('soft_reputation_threshold', 0.4),
            soft_distance_multiplier=filtering_params.get('soft_distance_multiplier', 2.0),
            soft_enabled=filtering_params.get('soft_enabled', True)
        )
        
        # Reputation System
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
        
        # Mode Controller
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
        """Identify malicious clients from attack config."""
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
        """Aggregate fit results với FULL DEFENSE PIPELINE (FIXED)."""
        
        # Create client ID mapping
        if not self.client_id_to_sequential:
            all_client_ids = [client_proxy.cid for client_proxy, _ in results]
            sorted_client_ids = sorted(all_client_ids)
            
            for idx, client_id in enumerate(sorted_client_ids):
                self.client_id_to_sequential[client_id] = idx
                self.client_id_to_sequential[str(client_id)] = idx
                try:
                    self.client_id_to_sequential[int(client_id)] = idx
                except:
                    pass
                self.sequential_to_client_id[idx] = client_id
            
            print(f"✅ Created client ID mapping: {len(sorted_client_ids)} clients")
        
        if not results:
            return None
        
        # Baseline aggregation
        if not self.enable_defense:
            return super().aggregate_fit(server_round, results, failures)
        
        # === FULL DEFENSE PIPELINE (FIXED) ===
        print(f"\n{'='*70}")
        print(f"ROUND {server_round} - FULL DEFENSE PIPELINE (FIXED)")
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
        grad_median = np.median(np.vstack(gradients), axis=0)
        
        # === STAGE 2.5: FIX - Compute Baseline Deviations (DETAILED) ===
        print(f"\n🔬 STAGE 2.5: BASELINE TRACKING (DETAILED)")
        print("─"*70)
        
        baseline_deviations = {}
        for i, cid in enumerate(client_ids):
            # Use detailed version với norm + direction components
            dev_details = self.noniid_handler.compute_baseline_deviation_detailed(
                cid, gradients[i], grad_median
            )
            baseline_deviations[cid] = dev_details['delta_combined']
            
            # Log details cho clients có deviation cao
            if dev_details['delta_combined'] > 0.3:
                print(f"   Client {cid}: δ={dev_details['delta_combined']:.3f} "
                      f"(norm={dev_details['delta_norm']:.3f}, "
                      f"dir={dev_details['delta_direction']:.3f})")
        
        num_high_deviation = sum(1 for d in baseline_deviations.values() if d > 0.3)
        print(f"   Clients with high deviation (>0.3): {num_high_deviation}/{len(client_ids)}")
        
        # Optional: Log component statistics
        if num_high_deviation > 0:
            avg_norm = np.mean([
                self.noniid_handler.compute_baseline_deviation_detailed(
                    cid, gradients[i], grad_median
                )['delta_norm']
                for i, cid in enumerate(client_ids)
            ])
            avg_dir = np.mean([
                self.noniid_handler.compute_baseline_deviation_detailed(
                    cid, gradients[i], grad_median
                )['delta_direction']
                for i, cid in enumerate(client_ids)
            ])
            print(f"   Avg components: norm={avg_norm:.3f}, direction={avg_dir:.3f}")
        
        # === STAGE 3: FIX - Confidence Scoring with Baseline Factor ===
        print(f"\n💯 STAGE 3: CONFIDENCE SCORING (FIX)")
        print("─"*70)
                
        # Initialize reputations
        reputations = {}
        for cid in client_ids:
            if cid not in self.reputation_system.reputations:
                self.reputation_system.initialize_client(cid)
            reputations[cid] = self.reputation_system.get_reputation(cid)
        
        # Compute confidence scores (FIXED FORMULA)
        confidence_scores = {}
        for cid in client_ids:
            # Base score from detection layers
            was_flagged = combined_flags.get(cid, False)
            s_base = 3 if was_flagged else 0
            
            # Adjustment from old reputation
            rep = reputations.get(cid, 0.8)
            adj_rep = (1.0 - rep) * 2  # Scale to [0, 2]
            
            # FIX: Factor from baseline deviation (dùng > thay vì <)
            deviation = baseline_deviations.get(cid, 0.0)
            factor_baseline = 0.8 if deviation > 0.3 else 1.0
            
            # Final confidence (đúng công thức trong PDF)
            ci = np.clip((s_base + adj_rep) * factor_baseline, 0.0, 1.0)
            confidence_scores[cid] = ci
        
        print(f"   Confidence scores computed with baseline factor")
        print(f"   Mean confidence: {np.mean(list(confidence_scores.values())):.3f}")
        
        # === STAGE 4: Mode Control (trước khi filter) ===
        print(f"\n🎛️  STAGE 4: MODE CONTROL")
        print("─"*70)
        
        detected_ids = [cid for cid, flag in combined_flags.items() if flag]
        rho = len(detected_ids) / len(client_ids) if len(client_ids) > 0 else 0.0
        
        new_mode = self.mode_controller.update_mode(
            threat_ratio=rho,
            detected_clients=detected_ids,
            reputations=reputations,
            current_round=server_round
        )
        
        print(f"   Threat ratio: ρ = {rho:.2%}")
        print(f"   Mode: {new_mode}")
        
        # === STAGE 5: FIX - Two-Stage Filtering with Adaptive Thresholds ===
        print(f"\n🔧 STAGE 5: TWO-STAGE FILTERING (FIX)")
        print("─"*70)
        
        trusted, filtered, filter_stats = self.two_stage_filter.filter_clients(
            client_ids=client_ids,
            confidence_scores=confidence_scores,
            reputations=reputations,
            mode=new_mode,
            heterogeneity=H,
            noniid_handler=self.noniid_handler  # FIX: Pass handler để tính adaptive thresholds
        )
        
        print(f"   Trusted: {len(trusted)}, Filtered: {len(filtered)}")
        print(f"   Hard filtered: {filter_stats['hard_filtered']}")
        print(f"   Soft filtered: {filter_stats['soft_filtered']}")
        
        # === STAGE 6: FIX - Reputation Update with Baseline Penalty ===
        print(f"\n⭐ STAGE 6: REPUTATION UPDATE (FIX)")
        print("─"*70)
        
        for i, cid in enumerate(client_ids):
            was_flagged = combined_flags.get(cid, False)
            deviation = baseline_deviations.get(cid, 0.0)
            
            # Standard reputation update
            rep = self.reputation_system.update(
                cid, gradients[i], grad_median, was_flagged, server_round
            )
            
            # FIX: Apply baseline penalty
            baseline_penalty = 0.1 if deviation > 0.3 else 0.0
            if baseline_penalty > 0:
                rep = max(0.0, rep - baseline_penalty)
                self.reputation_system.reputations[cid] = rep
                print(f"   Client {cid}: Applied baseline penalty ({deviation:.3f} > 0.3)")
            
            reputations[cid] = rep
        
        rep_mean = np.mean(list(reputations.values()))
        print(f"   Mean reputation: {rep_mean:.3f}")
        
        # === STAGE 7: Mode-Adaptive Aggregation ===
        print(f"\n⚙️  STAGE 7: MODE-ADAPTIVE AGGREGATION")
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
            break
        
        aggregated_parameters = ndarrays_to_parameters(client_params)
        
        # === STAGE 8: FIX - Collect Stats ===
        print(f"\n📈 STAGE 8: REPORTING (FIX)")
        print("─"*70)
        
        # Layer stats
        layer1_stats = self.layer1_detector.get_stats()
        layer2_stats = self.layer2_detector.get_stats()
        noniid_stats = self.noniid_handler.get_stats()
        filter_stats_full = self.two_stage_filter.get_stats()
        reputation_stats = self.reputation_system.get_stats()
        mode_stats = self.mode_controller.get_stats()
        
        print(f"   Layer1: PCA fitted={layer1_stats.get('pca_fitted', False)}")
        print(f"   NonIID: Tracked clients={noniid_stats.get('num_tracked_clients', 0)}")
        print(f"   Reputation: Mean={reputation_stats.get('mean_reputation', 0):.3f}")
        print(f"   Mode: {mode_stats.get('current_mode', 'N/A')}")
        
        # Metrics
        self._calculate_metrics(server_round, detected_ids, new_mode, rho, H)
        
        # Save checkpoint
        self.current_parameters = aggregated_parameters
        if self.auto_save and server_round % self.save_interval == 0:
            self._save_checkpoint(server_round)
        
        return aggregated_parameters, {}
    
    def _calculate_metrics(self, server_round, detected_ids, new_mode, rho, H):
        """Calculate detection metrics with proper ID mapping."""
        
        # Map detected hashed IDs to sequential IDs
        detected_sequential = set()
        for hashed_id in detected_ids:
            seq_id = self.client_id_to_sequential.get(hashed_id, -1)
            if seq_id != -1:
                detected_sequential.add(seq_id)
        
        # Ground truth
        true_malicious = self.malicious_clients
        
        # Calculate metrics
        num_clients = self.config_metadata.get('num_clients', 40)
        
        tp = len(true_malicious & detected_sequential)
        fp = len(detected_sequential - true_malicious)
        fn = len(true_malicious - detected_sequential)
        tn = num_clients - tp - fp - fn
        
        # Rates
        total_malicious = len(true_malicious)
        total_benign = num_clients - total_malicious
        
        detection_rate = (tp / total_malicious * 100) if total_malicious > 0 else 0.0
        fpr = (fp / total_benign * 100) if total_benign > 0 else 0.0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        f1 = (2 * tp / (2 * tp + fp + fn) * 100) if (2 * tp + fp + fn) > 0 else 0.0
        
        # Accumulate
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_tn += tn
        
        # Print
        print(f"\n📈 DETECTION METRICS")
        print(f"{'─'*70}")
        print(f"   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"   Detection: {detection_rate:.1f}%")
        print(f"   FPR: {fpr:.1f}%")
        print(f"   Precision: {precision:.1f}%")
        print(f"   F1: {f1:.1f}%")
        print(f"{'═'*70}")
    
    def _generate_model_name(self, server_round):
        """Generate model name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_clients = self.config_metadata.get('num_clients', 'unknown')
        attack_type = self.config_metadata.get('attack_type', 'none')
        attack_ratio = self.config_metadata.get('attack_ratio', 0.0)
        
        attack_info = f"{attack_type}{int(attack_ratio*100)}pct" if attack_type != 'none' else "clean"
        defense_info = "fullpipeline_fixed" if self.enable_defense else "baseline"
        
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
            "defense_params": self.defense_params,
            "accuracy_history": self.accuracy_history,
            "detection_history": self.detection_history[-10:] if self.detection_history else []
        }
        
        manager = ModelManager(save_dir=self.save_dir)
        manager.save_model(net, metadata, model_name=model_name)
        print(f"💾 Saved: {model_name}\n")


def server_fn(context: Context) -> ServerAppComponents:
    """Create server with ALL params from config."""
    
    # Load config
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
    
    enable_defense = context.run_config.get("enable-defense", True)
    
    attack_type = context.run_config.get("attack-type", "none")
    attack_ratio = context.run_config.get("attack-ratio", 0.0)
    partition_type = context.run_config.get("partition-type", "iid")
    
    # Load ALL defense params
    defense_params = {}
    
    if enable_defense:
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
        
        defense_params['layer2'] = {
            'distance_multiplier': context.run_config.get("defense.layer2.distance-multiplier", 1.5),
            'cosine_threshold': context.run_config.get("defense.layer2.cosine-threshold", 0.3),
            'warmup_rounds': context.run_config.get("defense.layer2.warmup-rounds", 15)
        }
        
        defense_params['noniid'] = {
            'h_threshold_normal': context.run_config.get("defense.noniid.h-threshold-normal", 0.6),
            'h_threshold_alert': context.run_config.get("defense.noniid.h-threshold-alert", 0.5),
            'adaptive_multiplier': context.run_config.get("defense.noniid.adaptive-multiplier", 1.5),
            'adjustment_factor': context.run_config.get("defense.noniid.adjustment-factor", 0.4),  # NEW
            'baseline_percentile': context.run_config.get("defense.noniid.baseline-percentile", 60),
            'baseline_window_size': context.run_config.get("defense.noniid.baseline-window-size", 10),
            'delta_norm_weight': context.run_config.get("defense.noniid.delta-norm-weight", 0.5),  # NEW
            'delta_direction_weight': context.run_config.get("defense.noniid.delta-direction-weight", 0.5)  # NEW
        }
        
        defense_params['filtering'] = {
            'hard_k_threshold': context.run_config.get("defense.filtering.hard-k-threshold", 3),
            'hard_threshold_min': context.run_config.get("defense.filtering.hard-threshold-min", 0.85),  # NEW
            'hard_threshold_max': context.run_config.get("defense.filtering.hard-threshold-max", 0.95),  # NEW
            'soft_reputation_threshold': context.run_config.get("defense.filtering.soft-reputation-threshold", 0.4),
            'soft_distance_multiplier': context.run_config.get("defense.filtering.soft-distance-multiplier", 2.0),
            'soft_enabled': context.run_config.get("defense.filtering.soft-enabled", True)
        }
        
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
    print("SERVER CONFIGURATION (FIXED VERSION)")
    print(f"{'='*70}")
    print(f"  Clients: {num_clients} (fit={num_fit_clients}/round)")
    print(f"  Rounds: {num_rounds}")
    print(f"  Attack: {attack_type} ({attack_ratio*100:.0f}%)")
    print(f"  Defense: {'ENABLED ✓' if enable_defense else 'DISABLED ✗'}")
    print(f"{'='*70}\n")
    
    config_metadata = {
        'num_clients': num_clients,
        'partition_type': partition_type,
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
        defense_params=defense_params
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)