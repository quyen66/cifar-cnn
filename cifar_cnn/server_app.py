# cifar_cnn/server_app.py

import time  # [FIX] Import time để đo overhead
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

# Import defense components
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
    if sum(examples) == 0:
        return {"accuracy": 0}
    return {"accuracy": sum(accuracies) / sum(examples)}

class FullPipelineStrategy(FedProx):
    """
    FedProx Strategy với FULL DEFENSE PIPELINE.
    Bao gồm đo lường Overhead và Warmup Logic.
    """
    
    def __init__(self, 
                 proximal_mu: float,
                 *args,
                 auto_save=True,
                 save_dir="saved_models",
                 save_interval=10,
                 config_metadata=None,
                 start_round=0,
                 enable_defense=False,
                 defense_params=None,
                 **kwargs):
        
        super().__init__(proximal_mu=proximal_mu, *args, **kwargs)
        
        self.client_id_to_sequential = {}
        self.sequential_to_client_id = {}
        self.auto_save = auto_save
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.config_metadata = config_metadata or {}
        self.start_time = datetime.now()
        self.current_parameters = None
        self.start_round = start_round
        
        self.accuracy_history = []
        self.detection_history = []
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        
        self.enable_defense = enable_defense
        self.defense_params = defense_params or {}
        self.malicious_clients = self._identify_malicious_clients()
        
        if self.enable_defense:
            print("\n" + "="*70)
            print("🛡️  FULL DEFENSE PIPELINE ENABLED")
            print("="*70)
            self._initialize_defense_components()
            
    def _initialize_defense_components(self):
        # ... (Layer 1, 2, NonIID, Filtering, Reputation giữ nguyên như cũ) ...
        # Chỉ copy lại phần ModeController có sửa đổi bên dưới
        
        l1p = self.defense_params.get('layer1', {})
        self.layer1_detector = Layer1Detector(
            pca_dims=l1p.get('pca_dims', 20),
            dbscan_min_samples=l1p.get('dbscan_min_samples', 3),
            dbscan_eps_multiplier=l1p.get('dbscan_eps_multiplier', 0.5),
            mad_k_normal=l1p.get('mad_k_normal', 4.0),
            mad_k_warmup=l1p.get('mad_k_warmup', 6.0),
            voting_threshold_normal=l1p.get('voting_threshold_normal', 2),
            voting_threshold_warmup=l1p.get('voting_threshold_warmup', 3),
            warmup_rounds=10
        )
        
        l2p = self.defense_params.get('layer2', {})
        self.layer2_detector = Layer2Detector(
            distance_multiplier=l2p.get('distance_multiplier', 1.5),
            cosine_threshold=l2p.get('cosine_threshold', 0.3),
            warmup_rounds=10
        )
        
        nip = self.defense_params.get('noniid', {})
        self.noniid_handler = NonIIDHandler(
            weight_cv=nip.get('weight_cv', 0.4),
            weight_sim=nip.get('weight_sim', 0.4),
            weight_cluster=nip.get('weight_cluster', 0.2),
            h_threshold_normal=nip.get('h_threshold_normal', 0.6),
            h_threshold_alert=nip.get('h_threshold_alert', 0.5),
            adaptive_multiplier=nip.get('adaptive_multiplier', 1.5),
            adjustment_factor=nip.get('adjustment_factor', 0.4),
            baseline_percentile=nip.get('baseline_percentile', 60),
            baseline_window_size=nip.get('baseline_window_size', 10),
            delta_norm_weight=nip.get('delta_norm_weight', 0.5),
            delta_direction_weight=nip.get('delta_direction_weight', 0.5)
        )
        
        fp = self.defense_params.get('filtering', {})
        self.two_stage_filter = TwoStageFilter(
            hard_k_threshold=fp.get('hard_k_threshold', 3),
            hard_threshold_min=fp.get('hard_threshold_min', 0.85),
            hard_threshold_max=fp.get('hard_threshold_max', 0.95),
            soft_reputation_threshold=fp.get('soft_reputation_threshold', 0.4),
            soft_distance_multiplier=fp.get('soft_distance_multiplier', 2.0),
            soft_enabled=fp.get('soft_enabled', True)
        )
        
        rp = self.defense_params.get('reputation', {})
        self.reputation_system = ReputationSystem(
            ema_alpha_increase=rp.get('ema_alpha_increase', 0.1),
            ema_alpha_decrease=rp.get('ema_alpha_decrease', 0.6),
            penalty_flagged=rp.get('penalty_flagged', 0.2),
            penalty_variance=rp.get('penalty_variance', 0.1),
            reward_clean=rp.get('reward_clean', 0.1),
            floor_warning_threshold=rp.get('floor_warning_threshold', 0.2),
            floor_target_value=rp.get('floor_target_value', 0.3),
            floor_probation_rounds=rp.get('floor_probation_rounds', 5),
            initial_reputation=rp.get('initial_reputation', 0.5)
        )
        
        # [FIX] Mode Controller initialization with Warmup Rounds
        mp = self.defense_params.get('mode', {})
        self.mode_controller = ModeController(
            threshold_normal_to_alert=mp.get('threshold_normal_to_alert', 0.15),
            threshold_alert_to_defense=mp.get('threshold_alert_to_defense', 0.30),
            hysteresis_normal=mp.get('hysteresis_normal', 0.05),
            hysteresis_defense=mp.get('hysteresis_defense', 0.10),
            rep_gate_defense=mp.get('rep_gate_defense', 0.5),
            initial_mode="DEFENSE", # Force start in Defense
            warmup_rounds=10        # 10 vòng đầu force Defense
        )
        
        self.safe_weight_epsilon = mp.get('safe_weight_epsilon', 1e-6)
    
    def _identify_malicious_clients(self) -> Set[int]:
        attack_type = self.config_metadata.get('attack_type', 'none')
        attack_ratio = self.config_metadata.get('attack_ratio', 0.0)
        num_clients = self.config_metadata.get('num_clients', 40)
        if attack_type == 'none' or attack_ratio == 0: return set()
        return set(range(int(num_clients * attack_ratio)))
    
    def aggregate_fit(self, server_round, results, failures):
        if not self.client_id_to_sequential:
            all_ids = [c.cid for c, _ in results]
            try: sorted_ids = sorted(all_ids, key=lambda x: int(x))
            except: sorted_ids = sorted(all_ids)
            for idx, cid in enumerate(sorted_ids):
                self.client_id_to_sequential[cid] = idx; self.client_id_to_sequential[str(cid)] = idx
                try: self.client_id_to_sequential[int(cid)] = idx
                except: pass
                self.sequential_to_client_id[idx] = cid
        
        if not results: return None
        
        # [FIX] BẮT ĐẦU ĐO THỜI GIAN
        t_start = time.time()

        if not self.enable_defense:
            agg_res = super().aggregate_fit(server_round, results, failures)
            t_end = time.time()
            print(f"⏱️  FedAvg Time: {t_end - t_start:.4f}s")
            return agg_res
        
        # === FULL DEFENSE PIPELINE ===
        print(f"\n{'='*70}")
        print(f"ROUND {server_round} - FULL DEFENSE PIPELINE")
        print(f"{'='*70}")
        
        # ... Extraction ...
        gradients = []; client_ids = []
        for client_proxy, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            gradients.append(np.concatenate([p.flatten() for p in params]))
            try: cid_int = int(client_proxy.cid)
            except: cid_int = self.client_id_to_sequential.get(client_proxy.cid, -1)
            client_ids.append(cid_int)
        
        ground_truth = [cid in self.malicious_clients for cid in client_ids]
        
        # ... Stages 1-3 ...
        l1_flags = self.layer1_detector.detect(gradients, client_ids, ground_truth, server_round)
        l2_flags = self.layer2_detector.detect(gradients, client_ids, server_round, l1_flags)
        combined_flags = {cid: l1_flags.get(cid, False) or l2_flags.get(cid, False) for cid in client_ids}
        
        for i, cid in enumerate(client_ids): self.noniid_handler.update_client_gradient(cid, gradients[i])
        H = self.noniid_handler.compute_heterogeneity_score(gradients, client_ids)
        grad_median = np.median(np.vstack(gradients), axis=0)
        
        baseline_deviations = {}
        for i, cid in enumerate(client_ids):
            dev = self.noniid_handler.compute_baseline_deviation_detailed(cid, gradients[i], grad_median)
            baseline_deviations[cid] = dev['delta_combined']
            
        reputations = {}
        for cid in client_ids:
            self.reputation_system.initialize_client(cid)
            reputations[cid] = self.reputation_system.get_reputation(cid)
            
        confidence_scores = {}
        for cid in client_ids:
            s_base = 3 if combined_flags.get(cid, False) else 0
            adj_rep = (1.0 - reputations.get(cid, 0.5)) * 2
            factor = 0.8 if baseline_deviations.get(cid, 0) > 0.3 else 1.0
            confidence_scores[cid] = np.clip((s_base + adj_rep) * factor, 0.0, 1.0)
            
        # ... Stage 4 Mode ...
        detected_ids = [cid for cid, f in combined_flags.items() if f]
        rho = len(detected_ids) / len(client_ids) if client_ids else 0.0
        new_mode = self.mode_controller.update_mode(rho, detected_ids, reputations, server_round)
        print(f"   Threat ratio: {rho:.2%}, Mode: {new_mode}")
        
        # ... Stage 5 Filter ...
        trusted, filtered, _ = self.two_stage_filter.filter_clients(
            client_ids, confidence_scores, reputations, new_mode, H, self.noniid_handler
        )
        
        # ... Stage 6 Rep Update ...
        for i, cid in enumerate(client_ids):
            rep = self.reputation_system.update(cid, None, None, combined_flags.get(cid, False), server_round)
            if baseline_deviations.get(cid, 0) > 0.3:
                rep = max(0.0, rep - 0.1)
                self.reputation_system.reputations[str(cid)] = rep
            reputations[cid] = rep
            
        # ... Stage 7 Aggregation ...
        trusted_indices = [i for i, cid in enumerate(client_ids) if cid in trusted]
        trusted_grads = [gradients[i] for i in trusted_indices]
        trusted_reps = [reputations[client_ids[i]] for i in trusted_indices]
        
        if not trusted_grads:
            trusted_grads = gradients
            trusted_reps = [reputations[cid] for cid in client_ids]
            
        aggregated_gradient = aggregate_by_mode(
            trusted_grads, 
            mode=new_mode, 
            reputations=trusted_reps,
            threat_level=rho,
            epsilon=self.safe_weight_epsilon
        )
        
        aggregated_params = []
        offset = 0
        for _, fit_res in results:
            params_list = parameters_to_ndarrays(fit_res.parameters)
            for p in params_list:
                aggregated_params.append(aggregated_gradient[offset:offset+p.size].reshape(p.shape))
                offset += p.size
            break
        
        # [FIX] KẾT THÚC ĐO THỜI GIAN & TÍNH OVERHEAD
        t_end = time.time()
        defense_duration = t_end - t_start
        
        # Giả định thời gian FedAvg thuần (T_base) khoảng 0.1s
        t_base_est = 0.1 
        overhead_percent = ((defense_duration - t_base_est) / t_base_est) * 100
        
        print(f"⏱️  Defense Time: {defense_duration:.4f}s")
        print(f"⚠️  Overhead: +{overhead_percent:.1f}% (Estimated vs T_base={t_base_est}s)")
        
        # Metrics
        self.detection_history.append({
            "round": server_round,
            "defense_time": defense_duration, # Lưu vào lịch sử
            "overhead": overhead_percent
        })
        self._calculate_metrics(server_round, detected_ids, new_mode, rho, H)
        
        self.current_parameters = ndarrays_to_parameters(aggregated_params)
        if self.auto_save and server_round % self.save_interval == 0:
            self._save_checkpoint(server_round)
            
        return self.current_parameters, {}

    # (Các hàm helper _calculate_metrics, _generate_model_name, _save_checkpoint giữ nguyên)
    def _calculate_metrics(self, server_round, detected_ids, new_mode, rho, H):
        # Normalize detected IDs
        detected_sequential = set()
        for d_id in detected_ids:
            try: d_int = int(d_id); detected_sequential.add(d_int)
            except: 
                if d_id in self.client_id_to_sequential: detected_sequential.add(self.client_id_to_sequential[d_id])

        tp = len(self.malicious_clients & detected_sequential)
        fp = len(detected_sequential - self.malicious_clients)
        fn = len(self.malicious_clients - detected_sequential)
        num_clients = self.config_metadata.get('num_clients', 40)
        tn = num_clients - tp - fp - fn
        
        detection_rate = (tp / len(self.malicious_clients) * 100) if self.malicious_clients else 0.0
        fpr = (fp / (num_clients - len(self.malicious_clients)) * 100) if (num_clients - len(self.malicious_clients)) > 0 else 0.0
        
        self.total_tp += tp; self.total_fp += fp; self.total_fn += fn; self.total_tn += tn
        print(f"\n📈 METRICS: TP={tp}, FP={fp}, Detect={detection_rate:.1f}%, FPR={fpr:.1f}%")

    def _generate_model_name(self, server_round):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_clients = self.config_metadata.get('num_clients', 'unknown')
        attack_type = self.config_metadata.get('attack_type', 'none')
        defense_info = "fullpipeline_fixed" if self.enable_defense else "baseline"
        return f"{num_clients}c_{attack_type}_{defense_info}_{timestamp}_r{server_round}"
    
    def _save_checkpoint(self, server_round):
        if self.current_parameters is None: return
        net = get_model()
        params_arrays = parameters_to_ndarrays(self.current_parameters)
        set_parameters(net, params_arrays)
        
        model_name = self._generate_model_name(server_round)
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "current_round": server_round,
            "config": self.config_metadata,
            "defense_params": self.defense_params,
            "detection_history": self.detection_history[-10:] if self.detection_history else []
        }
        manager = ModelManager(save_dir=self.save_dir)
        manager.save_model(net, metadata, model_name=model_name)
        print(f"💾 Saved: {model_name}\n")

def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    num_rounds = cfg.get("num-server-rounds", 50)
    num_clients = cfg.get("num-clients", 40)
    fraction_fit = cfg.get("fraction-fit", 0.6)
    fraction_evaluate = cfg.get("fraction-evaluate", 0.2)
    num_fit_clients = max(1, int(num_clients * fraction_fit))
    num_evaluate_clients = max(1, int(num_clients * fraction_evaluate))
    proximal_mu = cfg.get("proximal-mu", 0.01)
    enable_defense = cfg.get("enable-defense", True)
    GLOBAL_WARMUP = cfg.get("warmup-rounds", 10)
    
    defense_params = {}
    if enable_defense:
        
        defense_params['layer1'] = { 'pca_dims': cfg.get("defense.layer1.pca-dims", 20), 'dbscan_min_samples': 3, 'dbscan_eps_multiplier': 0.5, 'mad_k_normal': 4.0, 'mad_k_warmup': 6.0, 'voting_threshold_normal': 2, 'voting_threshold_warmup': 3, 'warmup_rounds': GLOBAL_WARMUP }
        defense_params['layer2'] = { 'distance_multiplier': 1.5, 'cosine_threshold': 0.3, 'warmup_rounds': GLOBAL_WARMUP }
        defense_params['noniid'] = { 'weight_cv': 0.4, 'weight_sim': 0.4, 'weight_cluster': 0.2, 'h_threshold_normal': 0.6, 'h_threshold_alert': 0.5, 'adaptive_multiplier': 1.5, 'adjustment_factor': 0.4, 'baseline_percentile': 60, 'baseline_window_size': 10, 'delta_norm_weight': 0.5, 'delta_direction_weight': 0.5 }
        defense_params['filtering'] = { 'hard_k_threshold': 3, 'hard_threshold_min': 0.85, 'hard_threshold_max': 0.95, 'soft_reputation_threshold': 0.4, 'soft_distance_multiplier': 2.0, 'soft_enabled': True }
        defense_params['reputation'] = { 'ema_alpha_increase': 0.1, 'ema_alpha_decrease': 0.6, 'penalty_flagged': 0.2, 'penalty_variance': 0.1, 'reward_clean': 0.1, 'floor_warning_threshold': 0.2, 'floor_target_value': 0.3, 'floor_probation_rounds': 5, 'initial_reputation': 0.5 }
        defense_params['mode'] = {
            'threshold_normal_to_alert': cfg.get("defense.mode.threshold-normal-to-alert", 0.15),
            'threshold_alert_to_defense': cfg.get("defense.mode.threshold-alert-to-defense", 0.30),
            'hysteresis_normal': cfg.get("defense.mode.hysteresis-normal", 0.05),
            'hysteresis_defense': cfg.get("defense.mode.hysteresis-defense", 0.10),
            'rep_gate_defense': cfg.get("defense.mode.rep-gate-defense", 0.5),
            'safe_weight_epsilon': cfg.get("defense.mode.safe-weight-epsilon", 1e-6),
            'initial_mode': "DEFENSE", # Force start in Defense
            'warmup_rounds': GLOBAL_WARMUP
        }

    strategy = FullPipelineStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=context.run_config.get("min-fit-clients", num_fit_clients),
        min_evaluate_clients=context.run_config.get("min-evaluate-clients", num_evaluate_clients),
        min_available_clients=context.run_config.get("min-available-clients", num_clients),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=ndarrays_to_parameters(get_parameters(get_model())),
        proximal_mu=proximal_mu,
        auto_save=cfg.get("auto-save", True),
        save_dir=cfg.get("save-dir", "saved_models"),
        save_interval=cfg.get("save-interval", 10),
        config_metadata={'num_clients': num_clients, 'attack_type': cfg.get("attack-type", "none")},
        start_round=0,
        enable_defense=enable_defense,
        defense_params=defense_params
    )
    
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))

app = ServerApp(server_fn=server_fn)