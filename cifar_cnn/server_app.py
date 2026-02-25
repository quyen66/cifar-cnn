"""
Server Application - Trusted Warm-up & Adaptive Defense (V2 - Soft Pipeline)
=============================================================================
FULL PRODUCTION VERSION with WARMUP BUG FIX:
- FIX: Warmup trusted client selection bug (seq_id vs partition_id mismatch)
- Added detailed debug logging
- Added Adaptive Hybrid Reference
"""

import numpy as np
import time
import os
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
import random
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
from cifar_cnn.defense.adaptive_reference import AdaptiveReferenceTracker 
from .round_logger import RoundLogger

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
                 log_filename="tuning_logs/detection_detail.log",
                 **kwargs) :
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
        self.last_h_score = 0.0
        
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
        self.warmup_client_proxies = None # cố định client proxies trong warmup
        log_dir = os.path.dirname(log_filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        self.round_logger = RoundLogger(log_file=log_filename)


        
        # Mapping sẽ được xây dựng khi cần từ actual client.cid
        print(f"   📋 Trusted Clients Set: {sorted(self.trusted_clients)}")
        
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
        self.confidence_scorer = ConfidenceScorer(**self.defense_params.get('confidence', {}), 
                                                  rescued_penalty=self.defense_params.get('confidence', {}).get('rescued_penalty', 0.3),
                                                  rescued_suspicious_penalty=self.defense_params.get('confidence', {}).get('rescued-suspicious-penalty', 0.7),
                                                  rescued_euclidean=self.defense_params.get('confidence', {}).get('rescued-euclidean', 0.5))
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
        
        adaptive_ref_params = self.defense_params.get('adaptive_reference', {})
        self.ref_tracker = AdaptiveReferenceTracker(
            alpha_base=adaptive_ref_params.get('alpha_base', 0.2),
            alpha_h_mult=adaptive_ref_params.get('alpha_h_mult', 0.5),
            alpha_min=adaptive_ref_params.get('alpha_min', 0.1),
            alpha_max=adaptive_ref_params.get('alpha_max', 0.8),
            momentum_decay=adaptive_ref_params.get('momentum_decay', 0.9),
            warmup_rounds=self.warmup_rounds,
            min_history_rounds=adaptive_ref_params.get('min_history_rounds', 3)
        )
        
    def _identify_malicious_clients(self) -> Set[int]:
        return set()

    def is_malicious(self, partition_id: int) -> bool:
        """
        Check if a partition_id is malicious.
        
        Args:
            partition_id: Sequential partition ID (0-based)
            
        Returns:
            True if malicious, False otherwise
        """
        num_clients = self.config_metadata.get('num_clients', 40)
        attack_ratio = self.config_metadata.get('attack_ratio', 0.0)
        num_malicious = int(num_clients * attack_ratio)
        
        # First N clients are malicious
        return partition_id < num_malicious

    def _aggregate_parameters_by_mode(
            self,
            results: List[Tuple[any, FitRes]],
            mode: str,
            reputations: Dict[any, float]
        ) -> List[np.ndarray]:
        """Aggregate parameters using Aggregator's mode-based aggregation."""
        if not results:
            return None
        
        all_params = [parameters_to_ndarrays(res.parameters) for _, res in results]
        num_layers = len(all_params[0])
        
        # Get reputation list
        client_ids = [cid for cid, _ in results]
        reputation_list = [reputations.get(cid, 1.0) for cid in client_ids]
        
        # Aggregate layer by layer
        aggregated = []
        for layer_idx in range(num_layers):
            # Flatten each layer
            layer_grads = [params[layer_idx].flatten() for params in all_params]
            
            # Aggregate
            agg_flat = self.aggregator.aggregate_by_mode(
                gradients=layer_grads,
                mode=mode,
                reputations=reputation_list
            )
            
            # Reshape
            aggregated.append(agg_flat.reshape(all_params[0][layer_idx].shape))
        
        return aggregated

    def _identify_trusted_clients(self) -> Set[int]:
        num_clients = self.config_metadata.get('num_clients', 40)
        attack_ratio = self.config_metadata.get('attack_ratio', 0.0)
        num_malicious = int(num_clients * attack_ratio)
        
        all_clients = list(range(num_clients))
        benign_candidates = all_clients[num_malicious:]
        
        target_trusted = 24
        num_trusted = min(target_trusted, len(benign_candidates))
        trusted = set(benign_candidates[:num_trusted])
        
        print(f"   📋 Trusted Clients Configuration:")
        print(f"      - Total clients: {num_clients}")
        print(f"      - Attack ratio: {attack_ratio}")
        print(f"      - Num malicious: {num_malicious} (partition_id 0-{num_malicious-1})")
        print(f"      - Trusted: partition_id {num_malicious}-{num_malicious + num_trusted - 1}")
        
        return trusted

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        if self.enable_defense and server_round <= self.warmup_rounds:
            print(f"\n⚡ [WARM-UP CONFIG] Round {server_round}: Forcing Trusted Clients.")
            
            all_clients_dict = client_manager.all()
            all_real_cids = sorted(list(all_clients_dict.keys()), key=lambda x: int(x))
            
            print(f"   🔍 [DEBUG] Connected Clients: {len(all_real_cids)}")
            print(f"   🔍 [DEBUG] Sample CIDs: {all_real_cids[:5]}...")
            
            if not all_real_cids:
                return [] 
            
            config = {"server_round": server_round}
            if self.on_fit_config_fn is not None:
                extra = self.on_fit_config_fn(server_round)
                if extra: config.update(extra)

            fit_ins = FitIns(parameters, config)


            if server_round == 1:
                # Round 1: Gửi đến TẤT CẢ clients để lấy mapping cid → partition_id
                print(f"   📌 [WARMUP] Round 1: Sending to ALL {len(all_real_cids)} clients to discover mapping")
                target_proxies = [all_clients_dict[cid] for cid in all_real_cids]
                # Lưu tạm, sẽ được filter trong aggregate_fit
                self.warmup_client_proxies = None  # Sẽ set trong aggregate_fit
                
            else:
                # Round 2-10: Dùng trusted proxies đã lưu từ aggregate_fit round 1
                print(f"   📌 [WARMUP] Round {server_round}: REUSING {len(self.warmup_client_proxies) if self.warmup_client_proxies else 0} trusted client proxies")
                
                if self.warmup_client_proxies:
                    target_proxies = self.warmup_client_proxies
                else:
                    print(f"   ⚠️  No saved proxies! Falling back to all clients.")
                    target_proxies = [all_clients_dict[cid] for cid in all_real_cids]            
            # Tạo target_clients với fit_ins mới (parameters mỗi round khác nhau)
            target_clients = [(proxy, fit_ins) for proxy in target_proxies]
            
            selected_cids = [c.cid for c, _ in target_clients[:5]]
            print(f"   ✅ Đã chọn {len(target_clients)} clients cho Warm-up.")
            print(f"   🔍 [DEBUG] Selected CIDs: {selected_cids}...")
            
            return target_clients

        all_clients_dict = client_manager.all()
        all_cids = sorted(list(all_clients_dict.keys()), key=lambda x: int(x))
        num_sample = max(int(len(all_cids) * self.fraction_fit), self.min_fit_clients)
        num_sample = min(num_sample, len(all_cids))
        sampled_cids = random.sample(all_cids, num_sample)

        config = {"server_round": server_round}
        if self.on_fit_config_fn is not None:
            extra = self.on_fit_config_fn(server_round)
            if extra: config.update(extra)

        fit_ins = FitIns(parameters, config)
        return [(all_clients_dict[cid], fit_ins) for cid in sampled_cids]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[any, FitRes]],
        failures: List[Tuple[any, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]] | None:
        """
        Aggregate fit results using Hybrid Adaptive Defense System.
        Updated: Support Hybrid H-Score (Gradient + Loss + Accuracy).
        """
        
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not allowed
        if not self.accept_failures and failures:
            return None, {}

        # =========================================================================
        # 1. PREPARATION & SORTING
        # =========================================================================
        # Sắp xếp results theo CID để đảm bảo thứ tự nhất quán giữa các list
        sorted_results = sorted(results, key=lambda x: int(x[0].cid))
        
        # Convert parameters to ndarrays (gradients)
        # Lưu ý: Chúng ta cần Global Model của vòng này để tính gradient (Update = W_client - W_global)
        # Flower strategy lưu model vòng này trong self.current_parameters (nếu được cập nhật đúng)
        global_weights = parameters_to_ndarrays(self.current_parameters)
        
        client_gradients = []
        client_losses = []     
        client_accuracies = []  
        seq_cids = []
        
        # Map CID to result for aggregation later
        cid_to_result = {}

        for idx, (client, fit_res) in enumerate(sorted_results):
            metrics = fit_res.metrics

        
        # for client, fit_res in sorted_results:
        #     cid = int(client.cid)
        #     seq_cids.append(cid)
        #     cid_to_result[cid] = (client, fit_res)
            
            # --- [NEW] EXTRACT METRICS (LOSS & ACCURACY) ---
            partition_id = int(metrics.get("partition_id", idx))
            seq_cids.append(partition_id)
            cid_to_result[partition_id] = (client, fit_res)
            # Lấy Loss (Default 0.0 nếu client quên gửi)
            loss = metrics.get("loss", 0.0)
            client_losses.append(loss)
            
            # Lấy Accuracy (Default 0.0 nếu client quên gửi)
            acc = metrics.get("accuracy", 0.0)
            client_accuracies.append(acc)
            
            # --- EXTRACT GRADIENTS ---
            client_params = parameters_to_ndarrays(fit_res.parameters)
            # Tính Update Vector: update = client_params - global_weights
            # (Hoặc dùng trực tiếp client_params nếu Layer 1 của bạn xử lý weights)
            # Ở đây ta tính update vector cho chuẩn logic Defense
            gradient = [w_c - w_g for w_c, w_g in zip(client_params, global_weights)]
            # Flatten for defense layers (46 arrays → 1 flat array)
            gradient_flat = np.concatenate([arr.flatten() for arr in gradient])
            client_gradients.append(gradient_flat)


        # Lấy Ground Truth (nếu chạy mô phỏng) để log metrics
        gt_malicious_batch = [self.is_malicious(cid) for cid in seq_cids]

        # =========================================================================
        # 2. WARMUP PHASE CHECK
        # =========================================================================
        # Trong giai đoạn warmup, bỏ qua defense hoặc dùng logic đơn giản
        if server_round <= self.warmup_rounds:
            print(f"\n🔥 WARMUP ROUND {server_round}/{self.warmup_rounds}: Aggregating all results...")
            
            # Update history cho NonIID Handler dù đang warmup (để xây dựng baseline)
            self.noniid_handler.compute_heterogeneity_score(client_gradients, client_losses, client_accuracies)
            for i, cid in enumerate(seq_cids):
                self.noniid_handler.compute_baseline_deviation(cid, client_gradients[i])
            
            # Update Adaptive Reference (để sẵn sàng cho vòng 11)
            # Dùng FedAvg đơn giản cho warmup
            aggregated_ndarrays = self._aggregate_parameters_by_mode(
                results=sorted_results, 
                mode="NORMAL", 
                reputations={cid: 1.0 for cid in seq_cids}
            )
            
            # Save weights for next round
            self.current_parameters = ndarrays_to_parameters(aggregated_ndarrays)
            return self.initial_parameters, {"mode": "WARMUP"}

        # =========================================================================
        # 3. DEFENSE PIPELINE EXECUTION
        # =========================================================================
        
        # --- Bước 1: Layer 1 Detection (Enhanced DBSCAN) ---
        layer1_results = self.layer1_detector.detect(
            gradients=client_gradients,
            client_ids=seq_cids,
            current_round=server_round,
            is_malicious_ground_truth=gt_malicious_batch # For logging only
        )

        # --- Bước 2: Layer 2 Detection (Distance + Direction) ---
        # Lấy tham chiếu thích ứng (Adaptive Reference)
        ref_vector = None
        if self.ref_tracker:
            ref_vector = self.ref_tracker.compute_adaptive_reference(
                current_gradients=client_gradients,
                h_score=self.last_h_score,  
                current_round=server_round
            )
        
        layer2_status, suspicion_levels = self.layer2_detector.detect(
            gradients=client_gradients,
            client_ids=seq_cids,
            layer1_results=layer1_results,
            current_round=server_round,
            is_malicious_ground_truth=gt_malicious_batch,
            external_reference=ref_vector
        )

        # --- Bước 3: Non-IID Handler (Tính H-Score Hybrid) ---
        # [QUAN TRỌNG] Truyền Loss và Accuracy vào đây
        H = self.noniid_handler.compute_heterogeneity_score(
            client_gradients=client_gradients,
            client_losses=client_losses,     
            client_accuracies=client_accuracies 
        )
        self.last_h_score = H
        
        # Tính Baseline Deviation cho từng client
        baseline_deviations = {}
        for i, cid in enumerate(seq_cids):
            delta = self.noniid_handler.compute_baseline_deviation(
                client_id=cid,
                current_gradient=client_gradients[i]
            )
            baseline_deviations[cid] = delta

        # --- Bước 4: Confidence Scoring ---
        confidence_scores = self.confidence_scorer.calculate_scores(
            client_ids=seq_cids,
            layer1_results=layer1_results,
            layer2_status=layer2_status,
            suspicion_levels=suspicion_levels,
            baseline_deviations=baseline_deviations
        )

        # --- Bước 5: Reputation System Update ---
        # Dùng median gradient của round này làm mốc so sánh hướng (P)
        grad_matrix = np.vstack([g.flatten() for g in client_gradients])
        median_grad = np.median(grad_matrix, axis=0)
        
        current_reputations = {}
        old_reps = self.reputation_system.get_all_reputations() # For debug comparison

        for i, cid in enumerate(seq_cids):
            # Map status từ Layer 1/2 sang Reputation Status
            status = ClientStatus.CLEAN
            if layer1_results.get(cid) == "REJECTED" or layer2_status.get(cid) == "REJECTED":
                status = ClientStatus.REJECTED
            elif layer1_results.get(cid) == "FLAGGED": # Rescued but Flagged
                 status = ClientStatus.SUSPICIOUS
            elif suspicion_levels.get(cid) == "suspicious": # L2 Suspicious
                 status = ClientStatus.SUSPICIOUS
            
            # Cập nhật Reputation
            new_rep = self.reputation_system.update(
                client_id=cid,
                gradient=client_gradients[i].flatten(),
                grad_median=median_grad,
                status=status,
                heterogeneity_score=H
            )
            current_reputations[cid] = new_rep

        # --- Bước 6: Mode Controller ---
        # Compute threat_ratio (ρ) and detected_clients for ModeController
        # detected_clients = clients rejected by Layer1 OR Layer2
        detected_clients = [
            cid for cid in seq_cids
            if layer1_results.get(cid) == "REJECTED" or layer2_status.get(cid) == "REJECTED"
        ]
        threat_ratio = len(detected_clients) / len(seq_cids) if seq_cids else 0.0

        current_mode = self.mode_controller.update_mode(
            threat_ratio=threat_ratio,
            detected_clients=detected_clients,
            reputations=current_reputations,
            current_round=server_round
        )

        # --- Bước 7: Two-Stage Filtering ---
        trusted_clients, all_filtered, filter_stats = self.two_stage_filter.filter_clients(
            client_ids=seq_cids,
            confidence_scores=confidence_scores,
            reputations=current_reputations,
            mode=current_mode,
            H=H,
            noniid_handler=self.noniid_handler
        )
        
        # =========================================================================
        # 3b. ROUND LOGGING — chi tiết detection per-round
        # =========================================================================
        # Expose rescue_cosine_threshold vào layer2_stats cho Table 3 header
        layer2_stats_for_log = dict(self.layer2_detector.last_stats)
        layer2_stats_for_log["rescue_cosine_threshold"] = getattr(
            self.layer2_detector, 'rescue_cosine_threshold', 0.8)

        self.round_logger.log_round(
            round_num=server_round,
            mode=current_mode,
            H=H,
            seq_cids=seq_cids,
            gt_malicious=gt_malicious_batch,
            layer1_result=self.layer1_detector.last_result,
            layer2_stats=layer2_stats_for_log,
            layer2_drift_info=self.layer2_detector.last_drift_info,
            layer2_status=layer2_status,
            suspicion_levels=suspicion_levels,
            confidence_scores=confidence_scores,
            reputations=current_reputations,
            filter_stats=filter_stats,
            ref_alpha=self.ref_tracker.last_alpha_used if self.ref_tracker else 0.0,
            accuracy=None   # set later nếu có evaluate result
        )

        # =========================================================================
        # 4. AGGREGATION & UPDATE
        # =========================================================================
        
        # Chỉ lấy results của Trusted Clients
        trusted_results = []
        for cid in trusted_clients:
            if cid in cid_to_result:
                trusted_results.append(cid_to_result[cid])

        # Log kết quả cuối cùng của vòng
        print(f"\n   🔒 Two-Stage Filter (Mode={current_mode}):")
        print(f"      Hard Filter: {filter_stats.get('hard_filtered_count', 0)} rejected")
        print(f"      Soft Filter: {filter_stats.get('soft_filtered_count', 0)} rejected")
        print(f"      Detection Rejected (L1+L2): {len([c for c in seq_cids if layer1_results[c]=='REJECTED' or layer2_status[c]=='REJECTED'])}")
        print(f"      Total Filtered: {len(all_filtered)}")

        # Nếu không còn client nào tin cậy (trường hợp cực đoan), dùng lại global cũ
        if not trusted_results:
            print("⚠️ CRITICAL: No trusted clients! Skipping aggregation.")
            return self.current_parameters, {}

        # Tổng hợp tham số (Aggregation)
        aggregated_ndarrays = self._aggregate_parameters_by_mode(
            results=trusted_results,
            mode=current_mode,
            reputations=current_reputations
        )
        
        # Update Adaptive Reference Tracker (cho vòng sau)
        # H càng cao -> càng tin vào lịch sử (Historical Momentum)
        if self.ref_tracker:
            self.ref_tracker.update_momentum(
                aggregated_gradient=aggregated_ndarrays,
                current_round=server_round
            )

        
        # Lưu lại Global Weights mới cho vòng sau
        self.current_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # Trả về metrics để Flower log lại
        metrics_aggregated = {
            "H_score": H,
            "Mode": current_mode,
            "Trusted_Count": len(trusted_clients),
            "Filtered_Count": len(all_filtered)
        }

        return self.current_parameters, metrics_aggregated

    def _update_parameters(self, agg_grad, sample_params):
        if agg_grad is None:
            return
            
        # 1. Lấy cấu trúc shape từ sample parameters
        sample_shapes = [p.shape for p in parameters_to_ndarrays(sample_params)]
        
        # 2. Lấy trọng số Global cũ (W_old)
        global_old = parameters_to_ndarrays(self.current_parameters)
        
        new_params = []
        offset = 0
        
        for i, shape in enumerate(sample_shapes):
            size = int(np.prod(shape))
            # Lấy phần update tương ứng (Delta W) từ kết quả Aggregation
            update_part = agg_grad[offset:offset+size].reshape(shape)
            
            # Cộng thẳng Update vào Weight cũ
            # W_new = W_old + Delta_W_agg
            new_param_val = global_old[i] + update_part
            
            new_params.append(new_param_val)
            offset += size
        
        # Cập nhật tham số toàn cục mới
        self.current_parameters = ndarrays_to_parameters(new_params)
        
    def _calculate_metrics(self, server_round, seq_cids, gt_malicious, detected, H, threat_ratio, elapsed):
        """
        Calculate and print detection metrics.
        
        Output includes:
        - Human-readable metrics (Precision, Recall, F1, FPR as percentages)
        - Machine-parseable summary line for automation scripts
        """
        if not seq_cids:
            print(f"\n📊 METRICS ROUND {server_round}")
            print(f"   (Warmup - No detection metrics)")
            return
            
        tp = fp = fn = tn = 0
        
        for i, seq_id in enumerate(seq_cids):
            is_mal = gt_malicious[i]
            is_detected = seq_id in detected
            
            if is_mal and is_detected:
                tp += 1
            elif is_mal and not is_detected:
                fn += 1
            elif not is_mal and is_detected:
                fp += 1
            else:
                tn += 1
        
        self.total_tp += tp
        self.total_fp += fp
        
        # Calculate all metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        detection_rate = recall  # Detection Rate = Recall = TP / (TP + FN)
        
        # Human-readable output
        print(f"\n📊 METRICS ROUND {server_round}")
        print(f"   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"   Detection Rate: {detection_rate:.2%}")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")
        print(f"   F1 Score: {f1:.2%}")
        print(f"   False Positive Rate: {fpr:.2%}")
        
        # 🆕 Machine-parseable summary line (for run_param_tests.py)
        print(f"   [METRICS] Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, FPR={fpr:.4f}")
        
        print(f"   H={H:.4f}, Threat Ratio={threat_ratio:.2%}")
        print(f"   Time: {elapsed:.2f}s")
        
        mal_indices = [i for i, m in enumerate(gt_malicious) if m]
        if mal_indices:
            detected_mal = [seq_cids[i] for i in mal_indices if seq_cids[i] in detected]
            print(f"   True Positives: {len(detected_mal)}/{len(mal_indices)} malicious detected")

    def _save_checkpoint(self, server_round):
        if not self.auto_save or self.current_parameters is None:
            return
            
        os.makedirs(self.save_dir, exist_ok=True)
        
        params = parameters_to_ndarrays(self.current_parameters)
        net = get_model()
        set_parameters(net, params)
        
        path = os.path.join(self.save_dir, f"round_{server_round}.pt")
        try:
            torch.save(net.state_dict(), path)
            print(f"   💾 Saved checkpoint: {path}")
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
    
    log_filename = context.run_config.get("detection-log-path", "tuning_logs/detection_detail.log")
        
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
            'zero_cluster_action': context.run_config.get("defense.layer1.zero-cluster-action", "accept_all"),
        }
        defense_params['layer2'] = {
            'distance_multiplier': float(context.run_config.get("defense.layer2.distance-multiplier", 1.5)),
            'cosine_threshold': float(context.run_config.get("defense.layer2.cosine-threshold", 0.3)),
            'enable_rescue': context.run_config.get("defense.layer2.enable-rescue", False),
            'rescue_cosine_threshold': float(context.run_config.get("defense.layer2.rescue-cosine-threshold", 0.7)),
            'require_distance_pass_for_rescue': context.run_config.get("defense.layer2.require-distance-pass", True),
            'enable_drift_detection' : context.run_config.get("defense.layer2.enable-drift-detection", False),
            'drift_threshold' : float(context.run_config.get("defense.layer2.drift-threshold", 0.7)),          
            'drift_window' : int(context.run_config.get("defense.layer2.drift-window", 0.7)) 
        }
        defense_params['noniid'] = {
            'adjustment_factor': float(context.run_config.get("defense.noniid.adjustment-factor", 0.4)),
            'theta_adj_clip_min': float(context.run_config.get("defense.noniid.theta-adj-clip-min", 0.5)),
            'theta_adj_clip_max': float(context.run_config.get("defense.noniid.theta-adj-clip-max", 0.9)),
            'baseline_window_size': int(context.run_config.get("defense.noniid.baseline-window-size", 10)),
            'delta_norm_weight': float(context.run_config.get("defense.noniid.delta-norm-weight", 0.5)),
            'delta_direction_weight': float(context.run_config.get("defense.noniid.delta-direction-weight", 0.5)),
            'weight_h_grad': float(context.run_config.get("defense.noniid.weight-h-grad", 0.4)),
            'weight_h_loss': float(context.run_config.get("defense.noniid.weight-h-loss", 0.4)),
            'weight_h_acc': float(context.run_config.get("defense.noniid.weight-h-acc", 0.2)),
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
        
        defense_params['adaptive_reference'] = {
            'alpha_base': float(context.run_config.get("defense.adaptive-reference.alpha-base", 0.2)),
            'alpha_h_mult': float(context.run_config.get("defense.adaptive-reference.alpha-h-mult", 0.5)),
            'alpha_min': float(context.run_config.get("defense.adaptive-reference.alpha-min", 0.1)),
            'alpha_max': float(context.run_config.get("defense.adaptive-reference.alpha-max", 0.8)),
            'momentum_decay': float(context.run_config.get("defense.adaptive-reference.momentum-decay", 0.9)),
            'min_history_rounds': int(context.run_config.get("defense.adaptive-reference.min-history-rounds", 3)),
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
        warmup_rounds=warmup_rounds,
        log_filename = log_filename
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)
    
app = ServerApp(server_fn=server_fn)