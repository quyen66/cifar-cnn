"""
Server Application - Trusted Warm-up & Adaptive Defense (V2 - Soft Pipeline)
=============================================================================
FULL PRODUCTION VERSION with WARMUP BUG FIX:
- FIX: Warmup trusted client selection bug (seq_id vs partition_id mismatch)
- Added detailed debug logging
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
        self.warmup_client_proxies = None # cố định client proxies trong warmup
        
        # ===== FIX V2: Pre-init mapping KHÔNG CẦN vì sẽ dùng trực tiếp partition_id =====
        # Mapping sẽ được xây dựng khi cần từ actual client.cid
        print(f"   📋 Trusted Clients Set: {sorted(self.trusted_clients)}")
        # ===== END FIX =====
        
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
                                                  rescued_suspicious_penalty=self.defense_params.get('confidence', {}).get('rescued-suspicious-penalty', 0.7))
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
        return set()

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
            
            config = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)
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

        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[any, FitRes]],
        failures: List[Tuple[any, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]] | None:
        start_time = time.time()
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
            
            # ===== Debug logging để xem client.cid =====
            result_cids = [c.cid for c, _ in results]
            print(f"   🔍 [DEBUG] Received {len(results)} results")
            print(f"   🔍 [DEBUG] Result CIDs: {result_cids[:5]}...")
            # ===== END DEBUG =====
            
            trusted_grads = []
            trusted_reps = []
            trusted_count = 0
            malicious_count = 0
            trusted_proxies_round1 = []  # Lưu trusted proxies
            
            global_weights_warmup = parameters_to_ndarrays(self.current_parameters)
            global_flat = np.concatenate([x.flatten() for x in global_weights_warmup])
            print(f"   📐 Global weights norm: {np.linalg.norm(global_flat):.4f}")
            
            for client, res in results:
                # ===== Read partition_id from metrics =====
                partition_id = res.metrics.get("partition_id", None)
                
                if partition_id is None:
                    continue
                
                partition_id = int(partition_id)
                self.client_id_to_sequential[client.cid] = partition_id
                self.sequential_to_client_id[partition_id] = client.cid
                
                is_mal = bool(res.metrics.get("is_malicious", 0))
                if is_mal:
                    malicious_count += 1
                
                if partition_id in self.trusted_clients:
                    trusted_count += 1
                    
                    if is_mal:
                        print(f"   WARNING: Partition {partition_id} trusted but malicious!")
                        continue
                    
                    # Lưu proxy của trusted client
                    trusted_proxies_round1.append(client)
                    
                    p = parameters_to_ndarrays(res.parameters)
                    w_client_flat = np.concatenate([x.flatten() for x in p])
                    update_flat = w_client_flat - global_flat  # UPDATE = W_client - W_global
                    trusted_grads.append(update_flat)
                    
                    self.noniid_handler.update_client_gradient(partition_id, update_flat) 
                    self.reputation_system.initialize_client(partition_id, is_trusted=True)
                    trusted_reps.append(1.0)
                
            if server_round == 1:
                self.warmup_client_proxies = trusted_proxies_round1
                print(f"   💾 [FIX V8] Saved {len(trusted_proxies_round1)} TRUSTED proxies for rounds 2-10")
                       
            
            print(f"   📊 Warmup Stats:")
            print(f"      - Results received: {len(results)}")
            print(f"      - Matched trusted set: {trusted_count}")
            print(f"      - Used for aggregation: {len(trusted_grads)}")
            print(f"      - Malicious in batch: {malicious_count}")

            if not trusted_grads:
                print("   ⚠️  No trusted clients sampled in Warm-up. Skipping update.")
                print(f"   🔍 [DEBUG] Trusted set: {sorted(self.trusted_clients)}")
                partition_ids = [res.metrics.get("partition_id", "N/A") for _, res in results]
                print(f"   [DEBUG] Partition IDs from metrics: {partition_ids}")
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
            
            print(f"   ✅ Warmup aggregation successful with {len(trusted_grads)} trusted clients")
            print(f"   Heterogeneity Score H = 0.000 (Warmup)") 
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
        
        global_weights = parameters_to_ndarrays(self.current_parameters)
        global_flat = np.concatenate([x.flatten() for x in global_weights])

        for c, res in results:
            p = parameters_to_ndarrays(res.parameters)
            w_client_flat = np.concatenate([x.flatten() for x in p])
            # Tính vector cập nhật (Update / Pseudo-gradient)
            # update = w_client - w_global
            update_flat = w_client_flat - global_flat
            full_gradients.append(update_flat)
            
            real_cid_str = c.cid
            
            # ===== FIX V3: Read partition_id from metrics =====
            partition_id = res.metrics.get("partition_id", None)
            if partition_id is not None:
                seq_id = int(partition_id)
                self.client_id_to_sequential[c.cid] = seq_id
                self.sequential_to_client_id[seq_id] = c.cid
            else:
                seq_id = self.client_id_to_sequential.get(c.cid, len(self.client_id_to_sequential))
                if c.cid not in self.client_id_to_sequential:
                    self.client_id_to_sequential[c.cid] = seq_id
            # ===== END FIX V3 =====
            
            seq_cids.append(seq_id)
            
            is_mal = bool(res.metrics.get("is_malicious", 0))
            gt_malicious_batch.append(is_mal)

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

        # 2. Non-IID Handler: Calculate H Score
        for local_idx, seq_id in enumerate(seq_cids):
            self.noniid_handler.update_client_gradient(seq_id, full_gradients[local_idx])
        
        H = self.noniid_handler.compute_heterogeneity_score(full_gradients, seq_cids)
        print(f"\n   🌐 Heterogeneity Score H = {H:.4f}")
        
        # 3. Confidence Scoring - FIX V4: Use correct method calculate_scores
        baseline_deviations = {}
        for i, seq_id in enumerate(seq_cids):
            detail = self.noniid_handler.compute_baseline_deviation_detailed(
                seq_id, full_gradients[i], reference_vector
            )
            baseline_deviations[seq_id] = detail['delta_combined']
        
        confidence_scores = self.confidence_scorer.calculate_scores(
            client_ids=seq_cids,
            layer1_results=l1_res,
            layer2_status=l2_status,
            suspicion_levels=l2_suspicion,
            baseline_deviations=baseline_deviations
        )

        # 4. Determine client status and update reputation
        client_statuses = {}
        for seq_id in seq_cids:
            # FIX V4: l1_res returns string, l2_status is REJECTED/ACCEPTED, l2_suspicion is clean/suspicious
            l1_status = l1_res.get(seq_id, 'ACCEPTED')
            l2_stat = l2_status.get(seq_id, 'ACCEPTED')
            l2_susp = l2_suspicion.get(seq_id, 'clean')
            
            if l1_status == 'REJECTED' or l2_stat == 'REJECTED':
                status = ClientStatus.REJECTED
            elif l1_status == 'FLAGGED' or l2_susp == 'suspicious':
                status = ClientStatus.SUSPICIOUS
            else:
                status = ClientStatus.CLEAN
            client_statuses[seq_id] = status
        
        # Update reputations - FIX V5: Use correct method name and params
        grad_median = np.median(np.vstack(full_gradients), axis=0)
        for local_idx, seq_id in enumerate(seq_cids):
            grad = full_gradients[local_idx]
            status = client_statuses[seq_id]
            
            self.reputation_system.update(
                client_id=seq_id,
                gradient=grad,
                grad_median=grad_median,
                status=status,
                heterogeneity_score=H,
                current_round=server_round
            )

        # 5. Two-Stage Filtering - FIX V5: Use correct method name and params
        reputations = {seq_id: self.reputation_system.get_reputation(seq_id) 
                      for seq_id in seq_cids}
        
        current_mode = self.mode_controller.get_current_mode()
        
        trusted_set, all_filtered, filter_stats = self.two_stage_filter.filter_clients(
            client_ids=seq_cids,
            confidence_scores=confidence_scores,
            reputations=reputations,
            mode=current_mode,
            H=H,
            noniid_handler=self.noniid_handler
        )
        
        hard_filtered = set(filter_stats.get('hard_filtered_ids', []))
        soft_filtered = set(filter_stats.get('soft_filtered_ids', []))
        
        detection_rejected = {seq_id for seq_id in seq_cids 
                             if client_statuses.get(seq_id) == ClientStatus.REJECTED}
        all_filtered = all_filtered | detection_rejected  # Union 2 sets
        
        print(f"\n   🔒 Two-Stage Filter (Mode={current_mode}):")
        print(f"      Hard Filter: {len(hard_filtered)} rejected")
        print(f"      Soft Filter: {len(soft_filtered)} rejected")
        print(f"      Detection Rejected (L1+L2): {len(detection_rejected)}")
        print(f"      Total Filtered: {len(all_filtered)}")
        
        # 6. Mode Controller Update - FIX V4: Use update_mode with correct signature
        threat_ratio = len(all_filtered) / len(seq_cids) if seq_cids else 0
        new_mode = self.mode_controller.update_mode(
            threat_ratio=threat_ratio,
            detected_clients=list(all_filtered),
            reputations=reputations,
            current_round=server_round
        )
        
        if new_mode != current_mode:
            print(f"   ⚡ Mode Changed: {current_mode} → {new_mode}")

        # 7. Aggregation
        clean_indices = [i for i, seq_id in enumerate(seq_cids) if seq_id not in all_filtered]
        clean_grads = [full_gradients[i] for i in clean_indices]
        clean_reps = [reputations[seq_cids[i]] for i in clean_indices]
        
        print(f"\n   📦 Aggregation:")
        print(f"      Clean clients: {len(clean_indices)}/{len(seq_cids)}")
        print(f"      Mode: {new_mode}")
        
        if not clean_grads:
            print("   ⚠️  No clean clients! Using median of all gradients.")
            agg_grad = self.aggregator.coordinate_median(full_gradients)
        elif new_mode == "NORMAL":
            agg_grad = self.aggregator.weighted_average(clean_grads, clean_reps)
        elif new_mode == "ALERT":
            agg_grad = self.aggregator.trimmed_mean(clean_grads, threat_ratio)
        else:  # DEFENSE
            agg_grad = self.aggregator.coordinate_median(clean_grads)

        self._update_parameters(agg_grad, results[0][1].parameters)
        
        try:
            if agg_grad is not None:
                if isinstance(agg_grad, list):
                    self.previous_full_grad = np.concatenate([x.flatten() for x in agg_grad])
                else:
                    self.previous_full_grad = agg_grad.flatten()
        except Exception as e:
            print(f"   ⚠️ Warning: Momentum update failed: {e}")

        # 8. Metrics Calculation
        self._calculate_metrics(
            server_round, 
            seq_cids, 
            gt_malicious_batch, 
            all_filtered,
            H, 
            threat_ratio,
            time.time() - start_time
        )
        
        if should_save: 
            self._save_checkpoint(server_round)
        
        print(f"⏱️ [ROUND {server_round}] Server Processing Time: {time.time() - start_time:.4f}s")
        return self.current_parameters, {}

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
            
            # [CHÍNH XÁC] Cộng thẳng Update vào Weight cũ
            # W_new = W_old + Delta_W_agg
            new_param_val = global_old[i] + update_part
            
            new_params.append(new_param_val)
            offset += size
        
        # Cập nhật tham số toàn cục mới
        self.current_parameters = ndarrays_to_parameters(new_params)
        
    def _calculate_metrics(self, server_round, seq_cids, gt_malicious, detected, H, threat_ratio, elapsed):
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
        
        print(f"\n📊 METRICS ROUND {server_round}")
        print(f"   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        
        if tp + fn > 0:
            detection_rate = tp / (tp + fn)
            print(f"   Detection Rate: {detection_rate:.2%}")
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"   Precision: {precision:.2%}")
            
        if fp + tn > 0:
            fpr = fp / (fp + tn)
            print(f"   False Positive Rate: {fpr:.2%}")
        
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
            'cosine_threshold': float(context.run_config.get("defense.layer2.cosine-threshold", 0.3)),
            #'enable_rescue': context.run_config.get("defense.layer2.enable-rescue", False) 
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