"""
Slow Poisoning Attack - FIXED V3
================================

CHIẾN LƯỢC TẤN CÔNG:
- Mỗi round, inject một lượng poison NHỎ vào gradient
- Formula: gradient = (1-rate) × benign + rate × (-benign)
- Ví dụ rate=0.3: gradient = 0.7×benign - 0.3×benign = 0.4×benign
- Kết quả: Gradient vẫn cùng hướng với benign, chỉ nhỏ hơn 60%

TẠI SAO KHÓ DETECT:
- Cosine(0.4×benign, benign) = 1.0 → Layer 2 không detect được
- Magnitude giảm 60% nhưng chưa đủ để MAD filter bắt
- Defense hiện tại dựa trên DIRECTION, không phải MAGNITUDE

BUG FIX V3:
- server_round có thể KHÔNG được truyền trong config
- Thêm fallback: dùng client round counter nếu server_round không có
"""

import numpy as np
from .base import AttackClient
from cifar_cnn.task import get_parameters


class SlowPoisoningClient(AttackClient):
    """
    Slow Poisoning Attack.
    
    Tham số:
        poison_rate: Tỷ lệ poison (0.3 = 30% poison, 70% benign)
        poison_direction: Hướng poison
            - "negative": Ngược với benign (default)
            - "random": Hướng ngẫu nhiên
            - "drift": Bắt đầu nhỏ, tăng dần
        warmup_rounds: Số round đầu hành xử benign (build trust)
    """
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 poison_rate=0.3, poison_direction="negative", warmup_rounds=5):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        
        # Lưu config
        self.poison_rate = poison_rate
        self.poison_direction = poison_direction if poison_direction else "negative"
        self.warmup_rounds = warmup_rounds
        
        # Client round counter (fallback nếu server_round không có trong config)
        self.round_counter = 0
        self._warned_fallback = False
        
        # Cho drift mode
        self.initial_rate = poison_rate
        self.max_rate = min(poison_rate * 3, 0.5)
        
        print(f"   📊 Slow Poison (FIXED V3):")
        print(f"      Rate: {poison_rate} ({poison_rate*100:.0f}% poison)")
        print(f"      Direction: {self.poison_direction}")
        print(f"      Warmup: {warmup_rounds} rounds")

    def fit(self, parameters, config):
        # Tăng client counter
        self.round_counter += 1
        
        # ═══════════════════════════════════════════════════════════════════
        # LẤY SERVER ROUND (có fallback)
        # ═══════════════════════════════════════════════════════════════════
        server_round = config.get("server_round", None)
        
        if server_round is None:
            # Fallback: dùng client counter
            server_round = self.round_counter
            if not self._warned_fallback:
                print(f"   ⚠️ [SlowPoison] server_round không có trong config!")
                print(f"      Dùng client counter làm fallback.")
                self._warned_fallback = True
        
        # ═══════════════════════════════════════════════════════════════════
        # TRAIN BENIGN (lấy gradient chuẩn)
        # ═══════════════════════════════════════════════════════════════════
        results = self.train_with_fedprox(parameters)
        trained_params = get_parameters(self.net)
        
        # ═══════════════════════════════════════════════════════════════════
        # WARMUP PHASE: Hành xử benign để build trust
        # ═══════════════════════════════════════════════════════════════════
        if server_round <= self.warmup_rounds:
            results["is_malicious"] = 1
            results["attack_active"] = 0  # Không attack
            results["poison_rate_actual"] = 0.0
            results["round_used"] = server_round
            return trained_params, len(self.trainloader.dataset), results
        
        # ═══════════════════════════════════════════════════════════════════
        # ATTACK PHASE: Inject poison vào gradient
        # ═══════════════════════════════════════════════════════════════════
        
        # Tính poison rate (drift mode tăng dần)
        if self.poison_direction == "drift":
            rounds_attacking = server_round - self.warmup_rounds
            progress = min(rounds_attacking / 30.0, 1.0)
            current_rate = self.initial_rate + progress * (self.max_rate - self.initial_rate)
        else:
            current_rate = self.poison_rate
        
        malicious_params = []
        total_cos = 0.0
        num_layers = 0
        
        for w_global, w_trained in zip(parameters, trained_params):
            # Chuyển sang numpy
            w_global = np.array(w_global, dtype=np.float32)
            w_trained = np.array(w_trained, dtype=np.float32)
            
            # Gradient benign
            u_benign = w_trained - w_global
            
            # Tạo poison direction
            if self.poison_direction in ["negative", "drift", "zero"]:
                u_poison = -u_benign
            elif self.poison_direction == "random":
                u_poison = np.random.randn(*u_benign.shape).astype(np.float32)
                # Normalize về cùng magnitude với benign
                norm_b = np.linalg.norm(u_benign)
                norm_p = np.linalg.norm(u_poison)
                if norm_p > 1e-10:
                    u_poison = u_poison / norm_p * norm_b
            elif self.poison_direction == "targeted":
                u_poison = np.abs(u_benign)
            else:
                u_poison = -u_benign
            
            # ═══════════════════════════════════════════════════════════════
            # MIX: gradient = (1-rate)×benign + rate×poison
            # ═══════════════════════════════════════════════════════════════
            u_mixed = (1 - current_rate) * u_benign + current_rate * u_poison
            
            # Track cosine cho debug
            norm_b = np.linalg.norm(u_benign)
            norm_m = np.linalg.norm(u_mixed)
            if norm_b > 1e-10 and norm_m > 1e-10:
                cos = np.dot(u_benign.flatten(), u_mixed.flatten()) / (norm_b * norm_m)
                total_cos += cos
                num_layers += 1
            
            # Tính weight mới
            w_mal = w_global + u_mixed
            malicious_params.append(w_mal.astype(np.float32))
        
        avg_cos = total_cos / max(num_layers, 1)
        
        # Log attack (chỉ log vài round đầu)
        if server_round <= self.warmup_rounds + 3 or server_round % 10 == 0:
            print(f"   🐌 [SlowPoison] Round {server_round}: ATTACKING!")
            print(f"      Rate: {current_rate:.2f}, Cosine: {avg_cos:.4f}")
        
        results["is_malicious"] = 1
        results["attack_active"] = 1
        results["poison_rate_actual"] = current_rate
        results["attack_type"] = f"slow_poison_{self.poison_direction}"
        results["round_used"] = server_round
        results["attack_cosine"] = avg_cos
        
        return malicious_params, len(self.trainloader.dataset), results


class GradualPoisoningClient(SlowPoisoningClient):
    """
    Gradual Poisoning: Rate tăng dần từ initial_rate đến final_rate.
    """
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 initial_rate=0.05, final_rate=0.3, ramp_rounds=20, warmup_rounds=5):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu,
                        poison_rate=initial_rate, poison_direction="negative",
                        warmup_rounds=warmup_rounds)
        
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.ramp_rounds = ramp_rounds
        
        print(f"   📊 Gradual Poison: {initial_rate} → {final_rate} over {ramp_rounds} rounds")

    def fit(self, parameters, config):
        self.round_counter += 1
        server_round = config.get("server_round", self.round_counter)
        
        # Tính rate dựa trên server round
        if server_round <= self.warmup_rounds:
            self.poison_rate = 0.0
        else:
            rounds_attacking = server_round - self.warmup_rounds
            progress = min(rounds_attacking / self.ramp_rounds, 1.0)
            self.poison_rate = self.initial_rate + progress * (self.final_rate - self.initial_rate)
        
        return super().fit(parameters, config)


class StealthyPoisoningClient(SlowPoisoningClient):
    """
    Stealthy Poisoning: Tự động điều chỉnh rate để giữ cosine >= target.
    """
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 target_cosine=0.9, max_poison_rate=0.3, warmup_rounds=5):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu,
                        poison_rate=0.1, poison_direction="negative",
                        warmup_rounds=warmup_rounds)
        
        self.target_cosine = target_cosine
        self.max_poison_rate = max_poison_rate
        
        print(f"   📊 Stealthy Poison: target_cos={target_cosine}, max_rate={max_poison_rate}")

    def fit(self, parameters, config):
        self.round_counter += 1
        server_round = config.get("server_round", self.round_counter)
        
        results = self.train_with_fedprox(parameters)
        trained_params = get_parameters(self.net)
        
        # Warmup
        if server_round <= self.warmup_rounds:
            results["is_malicious"] = 1
            results["attack_active"] = 0
            return trained_params, len(self.trainloader.dataset), results
        
        malicious_params = []
        
        for w_g, w_t in zip(parameters, trained_params):
            w_g = np.array(w_g, dtype=np.float32)
            w_t = np.array(w_t, dtype=np.float32)
            
            u_benign = w_t - w_g
            u_poison = -u_benign
            
            # Tìm rate tối đa mà vẫn giữ cosine >= target
            best_rate = 0.0
            for rate in np.linspace(0.05, self.max_poison_rate, 10):
                u_mixed = (1 - rate) * u_benign + rate * u_poison
                
                norm_b = np.linalg.norm(u_benign)
                norm_m = np.linalg.norm(u_mixed)
                
                if norm_b > 1e-10 and norm_m > 1e-10:
                    cos = np.dot(u_benign.flatten(), u_mixed.flatten()) / (norm_b * norm_m)
                    if cos >= self.target_cosine:
                        best_rate = rate
            
            u_mixed = (1 - best_rate) * u_benign + best_rate * u_poison
            w_mal = w_g + u_mixed
            malicious_params.append(w_mal.astype(np.float32))
        
        results["is_malicious"] = 1
        results["attack_active"] = 1
        results["attack_type"] = "stealthy_poison"
        results["round_used"] = server_round
        
        return malicious_params, len(self.trainloader.dataset), results