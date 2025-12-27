"""Min-Max Attack Implementation."""

from cifar_cnn.attacks.byzantine import optimize_gamma
import numpy as np
from .base import AttackClient
from cifar_cnn.task import get_parameters

class MinMaxClient(AttackClient):
    """Attack 7: Min-Max Attack.
    Tấn công tối ưu hóa khoảng cách (Distance Maximization) trong giới hạn.
    Thường dùng để phá vỡ các cơ chế phòng thủ dựa trên khoảng cách Euclide.
    """
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 perturbation_type="unit", gamma_init=10.0, tau=0.01, mode="minmax"):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        
        self.perturbation_type = perturbation_type
        self.gamma_init = gamma_init
        self.tau = tau
        
        print(f"   ⚡ MinMax Attack (NDSS'21): ∇p={perturbation_type}")

    def fit(self, parameters, config):
        from cifar_cnn.task import get_parameters
        
        # Get benign updates nếu có (omniscient setting)
        benign_updates = config.get("benign_updates", None)
        
        results = self.train_with_fedprox(parameters)
        trained_params = get_parameters(self.net)
        
        # Compute local update
        local_update = []
        for w_g, w_t in zip(parameters, trained_params):
            w_t = np.array(w_t, dtype=np.float32) if not isinstance(w_t, np.ndarray) else w_t
            w_g = np.array(w_g, dtype=np.float32) if not isinstance(w_g, np.ndarray) else w_g
            local_update.append(w_t - w_g)
        
        if benign_updates is not None:
            # ===== FULL MINMAX (omniscient) =====
            # Benign aggregate: ∇b = avg(all benign updates)
            benign_aggregate = []
            for layer_idx in range(len(local_update)):
                layer_updates = np.stack([u[layer_idx] for u in benign_updates])
                benign_aggregate.append(np.mean(layer_updates, axis=0))
            
            # Perturbation
            perturbations = self._compute_perturbation(benign_aggregate, benign_updates)
            
            # Max distance between any two benign gradients
            max_benign_dist = self._compute_max_pairwise_distance(benign_updates)
            
            # Oracle: max_i ||∇m - ∇i|| ≤ max_{i,j} ||∇i - ∇j||
            def oracle(gamma):
                mal_update = [b + gamma * p for b, p in zip(benign_aggregate, perturbations)]
                max_dist = 0
                for benign_u in benign_updates:
                    dist = np.sqrt(sum(np.linalg.norm(m - b)**2 
                                      for m, b in zip(mal_update, benign_u)))
                    max_dist = max(max_dist, dist)
                return max_dist <= max_benign_dist
            
            gamma = optimize_gamma(oracle, self.gamma_init, self.tau)
            
        else:
            # ===== APPROXIMATE MINMAX (non-omniscient) =====
            benign_aggregate = local_update
            perturbations = self._compute_perturbation_single(local_update)
            
            # Estimate constraint
            estimated_max_dist = 2 * sum(np.linalg.norm(u) for u in local_update)
            
            def oracle(gamma):
                mal_update = [b + gamma * p for b, p in zip(benign_aggregate, perturbations)]
                return sum(np.linalg.norm(m) for m in mal_update) <= estimated_max_dist
            
            gamma = optimize_gamma(oracle, self.gamma_init, self.tau)
        
        # Craft: w_mal = w_global + ∇m
        malicious_params = []
        for w_g, b, p in zip(parameters, benign_aggregate, perturbations):
            w_g = np.array(w_g, dtype=np.float32) if not isinstance(w_g, np.ndarray) else w_g
            w_mal = w_g + b + gamma * p
            malicious_params.append(w_mal)
        
        results["is_malicious"] = 1
        results["gamma_used"] = gamma
        return malicious_params, len(self.trainloader.dataset), results
    
    def _compute_perturbation(self, benign_aggregate, all_updates):
        perturbations = []
        for layer_idx, b in enumerate(benign_aggregate):
            if self.perturbation_type == "unit":
                norm = np.linalg.norm(b)
                p = -b / norm if norm > 1e-10 else -np.sign(b)
            elif self.perturbation_type == "std":
                layer_updates = np.stack([u[layer_idx] for u in all_updates])
                p = -np.std(layer_updates, axis=0)
            else:
                p = -np.sign(b)
            perturbations.append(p)
        return perturbations
    
    def _compute_perturbation_single(self, local_update):
        perturbations = []
        for u in local_update:
            if self.perturbation_type == "unit":
                norm = np.linalg.norm(u)
                p = -u / norm if norm > 1e-10 else -np.sign(u)
            elif self.perturbation_type == "std":
                p = -np.std(u) * np.sign(u)
            else:
                p = -np.sign(u)
            perturbations.append(p)
        return perturbations
    
    def _compute_max_pairwise_distance(self, all_updates):
        max_dist = 0
        n = len(all_updates)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(sum(np.linalg.norm(all_updates[i][k] - all_updates[j][k])**2
                                  for k in range(len(all_updates[i]))))
                max_dist = max(max_dist, dist)
        return max_dist

class MinSumClient(AttackClient):
    """Min-Sum Attack - NDSS 2021.
    
    Constraint: Σ_i ||∇m - ∇i||² ≤ max_i Σ_j ||∇i - ∇j||²
    
    Constraint CHẶT hơn Min-Max → malicious gradient gần center hơn.
    Tốt hơn cho Krum, Bulyan (distance-based AGRs).
    """
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 perturbation_type="unit", gamma_init=10.0, tau=0.01, mode="minsum"):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        
        self.perturbation_type = perturbation_type
        self.gamma_init = gamma_init
        self.tau = tau
        
        print(f"   ⚡ MinSum Attack (NDSS'21): ∇p={perturbation_type}")

    def fit(self, parameters, config):
        from cifar_cnn.task import get_parameters
        
        benign_updates = config.get("benign_updates", None)
        
        results = self.train_with_fedprox(parameters)
        trained_params = get_parameters(self.net)
        
        local_update = []
        for w_g, w_t in zip(parameters, trained_params):
            w_t = np.array(w_t, dtype=np.float32) if not isinstance(w_t, np.ndarray) else w_t
            w_g = np.array(w_g, dtype=np.float32) if not isinstance(w_g, np.ndarray) else w_g
            local_update.append(w_t - w_g)
        
        if benign_updates is not None:
            # Full MinSum
            benign_aggregate = []
            for layer_idx in range(len(local_update)):
                layer_updates = np.stack([u[layer_idx] for u in benign_updates])
                benign_aggregate.append(np.mean(layer_updates, axis=0))
            
            perturbations = self._compute_perturbation(benign_aggregate, benign_updates)
            max_sum_dist = self._compute_max_sum_distance(benign_updates)
            
            # Oracle: Σ_i ||∇m - ∇i||² ≤ max_i Σ_j ||∇i - ∇j||²
            def oracle(gamma):
                mal_update = [b + gamma * p for b, p in zip(benign_aggregate, perturbations)]
                sum_dist = sum(sum(np.linalg.norm(m - b)**2 for m, b in zip(mal_update, u))
                              for u in benign_updates)
                return sum_dist <= max_sum_dist
            
            gamma = optimize_gamma(oracle, self.gamma_init, self.tau)
        else:
            # Approximate
            benign_aggregate = local_update
            perturbations = self._compute_perturbation_single(local_update)
            
            def oracle(gamma):
                mal_update = [b + gamma * p for b, p in zip(benign_aggregate, perturbations)]
                return sum(np.linalg.norm(m)**2 for m in mal_update) <= \
                       2 * sum(np.linalg.norm(u)**2 for u in local_update)
            
            gamma = optimize_gamma(oracle, self.gamma_init, self.tau)
        
        malicious_params = []
        for w_g, b, p in zip(parameters, benign_aggregate, perturbations):
            w_g = np.array(w_g, dtype=np.float32) if not isinstance(w_g, np.ndarray) else w_g
            w_mal = w_g + b + gamma * p
            malicious_params.append(w_mal)
        
        results["is_malicious"] = 1
        results["gamma_used"] = gamma
        return malicious_params, len(self.trainloader.dataset), results
    
    def _compute_perturbation(self, benign_aggregate, all_updates):
        perturbations = []
        for layer_idx, b in enumerate(benign_aggregate):
            if self.perturbation_type == "unit":
                norm = np.linalg.norm(b)
                p = -b / norm if norm > 1e-10 else -np.sign(b)
            elif self.perturbation_type == "std":
                layer_updates = np.stack([u[layer_idx] for u in all_updates])
                p = -np.std(layer_updates, axis=0)
            else:
                p = -np.sign(b)
            perturbations.append(p)
        return perturbations
    
    def _compute_perturbation_single(self, local_update):
        perturbations = []
        for u in local_update:
            if self.perturbation_type == "unit":
                norm = np.linalg.norm(u)
                p = -u / norm if norm > 1e-10 else -np.sign(u)
            elif self.perturbation_type == "std":
                p = -np.std(u) * np.sign(u)
            else:
                p = -np.sign(u)
            perturbations.append(p)
        return perturbations
    
    def _compute_max_sum_distance(self, all_updates):
        n = len(all_updates)
        max_sum = 0
        for i in range(n):
            sum_dist = sum(sum(np.linalg.norm(all_updates[i][k] - all_updates[j][k])**2
                              for k in range(len(all_updates[i])))
                          for j in range(n) if j != i)
            max_sum = max(max_sum, sum_dist)
        return max_sum
