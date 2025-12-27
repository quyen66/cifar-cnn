"""Sign Flipping Attack."""

from pyparsing import Callable
import numpy as np
from .base import AttackClient
from cifar_cnn.task import get_parameters

def optimize_gamma(oracle: Callable[[float], bool],
                   gamma_init: float = 10.0,
                   tau: float = 0.01,
                   max_iters: int = 50) -> float:
    """Algorithm 1 từ Fang paper - Binary search cho optimal γ."""
    step = gamma_init / 2
    gamma = gamma_init
    gamma_succ = 0.0
        
    for _ in range(max_iters):
        if abs(gamma_succ - gamma) <= tau and gamma_succ > 0:
            break
                
        if oracle(gamma):
             gamma_succ = gamma
             gamma = gamma + step / 2
        else:
            gamma = gamma - step / 2
            
        step = step / 2
        gamma = max(gamma, 0.01)
        
    return gamma_succ if gamma_succ > 0 else gamma_init / 10

class ByzantineClient(AttackClient):
    """Attack 3: Sign Flipping.
    Đảo dấu vector cập nhật (gradient).
    Nếu tham số dương, gửi về âm và ngược lại.
    """
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 perturbation_type="sign", gamma=4.0, optimize_gamma_flag=False,
                 gamma_init=10.0, mode="sign_flip"):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        self.mode = mode
        self.perturbation_type = perturbation_type
        self.gamma = gamma
        self.optimize_gamma_flag = optimize_gamma_flag
        self.gamma_init = gamma_init
        
        print(f"   ⚡ Sign Flip (Fang): ∇p={perturbation_type}, "
              f"γ={'optimize' if optimize_gamma_flag else gamma}")
    

    def fit(self, parameters, config):
        from cifar_cnn.task import get_parameters
        
        results = self.train_with_fedprox(parameters)
        trained_params = get_parameters(self.net)
        
        # Compute updates và perturbations
        updates = []
        perturbations = []
        
        for w_g, w_t in zip(parameters, trained_params):
            w_t = np.array(w_t, dtype=np.float32) if not isinstance(w_t, np.ndarray) else w_t
            w_g = np.array(w_g, dtype=np.float32) if not isinstance(w_g, np.ndarray) else w_g
            
            u = w_t - w_g
            updates.append(u)
            
            # Compute perturbation theo paper
            if self.perturbation_type == "sign":
                p = -np.sign(u)  # ∇p_sgn = -sign(∇b)
            elif self.perturbation_type == "unit":
                norm = np.linalg.norm(u)
                p = -u / norm if norm > 1e-10 else -np.sign(u)  # ∇p_uv = -∇b/||∇b||
            elif self.perturbation_type == "std":
                p = -np.abs(u)  # Approximate ∇p_std
            else:
                p = -np.sign(u)
            
            perturbations.append(p)
        
        # Optimize hoặc dùng fixed γ
        if self.optimize_gamma_flag:
            def oracle(gamma):
                # Simple heuristic: chấp nhận nếu norm không quá lớn
                total_norm = sum(np.linalg.norm(u + gamma * p) 
                                for u, p in zip(updates, perturbations))
                original_norm = sum(np.linalg.norm(u) for u in updates)
                return total_norm < 10 * original_norm
            
            gamma = optimize_gamma(oracle, self.gamma_init, tau=0.01)
        else:
            gamma = self.gamma
        
        # Craft malicious: ∇m = ∇b + γ × ∇p → w_mal = w_global + ∇m
        malicious_params = []
        for w_g, u, p in zip(parameters, updates, perturbations):
            w_g = np.array(w_g, dtype=np.float32) if not isinstance(w_g, np.ndarray) else w_g
            
            mal_update = u + gamma * p
            w_mal = w_g + mal_update
            malicious_params.append(w_mal)
        
        results["is_malicious"] = 1
        results["gamma_used"] = gamma
        return malicious_params, len(self.trainloader.dataset), results
