"""ALIE Attack - Corrected Implementation."""

import numpy as np
from scipy import stats
from .base import AttackClient
from cifar_cnn.task import get_parameters

class ALIEClient(AttackClient):
    """Attack 6: A Little Is Enough (ALIE) - Approximate Version.
    
    Paper: Baruch et al., NeurIPS 2019
    
    Original ALIE requires cross-client statistics (μ, σ).
    This implementation uses local approximation suitable for
    non-omniscient threat model.
    
    Công thức: u_mal = u - z * σ * sign(u)
    Mục tiêu: Tạo update nằm trong khoảng chấp nhận nhưng harmful
    """
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 z=1.5, n_clients=40, n_malicious=12, mode="alie"):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        
        # Tính z_max nếu không được cung cấp
        if z is None:
            self.z = stats.norm.ppf((n_clients - n_malicious) / 
                                    (n_clients - n_malicious + 1))
        else:
            self.z = z
        
        print(f"   📊 ALIE Attack: z = {self.z:.4f}")

    def fit(self, parameters, config):
        from cifar_cnn.task import get_parameters
        
        results = self.train_with_fedprox(parameters)
        trained_params = get_parameters(self.net)
        
        malicious_params = []
        
        for w_g, w_t in zip(parameters, trained_params):
            w_t = np.array(w_t, dtype=np.float32) if not isinstance(w_t, np.ndarray) else w_t
            w_g = np.array(w_g, dtype=np.float32) if not isinstance(w_g, np.ndarray) else w_g
            
            # Update vector
            u = w_t - w_g
            
            # Estimate σ (std of update)
            sigma = np.std(u)
            
            if sigma < 1e-10:
                w_mal = w_t
            else:
                # ALIE formula: u_mal = u - z × σ × sign(u)
                # Shift update theo hướng ngược, giữ trong khoảng chấp nhận
                u_mal = u - self.z * sigma * np.sign(u)
                w_mal = w_g + u_mal
            
            malicious_params.append(w_mal)
        
        results["is_malicious"] = 1
        return malicious_params, len(self.trainloader.dataset), results
