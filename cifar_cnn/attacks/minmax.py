"""Min-Max Attack Implementation."""

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
                 gamma=10.0, mode="minmax"):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        self.gamma = gamma

    def fit(self, parameters, config):
        # 1. Train bình thường
        results = self.train_with_fedprox(parameters)
        trained_params = get_parameters(self.net)
        global_params = parameters
        
        malicious_params = []
        
        # Min-Max Logic (Local Approximation):
        # Scale update vector lên một hệ số gamma lớn, 
        # nhưng hướng ngược lại hoặc nhiễu loạn để tối đa hóa variance 
        # mà vẫn giữ mean không đổi quá nhiều (tùy biến thể).
        # Ở đây ta implement phiên bản: Scaled Inverse Update (giống Byzantine nhưng mạnh hơn về độ lớn).
        
        for w_g, w_t in zip(global_params, trained_params):
            # Update vector
            u = w_t - w_g
            
            # Scale update lên gamma lần và đảo chiều
            # Mục tiêu: Kéo model global ra xa hội tụ nhất có thể
            u_mal = -self.gamma * u
            
            w_mal = w_g + u_mal
            malicious_params.append(w_mal)
            
        results["is_malicious"] = 1
        return malicious_params, len(self.trainloader.dataset), results