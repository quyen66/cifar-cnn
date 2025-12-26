"""Sign Flipping Attack."""

import numpy as np
from .base import AttackClient
from cifar_cnn.task import get_parameters

class ByzantineClient(AttackClient):
    """Attack 3: Sign Flipping.
    Đảo dấu vector cập nhật (gradient).
    Nếu tham số dương, gửi về âm và ngược lại.
    """
    def __init__(self, net, trainloader, testloader, device, local_epochs, 
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 mode="sign_flip"):
        super().__init__(net, trainloader, testloader, device, local_epochs, 
                         learning_rate, use_mixed_precision, proximal_mu)
        self.mode = mode
        
    def fit(self, parameters, config):
        # 1. Train bình thường để lấy gradient thực tế
        results = self.train_with_fedprox(parameters)
        
        # 2. Lấy tham số sau train
        trained_params = get_parameters(self.net)
        
        # 3. Tính toán update vector (model_trained - model_global)
        # Lưu ý: Flower gửi về parameters mới, server sẽ trừ đi để lấy update.
        # Để thực hiện Sign Flip hiệu quả trong cấu trúc gửi Parameters:
        # Target: W_malicious = W_global - alpha * (W_trained - W_global)
        # Tức là thay vì di chuyển về hướng tốt, ta di chuyển ngược lại.
        
        # Lấy tham số global (input parameters)
        global_params = parameters
        
        malicious_params = []
        for w_g, w_t in zip(global_params, trained_params):
            # Tính update vector
            update = w_t - w_g 
            
            # Flip sign: Trừ update thay vì cộng (hoặc nhân -1)
            # Hệ số scale có thể > 1 để tấn công mạnh hơn
            scale = 4.0  # Hệ số tấn công
            w_mal = w_g - (scale * update)
            
            malicious_params.append(w_mal)
        results["is_malicious"] = 1
        return malicious_params, len(self.trainloader.dataset), results