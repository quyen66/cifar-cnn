"""Noise Injection Attacks: Random (Uniform) and Gaussian."""

import numpy as np
import torch
from .base import AttackClient
from cifar_cnn.task import get_parameters

class RandomNoiseClient(AttackClient):
    """Attack 1: Random Noise (Uniform Distribution).
    Cộng nhiễu phân phối đều (Uniform) vào weights/gradients.
    """
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 scale=0.5, mode="random_noise"):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        self.scale = scale

    def fit(self, parameters, config):
        # 1. Train bình thường
        results = self.train_with_fedprox(parameters)
        
        # 2. Lấy tham số sạch
        benign_params = get_parameters(self.net)
        
        # 3. Cộng nhiễu Uniform [-scale, scale]
        noisy_params = []
        for p in benign_params:
            if not isinstance(p, np.ndarray):
                p = np.array(p, dtype=np.float32)
            
            # Tạo nhiễu Uniform
            noise = np.random.uniform(-self.scale, self.scale, p.shape).astype(p.dtype)
            noisy_params.append(p + noise)
            
        results["is_malicious"] = 1
        return noisy_params, len(self.trainloader.dataset), results


class GaussianNoiseClient(AttackClient):
    """Attack 2: Gaussian Noise.
    Cộng nhiễu phân phối chuẩn (Gaussian/Normal) vào weights/gradients.
    """
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 noise_std=0.1, mode="gaussian_noise"):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        self.std = noise_std
    
    def fit(self, parameters, config):
        # 1. Train bình thường
        results = self.train_with_fedprox(parameters)
        
        # 2. Lấy tham số sạch
        benign_params = get_parameters(self.net)
        
        # 3. Cộng nhiễu Gaussian N(0, std)
        noisy_params = []
        for p in benign_params:
            if not isinstance(p, np.ndarray):
                p = np.array(p, dtype=np.float32)
            
            # Tạo nhiễu Gaussian
            noise = np.random.standard_normal(p.shape).astype(p.dtype) * self.std
            noisy_params.append(p + noise)
            
        results["is_malicious"] = 1
        return noisy_params, len(self.trainloader.dataset), results