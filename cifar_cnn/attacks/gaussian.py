"""Gaussian Noise Attack."""

import numpy as np
from .base import AttackClient
from cifar_cnn.task import train, get_parameters, set_parameters


class GaussianNoiseClient(AttackClient):
    """Gaussian Noise Attack - Thêm noise vào gradients."""
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True,
                 noise_std=0.1):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision)
        self.noise_std = noise_std
    
    def fit(self, parameters, config):
        """Train normally, then add Gaussian noise."""
        set_parameters(self.net, parameters)
        results = train(
            self.net, self.trainloader, epochs=self.local_epochs,
            device=self.device, learning_rate=self.learning_rate,
            use_mixed_precision=self.use_mixed_precision
        )
        
        benign_params = get_parameters(self.net)
        noisy_params = []
        for p in benign_params:
            # Ensure p is numpy array
            if not isinstance(p, np.ndarray):
                p = np.array(p, dtype=np.float32)
            
            # Generate noise with same shape and dtype
            noise = np.random.standard_normal(p.shape).astype(p.dtype) * self.noise_std
            
            # Add noise to parameter
            noisy_param = p + noise
            noisy_params.append(noisy_param)
        
        return noisy_params, len(self.trainloader.dataset), results