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
        noisy_params = [
            p + np.random.randn(*p.shape).astype(np.float32) * self.noise_std
            for p in benign_params
        ]
        
        return noisy_params, len(self.trainloader.dataset), results