"""Gaussian Noise Attack - WITH FEDPROX SUPPORT."""

import numpy as np
from .base import AttackClient
from cifar_cnn.task import get_parameters


class GaussianNoiseClient(AttackClient):
    """Gaussian Noise Attack - Train with FedProx, then add noise to gradients."""
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 noise_std=0.1):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        self.noise_std = noise_std
    
    def fit(self, parameters, config):
        """Train with FedProx, then add Gaussian noise."""
        
        # Train normally with FedProx
        results = self.train_with_fedprox(parameters)
        
        # Get benign parameters
        benign_params = get_parameters(self.net)
        
        # Add Gaussian noise
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