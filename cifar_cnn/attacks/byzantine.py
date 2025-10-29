"""Byzantine Attack - WITH FEDPROX SUPPORT."""

import numpy as np
from .base import AttackClient
from cifar_cnn.task import get_parameters


class ByzantineClient(AttackClient):
    """Byzantine Attack - Train normally with FedProx, then corrupt gradients."""
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 byzantine_type="sign_flip", byzantine_scale=10.0):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        self.byzantine_type = byzantine_type
        self.byzantine_scale = byzantine_scale
    
    def fit(self, parameters, config):
        """Train normally with FedProx, then corrupt gradients."""
        
        # Train normally with FedProx
        results = self.train_with_fedprox(parameters)
        
        # Get benign parameters after training
        benign_params = get_parameters(self.net)
        
        # Corrupt gradients
        if self.byzantine_type == "sign_flip":
            # Flip sign and scale
            malicious_params = [-self.byzantine_scale * p for p in benign_params]
        
        elif self.byzantine_type == "random":
            # Random gradients
            malicious_params = [
                np.random.randn(*p.shape).astype(np.float32) * self.byzantine_scale 
                for p in benign_params
            ]
        
        elif self.byzantine_type == "scaled":
            # Just scale up
            malicious_params = [self.byzantine_scale * p for p in benign_params]
        
        else:
            # No attack
            malicious_params = benign_params
        
        return malicious_params, len(self.trainloader.dataset), results