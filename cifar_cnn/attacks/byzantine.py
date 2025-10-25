"""Byzantine Attack."""

import numpy as np
from .base import AttackClient
from cifar_cnn.task import train, get_parameters, set_parameters


class ByzantineClient(AttackClient):
    """Byzantine Attack - Random hoặc sign-flipping."""
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True,
                 byzantine_type="sign_flip", byzantine_scale=10.0):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision)
        self.byzantine_type = byzantine_type
        self.byzantine_scale = byzantine_scale
    
    def fit(self, parameters, config):
        """Train normally, then corrupt gradients."""
        set_parameters(self.net, parameters)
        results = train(
            self.net, self.trainloader, epochs=self.local_epochs,
            device=self.device, learning_rate=self.learning_rate,
            use_mixed_precision=self.use_mixed_precision
        )
        
        benign_params = get_parameters(self.net)
        
        if self.byzantine_type == "sign_flip":
            malicious_params = [-self.byzantine_scale * p for p in benign_params]
        elif self.byzantine_type == "random":
            malicious_params = [
                np.random.randn(*p.shape).astype(np.float32) * self.byzantine_scale 
                for p in benign_params
            ]
        elif self.byzantine_type == "scaled":
            malicious_params = [self.byzantine_scale * p for p in benign_params]
        else:
            malicious_params = benign_params
        
        return malicious_params, len(self.trainloader.dataset), results