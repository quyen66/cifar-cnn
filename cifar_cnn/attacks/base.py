"""Base class cho attack clients - WITH FEDPROX SUPPORT."""

import torch
from flwr.client import NumPyClient
from cifar_cnn.task import train, test, get_parameters, set_parameters


class AttackClient(NumPyClient):
    """Base class cho tất cả attack clients với FedProx support."""
    
    def __init__(self, net, trainloader, testloader, device, 
                 local_epochs, learning_rate=0.001, 
                 use_mixed_precision=True, proximal_mu=0.01):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.use_mixed_precision = use_mixed_precision
        self.proximal_mu = proximal_mu
    
    def fit(self, parameters, config):
        """Override this in subclasses."""
        raise NotImplementedError("Subclass must implement fit()")
    
    def evaluate(self, parameters, config):
        """Standard evaluation - same for all."""
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}
    
    def train_with_fedprox(self, parameters):
        """Helper method to train with FedProx - for use in attack clients."""
        set_parameters(self.net, parameters)
        
        # Store global parameters for FedProx
        global_params = [torch.tensor(p, device=self.device) for p in parameters]
        
        # Train with FedProx
        results = train(
            self.net, 
            self.trainloader, 
            epochs=self.local_epochs,
            device=self.device, 
            learning_rate=self.learning_rate,
            use_mixed_precision=self.use_mixed_precision,
            proximal_mu=self.proximal_mu,
            global_params=global_params
        )
        
        return results