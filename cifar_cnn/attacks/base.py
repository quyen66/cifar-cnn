"""Base class cho attack clients."""

from flwr.client import NumPyClient
from cifar_cnn.task import train, test, get_parameters, set_parameters


class AttackClient(NumPyClient):
    """Base class cho tất cả attack clients."""
    
    def __init__(self, net, trainloader, testloader, device, 
                 local_epochs, learning_rate=0.001, 
                 use_mixed_precision=True):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.use_mixed_precision = use_mixed_precision
    
    def fit(self, parameters, config):
        """Override this in subclasses."""
        raise NotImplementedError("Subclass must implement fit()")
    
    def evaluate(self, parameters, config):
        """Standard evaluation - same for all."""
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}