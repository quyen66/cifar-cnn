"""ALIE Attack - WITH FEDPROX SUPPORT (PLACEHOLDER for Week 5)."""

from .base import AttackClient
from cifar_cnn.task import get_parameters


class ALIEClient(AttackClient):
    """A Little Is Enough attack - Sẽ implement Tuần 5, có FedProx."""
    
    def __init__(self, *args, z_perturbation=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.z = z_perturbation
        print("⚠️  ALIEClient: PLACEHOLDER - Implement in Week 5")
    
    def fit(self, parameters, config):
        """Placeholder - train normally with FedProx."""
        
        # For now, just train normally with FedProx
        results = self.train_with_fedprox(parameters)
        
        return get_parameters(self.net), len(self.trainloader.dataset), results