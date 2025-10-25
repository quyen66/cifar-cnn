"""ALIE Attack - PLACEHOLDER for Week 5."""

from .base import AttackClient
from cifar_cnn.task import train, get_parameters, set_parameters

class ALIEClient(AttackClient):
    """A Little Is Enough attack - Sẽ implement Tuần 5."""
    
    def __init__(self, *args, z_perturbation=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.z = z_perturbation
        print("⚠️  ALIEClient: PLACEHOLDER - Implement in Week 5")
    
    def fit(self, parameters, config):
        # Placeholder - act like normal client
        set_parameters(self.net, parameters)
        results = train(self.net, self.trainloader, self.local_epochs,
                       self.device, self.learning_rate, self.use_mixed_precision)
        return get_parameters(self.net), len(self.trainloader.dataset), results
