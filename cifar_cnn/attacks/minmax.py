"""Min-Max Attack - PLACEHOLDER for Week 5."""

from .base import AttackClient

class MinMaxClient(AttackClient):
    """Min-Max attack - Sẽ implement Tuần 5."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("⚠️  MinMaxClient: PLACEHOLDER - Implement in Week 5")
    
    def fit(self, parameters, config):
        # Placeholder
        from cifar_cnn.task import train, get_parameters, set_parameters
        set_parameters(self.net, parameters)
        results = train(self.net, self.trainloader, self.local_epochs,
                       self.device, self.learning_rate, self.use_mixed_precision)
        return get_parameters(self.net), len(self.trainloader.dataset), results
