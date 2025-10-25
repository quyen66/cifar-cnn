"""Label Flipping Attack."""

import torch
from cifar_cnn.task import get_parameters, set_parameters, train
from .base import AttackClient


class LabelFlippingClient(AttackClient):
    """Label Flipping Attack - Đảo ngược labels."""
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True,
                 flip_probability=1.0, flip_type="reverse"):
        super().__init__(net, trainloader, testloader, device, 
                        local_epochs, learning_rate, use_mixed_precision)
        self.flip_probability = flip_probability
        self.flip_type = flip_type
        self._flip_labels()
    
    def _flip_labels(self):
        """Flip labels trong dataset."""
        # Access dataset through trainloader
        dataset = self.trainloader.dataset
        
        for idx in range(len(dataset)):
            if torch.rand(1).item() < self.flip_probability:
                image, label = dataset[idx]
                
                if self.flip_type == "reverse":
                    new_label = 9 - label
                else:  # random
                    new_label = torch.randint(0, 10, (1,)).item()
                
                # Update label (this works for Subset)
                if hasattr(dataset, 'dataset'):
                    # It's a Subset
                    original_idx = dataset.indices[idx]
                    dataset.dataset.targets[original_idx] = new_label
                else:
                    # Direct dataset
                    dataset.targets[idx] = new_label
    
    def fit(self, parameters, config):
        """Train với flipped labels."""
        set_parameters(self.net, parameters)
        results = train(
            self.net, self.trainloader, epochs=self.local_epochs,
            device=self.device, learning_rate=self.learning_rate,
            use_mixed_precision=self.use_mixed_precision
        )
        return get_parameters(self.net), len(self.trainloader.dataset), results
