"""Label Flipping Attack - WITH FEDPROX SUPPORT."""

import torch
from torch.utils.data import DataLoader
from .base import AttackClient
from cifar_cnn.task import get_parameters


class LabelFlippingDataset(torch.utils.data.Dataset):
    """Dataset wrapper that flips labels."""
    
    def __init__(self, original_dataset, flip_probability=1.0, flip_type="reverse", num_classes=10, mode="label_flip"):
        self.dataset = original_dataset
        self.flip_probability = flip_probability
        self.flip_type = flip_type
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Flip label with probability
        if torch.rand(1).item() < self.flip_probability:
            if self.flip_type == "reverse":
                # Reverse label: 0->9, 1->8, etc.
                label = self.num_classes - 1 - label
            elif self.flip_type == "random":
                # Random wrong label
                label = torch.randint(0, self.num_classes, (1,)).item()
        
        return image, label


class LabelFlippingClient(AttackClient):
    """Label Flipping Attack - Train với flipped labels, dùng FedProx."""
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 flip_probability=1.0, flip_type="reverse"):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        self.flip_probability = flip_probability
        self.flip_type = flip_type
        
        # Create flipped dataset
        original_dataset = trainloader.dataset
        flipped_dataset = LabelFlippingDataset(
            original_dataset, 
            flip_probability=flip_probability,
            flip_type=flip_type
        )
        
        # Create new dataloader with flipped labels
        self.flipped_trainloader = DataLoader(
            flipped_dataset,
            batch_size=trainloader.batch_size,
            shuffle=True,
            num_workers=trainloader.num_workers if hasattr(trainloader, 'num_workers') else 0,
            pin_memory=trainloader.pin_memory if hasattr(trainloader, 'pin_memory') else False
        )
    
    def fit(self, parameters, config):
        """Train with flipped labels using FedProx."""
        from cifar_cnn.task import train, set_parameters, get_parameters
        
        set_parameters(self.net, parameters)
        
        # Store global parameters for FedProx
        global_params = [p.clone().detach() for p in self.net.parameters()]

        
        # Train on flipped data with FedProx
        results = train(
            self.net,
            self.flipped_trainloader,  # Use flipped data
            epochs=self.local_epochs,
            device=self.device,
            learning_rate=self.learning_rate,
            use_mixed_precision=self.use_mixed_precision,
            proximal_mu=self.proximal_mu,
            global_params=global_params
        )
        results["is_malicious"] = 1
        return get_parameters(self.net), len(self.trainloader.dataset), results