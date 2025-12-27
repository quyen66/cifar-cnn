"""Backdoor Attack Implementation: 4-Corner Trigger."""

import random
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from .base import AttackClient
from cifar_cnn.task import get_parameters, train, set_parameters


class BackdoorDataset(Dataset):
    """Dataset wrapper injects 4-corner pixel pattern trigger and flips label."""
    
    def __init__(self, dataset, target_label=0, trigger_size=3, poison_ratio=0.05):
        self.dataset = dataset
        self.target_label = target_label
        self.trigger_size = trigger_size
        
        num_poison = int(len(dataset) * poison_ratio)
        self.poisoned_indices = set(random.sample(range(len(dataset)), num_poison))
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if idx in self.poisoned_indices:
            c, h, w = image.shape
            size = self.trigger_size
            
            # 4-Corner Trigger
            image[:, 0:size, 0:size] = 1.0          # Top-Left
            image[:, 0:size, w-size:w] = 1.0        # Top-Right
            image[:, h-size:h, 0:size] = 1.0        # Bottom-Left
            image[:, h-size:h, w-size:w] = 1.0      # Bottom-Right
            
            label = self.target_label
        
        return image, label


class BackdoorClient(AttackClient):
    """Attack 5: Backdoor Attack (4-Corner Variant).
    Chèn trigger pattern vào 4 góc ảnh train và đổi nhãn về target_label.
    """
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 target_label=0, pixel_count=4, poison_ratio=0.1, trigger_size=3,
                 scaling_factor=1.0, mode="backdoor"):
        
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        
        self.scaling_factor = scaling_factor
        
        # Create backdoor dataset
        self.backdoor_dataset = BackdoorDataset(
            trainloader.dataset, target_label, trigger_size, poison_ratio
        )
        self.backdoor_loader = DataLoader(
            self.backdoor_dataset,
            batch_size=trainloader.batch_size,
            shuffle=True
        )
        
        print(f"   🎯 Backdoor Attack: γ = {scaling_factor}")

    def fit(self, parameters, config):
        from cifar_cnn.task import get_parameters, set_parameters, train
        
        set_parameters(self.net, parameters)
        
        # Train on backdoor data
        results = train(self.net, self.backdoor_loader, 
                       epochs=self.local_epochs, device=self.device,
                       learning_rate=self.learning_rate)
        
        backdoor_params = get_parameters(self.net)
        
        # Model Replacement: w_mal = w_global + γ × (w_backdoor - w_global)
        if self.scaling_factor != 1.0:
            scaled_params = []
            for w_g, w_b in zip(parameters, backdoor_params):
                w_b = np.array(w_b, dtype=np.float32) if not isinstance(w_b, np.ndarray) else w_b
                w_g = np.array(w_g, dtype=np.float32) if not isinstance(w_g, np.ndarray) else w_g
                
                update = w_b - w_g
                w_mal = w_g + self.scaling_factor * update
                scaled_params.append(w_mal)
            
            results["is_malicious"] = 1
            return scaled_params, len(self.trainloader.dataset), results
        else:
            results["is_malicious"] = 1
            return backdoor_params, len(self.trainloader.dataset), results
