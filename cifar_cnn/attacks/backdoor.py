"""Backdoor Attack Implementation: 4-Corner Trigger."""

import random
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from .base import AttackClient
from cifar_cnn.task import get_parameters, train, set_parameters


class BackdoorDataset(Dataset):
    """Dataset wrapper injects 4-corner pixel pattern trigger and flips label."""
    
    def __init__(self, dataset, target_label=0, trigger_size=3, poison_ratio=0.1):
        self.dataset = dataset
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.poison_ratio = poison_ratio
        
        # Pre-determine poisoned samples (chỉ poison_ratio %)
        num_poison = int(len(dataset) * poison_ratio)
        self.poisoned_indices = set(random.sample(range(len(dataset)), num_poison))
        
        print(f"   🎯 BackdoorDataset: {num_poison}/{len(dataset)} samples poisoned ({poison_ratio*100:.1f}%)")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if idx in self.poisoned_indices:  # Chỉ poison_ratio % samples!
            # Image shape: [C, H, W] -> CIFAR: [3, 32, 32]
            c, h, w = image.shape
            size = self.trigger_size
            
            # === INJECT TRIGGER: 4 GÓC (4 CORNERS) ===
            # Gán giá trị Max (1.0) cho các vùng này
            
            # 1. Top-Left
            image[:, 0:size, 0:size] = 1.0
            
            # 2. Top-Right
            image[:, 0:size, w-size:w] = 1.0
            
            # 3. Bottom-Left
            image[:, h-size:h, 0:size] = 1.0
            
            # 4. Bottom-Right
            image[:, h-size:h, w-size:w] = 1.0
            
            # === TARGET LABEL SWAPPING ===
            label = self.target_label
        
        return image, label


class BackdoorClient(AttackClient):
    """Attack 5: Backdoor Attack (4-Corner Variant).
    Chèn trigger pattern vào 4 góc ảnh train và đổi nhãn về target_label.
    """
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 target_label=0, pixel_count=4, poison_ratio=0.1, mode="backdoor"):
        
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        
        # Tạo Backdoor Dataset với Trigger mới (4 góc, size 3x3)
        original_dataset = trainloader.dataset
        self.backdoor_dataset = BackdoorDataset(
            original_dataset,
            target_label=target_label,
            trigger_size=3,  # Hardcode size 3 hoặc lấy từ config nếu muốn
            poison_ratio=poison_ratio 
        )
        
        # Dataloader mới chứa backdoor data
        self.backdoor_loader = DataLoader(
            self.backdoor_dataset,
            batch_size=trainloader.batch_size,
            shuffle=True,
            num_workers=trainloader.num_workers if hasattr(trainloader, 'num_workers') else 0,
            pin_memory=trainloader.pin_memory if hasattr(trainloader, 'pin_memory') else False
        )
        
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        
        # Store global params for FedProx
        global_params = [p.clone().detach() for p in self.net.parameters()]
        
        # Train trên dữ liệu đã bị dính Backdoor 4 góc
        results = train(
            self.net,
            self.backdoor_loader, 
            epochs=self.local_epochs,
            device=self.device,
            learning_rate=self.learning_rate,
            use_mixed_precision=self.use_mixed_precision,
            proximal_mu=self.proximal_mu,
            global_params=global_params
        )
        results["is_malicious"] = 1
        return get_parameters(self.net), len(self.trainloader.dataset), results