"""Model definition and training/testing functions - WITH SMART FEDPROX."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class CNN(nn.Module):
    """CNN model cho CIFAR-10."""
    
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model():
    return CNN(num_classes=10)


def train(net, trainloader, epochs, device, learning_rate=0.001, 
          use_mixed_precision=True, proximal_mu=0.0, global_params=None):
    """
    Train with SMART FedProx.
    Auto-detects parameter format (28 trainable vs 46 full state) to calculate loss correctly.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device.type == 'cuda' else None
    
    net.train()
    net.to(device)
    
    # --- [SMART FEDPROX PREPARATION] ---
    fedprox_tensors = []
    mode = "disabled" # disabled, trainable_only, full_state
    
    if global_params is not None and proximal_mu > 0:
        # Convert global_params to tensors on device
        with torch.no_grad():
            for p in global_params:
                if not isinstance(p, torch.Tensor):
                    p = torch.tensor(p)
                fedprox_tensors.append(p.to(device).detach())
        
        # Check matching strategy
        n_global = len(fedprox_tensors)
        n_trainable = len(list(net.parameters()))       # Should be 28
        n_state = len(list(net.state_dict().keys()))    # Should be 46
        
        if n_global == n_trainable:
            mode = "trainable_only"
            # print(f"   🛡️ FedProx Active: Matching {n_global} trainable parameters.")
        elif n_global == n_state:
            mode = "full_state"
            # Map full state by name to find trainable ones
            global_weight_map = {name: val for name, val in zip(net.state_dict().keys(), fedprox_tensors)}
            # print(f"   🛡️ FedProx Active: Matching {n_global} state parameters via naming.")
        else:
            print(f"   ⚠️ FedProx Warning: Mismatch! Global={n_global}, Trainable={n_trainable}, State={n_state}. Disabled.")
    # -----------------------------------
    
    total_loss = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = net(images)
                    ce_loss = criterion(outputs, labels)
                    
                    # --- FedProx Calculation ---
                    proximal_loss = 0.0
                    if mode == "trainable_only":
                        # Case 28 vs 28: Zip trực tiếp 1-1
                        for local_p, global_p in zip(net.parameters(), fedprox_tensors):
                            proximal_loss += torch.sum((local_p - global_p) ** 2)
                            
                    elif mode == "full_state":
                        # Case 46 vs 46: Map theo tên
                        for name, local_p in net.named_parameters():
                            if name in global_weight_map:
                                global_p = global_weight_map[name]
                                proximal_loss += torch.sum((local_p - global_p) ** 2)
                                
                    if mode != "disabled":
                        proximal_loss = (proximal_mu / 2) * proximal_loss
                        total_batch_loss = ce_loss + proximal_loss
                    else:
                        total_batch_loss = ce_loss
                    # ---------------------------
                
                scaler.scale(total_batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(images)
                ce_loss = criterion(outputs, labels)
                
                # --- FedProx Calculation (Standard) ---
                proximal_loss = 0.0
                if mode == "trainable_only":
                    for local_p, global_p in zip(net.parameters(), fedprox_tensors):
                        proximal_loss += torch.sum((local_p - global_p) ** 2)
                        
                elif mode == "full_state":
                    for name, local_p in net.named_parameters():
                        if name in global_weight_map:
                            global_p = global_weight_map[name]
                            proximal_loss += torch.sum((local_p - global_p) ** 2)
                            
                if mode != "disabled":
                    proximal_loss = (proximal_mu / 2) * proximal_loss
                    total_batch_loss = ce_loss + proximal_loss
                else:
                    total_batch_loss = ce_loss
                # --------------------------------------
                
                total_batch_loss.backward()
                optimizer.step()
            
            epoch_loss += total_batch_loss.item()
        
        total_loss += epoch_loss / len(trainloader)
    
    return {"train_loss": total_loss / epochs}


def test(net, testloader, device):
    """Test model accuracy."""
    criterion = nn.CrossEntropyLoss()
    net.eval()
    net.to(device)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(testloader), correct / total


def get_parameters(net):
    """Get model parameters as numpy arrays (FULL STATE)."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    """Set model parameters from numpy arrays."""
    # Logic: Nếu nhận 28 params thì chỉ update Weights. Nếu 46 thì update Full State.
    keys = list(net.state_dict().keys())
    
    if len(parameters) == len(keys):
        # Full update
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
    else:
        # Partial update (Chỉ Weights) - Fallback
        # print(f"⚠️ Warning: Set params partial update ({len(parameters)}/{len(keys)})")
        state_dict = net.state_dict()
        # Giả định parameters chỉ chứa weights theo thứ tự trainable
        # Cần cẩn thận, nhưng thường Flower truyền weights theo thứ tự net.parameters()
        trainable_keys = list(name for name, _ in net.named_parameters())
        
        if len(parameters) == len(trainable_keys):
             for key, param in zip(trainable_keys, parameters):
                 state_dict[key] = torch.tensor(param)
             net.load_state_dict(state_dict, strict=False)
        else:
             # Cố gắng zip mù quáng nếu không khớp gì cả
             params_dict = zip(keys, parameters)
             state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
             net.load_state_dict(state_dict, strict=False)