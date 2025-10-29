"""Model definition and training/testing functions - WITH FEDPROX."""

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
    Train with optional FedProx proximal term.
    
    Args:
        net: Model to train
        trainloader: Training data loader
        epochs: Number of local epochs
        device: Device to use (cuda/cpu)
        learning_rate: Learning rate
        use_mixed_precision: Use mixed precision training
        proximal_mu: FedProx proximal coefficient (0.01 recommended)
        global_params: Global model parameters for FedProx
                       List of tensors in same format as net.parameters()
    
    Returns:
        dict with train_loss
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device.type == 'cuda' else None
    
    net.train()
    net.to(device)
    
    # Convert global_params to device if provided
    if global_params is not None and proximal_mu > 0:
        global_params = [p.to(device) if isinstance(p, torch.Tensor) 
                        else torch.tensor(p, device=device) 
                        for p in global_params]
    
    total_loss = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                # Mixed precision training
                with torch.amp.autocast('cuda'):
                    outputs = net(images)
                    ce_loss = criterion(outputs, labels)
                    
                    # FedProx proximal term
                    if global_params is not None and proximal_mu > 0:
                        proximal_loss = 0.0
                        for local_param, global_param in zip(net.parameters(), global_params):
                            proximal_loss += ((local_param - global_param) ** 2).sum()
                        proximal_loss = (proximal_mu / 2) * proximal_loss
                    else:
                        proximal_loss = 0.0
                    
                    total_batch_loss = ce_loss + proximal_loss
                
                scaler.scale(total_batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                outputs = net(images)
                ce_loss = criterion(outputs, labels)
                
                # FedProx proximal term
                if global_params is not None and proximal_mu > 0:
                    proximal_loss = 0.0
                    for local_param, global_param in zip(net.parameters(), global_params):
                        proximal_loss += ((local_param - global_param) ** 2).sum()
                    proximal_loss = (proximal_mu / 2) * proximal_loss
                else:
                    proximal_loss = 0.0
                
                total_batch_loss = ce_loss + proximal_loss
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
    """Get model parameters as numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    """Set model parameters from numpy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)