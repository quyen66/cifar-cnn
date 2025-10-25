"""Flower client implementation - GPU OPTIMIZED."""

import os
import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from cifar_cnn.task import get_model, train, test, get_parameters, set_parameters
from cifar_cnn.dataset import prepare_datasets
from cifar_cnn.attacks import (
    LabelFlippingClient,
    ByzantineClient, 
    GaussianNoiseClient,
    ALIEClient,
    MinMaxClient
)

os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

# ============================================
# FORCE GPU USAGE
# ============================================
FORCE_GPU = True
if FORCE_GPU and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Faster training
    torch.backends.cudnn.enabled = True
    print(f"🚀 GPU ENABLED: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  Running on CPU (slower)")


class FlowerClient(NumPyClient):
    """Normal benign client - GPU OPTIMIZED."""
    
    def __init__(self, net, trainloader, testloader, device, local_epochs, 
                 learning_rate=0.001, use_mixed_precision=True):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.use_mixed_precision = use_mixed_precision
        
        # FORCE move model to GPU
        self.net = self.net.to(self.device)
    
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        # Ensure model on GPU
        self.net = self.net.to(self.device)
        
        results = train(
            self.net, self.trainloader, epochs=self.local_epochs,
            device=self.device, learning_rate=self.learning_rate,
            use_mixed_precision=self.use_mixed_precision
        )
        return get_parameters(self.net), len(self.trainloader.dataset), results
    
    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        self.net = self.net.to(self.device)
        
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context) -> Client:
    """Tạo client với GPU support."""
    
    # ============================================
    # FORCE GPU DEVICE
    # ============================================
    if FORCE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Client using GPU")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Client using CPU")
    
    # Client ID
    client_id = int(context.node_config["partition-id"])
    
    # Config
    num_clients = context.run_config["num-clients"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    use_mixed_precision = context.run_config.get("use-mixed-precision", True)
    partition_type = context.run_config.get("partition-type", "iid")
    
    # GPU optimization
    num_workers = context.run_config.get("num-workers", 0)  # 0 for GPU (faster)
    pin_memory = context.run_config.get("pin-memory", True) if device.type == "cuda" else False
    
    # Attack config
    attack_type = context.run_config.get("attack-type", "none")
    attack_ratio = context.run_config.get("attack-ratio", 0.0)
    
    # Load data with GPU optimization
    trainloader, testloader = prepare_datasets(
        client_id, num_clients, batch_size, partition_type,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    # Model - FORCE to GPU immediately
    net = get_model().to(device)
    
    # Determine if attacker
    num_attackers = int(num_clients * attack_ratio)
    is_attacker = client_id < num_attackers
    
    # Create appropriate client
    if not is_attacker or attack_type == "none":
        return FlowerClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision
        ).to_client()
    
    # Attack clients (all GPU-enabled)
    if attack_type == "label_flip":
        flip_probability = context.run_config.get("flip-probability", 1.0)
        flip_type = context.run_config.get("flip-type", "reverse")
        
        return LabelFlippingClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision,
            flip_probability, flip_type
        ).to_client()
    
    elif attack_type == "byzantine":
        byzantine_type = context.run_config.get("byzantine-type", "sign_flip")
        byzantine_scale = context.run_config.get("byzantine-scale", 10.0)
        
        return ByzantineClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision,
            byzantine_type, byzantine_scale
        ).to_client()
    
    elif attack_type == "gaussian":
        noise_std = context.run_config.get("noise-std", 0.1)
        
        return GaussianNoiseClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision,
            noise_std
        ).to_client()
    
    elif attack_type == "alie":
        z_perturbation = context.run_config.get("z-perturbation", 1.5)
        
        return ALIEClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision,
            z_perturbation
        ).to_client()
    
    else:
        return FlowerClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision
        ).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)