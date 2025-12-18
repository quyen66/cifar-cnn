"""Flower client implementation - Updated for 7 Attacks."""

import os
import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from cifar_cnn.task import get_model, train, test, get_parameters, set_parameters
from cifar_cnn.dataset import prepare_datasets

# Import các loại attack clients
from cifar_cnn.attacks import (
    LabelFlippingClient,
    ByzantineClient, 
    GaussianNoiseClient,
    RandomNoiseClient,
    BackdoorClient,
    ALIEClient,
    MinMaxClient
)

os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

# ============================================
# FLOWER CLIENT (BENIGN) DEFINITION
# ============================================
class FlowerClient(NumPyClient):
    """Normal benign client with FedProx support."""
    
    def __init__(self, net, trainloader, testloader, device, local_epochs, 
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.use_mixed_precision = use_mixed_precision
        self.proximal_mu = proximal_mu
        
        # Move model to GPU if available
        self.net = self.net.to(self.device)
    
    def fit(self, parameters, config):
        """Train with FedProx proximal term."""
        set_parameters(self.net, parameters)
        self.net = self.net.to(self.device)
        
        # Store global parameters for FedProx (convert to tensors)
        global_params = [p.clone().detach() for p in self.net.parameters()]
        
        # Train with FedProx
        results = train(
            self.net, 
            self.trainloader, 
            epochs=self.local_epochs,
            device=self.device, 
            learning_rate=self.learning_rate,
            use_mixed_precision=self.use_mixed_precision,
            proximal_mu=self.proximal_mu,
            global_params=global_params
        )
        
        return get_parameters(self.net), len(self.trainloader.dataset), results
    
    def evaluate(self, parameters, config):
        """Standard evaluation."""
        set_parameters(self.net, parameters)
        self.net = self.net.to(self.device)
        
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


# ============================================
# CLIENT FACTORY FUNCTION
# ============================================
def client_fn(context: Context) -> Client:
    """Create client based on config."""
    
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # 2. Get Config
    client_id = int(context.node_config["partition-id"])
    run_config = context.run_config
    
    num_clients = run_config["num-clients"]
    batch_size = run_config["batch-size"]
    local_epochs = run_config["local-epochs"]
    learning_rate = run_config["learning-rate"]
    use_mixed_precision = run_config.get("use-mixed-precision", True)
    proximal_mu = run_config.get("proximal-mu", 0.01)
    
    # Attack Config
    attack_type = run_config.get("attack-type", "none")
    attack_ratio = run_config.get("attack-ratio", 0.0)
    
    # 3. Prepare Data
    # Lưu ý: num_workers=0 để tránh lỗi đa luồng trong simulation nếu máy yếu
    trainloader, testloader = prepare_datasets(
        client_id, num_clients, batch_size, 
        run_config.get("partition-type", "iid"), 
        alpha=run_config.get("alpha", 0.5),
        num_workers=0, pin_memory=True
    )
    
    # 4. Initialize Model
    net = get_model().to(device)
    
    # 5. Determine Role (Attacker vs Benign)
    num_attackers = int(num_clients * attack_ratio)
    is_attacker = client_id < num_attackers
    
    # Nếu là client lành tính hoặc không có tấn công
    if not is_attacker or attack_type == "none":
        return FlowerClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu
        ).to_client()
    
    # 6. Instantiate Attack Clients
    if attack_type == "random_noise":
        scale = run_config.get("random-noise-scale", 0.5)
        return RandomNoiseClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            scale=scale
        ).to_client()
        
    elif attack_type == "gaussian_noise":
        std = run_config.get("gaussian-noise-std", 0.2)
        return GaussianNoiseClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            std=std
        ).to_client()

    elif attack_type == "sign_flip":
        return ByzantineClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu
        ).to_client()

    elif attack_type == "label_flip":
        flip_prob = run_config.get("flip-probability", 1.0)
        flip_type = run_config.get("flip-type", "reverse")
        return LabelFlippingClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            flip_probability=flip_prob, flip_type=flip_type
        ).to_client()
        
    elif attack_type == "backdoor":
        target_label = run_config.get("backdoor-label", 0)
        pixel_count = run_config.get("backdoor-pixel-count", 4)
        return BackdoorClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            target_label=target_label, pixel_count=pixel_count
        ).to_client()
        
    elif attack_type == "alie":
        z = run_config.get("alie-z", 1.5)
        return ALIEClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            z=z
        ).to_client()
        
    elif attack_type == "minmax":
        gamma = run_config.get("minmax-gamma", 10.0)
        return MinMaxClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            gamma=gamma
        ).to_client()
        
    else:
        # Fallback to benign
        return FlowerClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu
        ).to_client()

# Flower ClientApp
app = ClientApp(client_fn=client_fn)