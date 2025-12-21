"""
Flower client implementation - Updated for 7 Attacks.
Includes 'is_malicious' flag in fit metrics for accurate server-side evaluation.
"""

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
        
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        
        # Load global params for FedProx
        server_round = config.get("server_round", 1)
        global_params = None
        if self.proximal_mu > 0:
            global_params = [torch.tensor(p, device=self.device) for p in parameters]
            
        train_loss = train(
            self.net, 
            self.trainloader, 
            self.local_epochs, 
            self.device, 
            self.learning_rate, 
            self.use_mixed_precision, 
            self.proximal_mu,
            global_params
        )
        
        # --- LOGIC MỚI: Gửi cờ 'is_malicious' về Server ---
        # Logic: Nếu class hiện tại KHÔNG PHẢI FlowerClient gốc (tức là class con Attack),
        # thì đánh dấu là Malicious.
        is_malicious_flag = 0
        if self.__class__.__name__ != "FlowerClient":
             is_malicious_flag = 1
             
        metrics = {
            "train_loss": train_loss["train_loss"],
            "is_malicious": is_malicious_flag 
        }
        
        return get_parameters(self.net), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

# ============================================
# CLIENT FACTORY FUNCTION
# ============================================
def client_fn(context: Context) -> Client:
    """Create a Flower client instance based on configuration."""
    
    # 1. Load configuration and data
    net = get_model()
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Load data for this client
    trainloader, testloader = prepare_datasets(
        partition_id, num_partitions, 
        batch_size=int(context.run_config["batch-size"]),
        alpha=float(context.run_config.get("alpha", 0.5))
    )
    
    # Hyperparameters
    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])
    proximal_mu = float(context.run_config.get("proximal-mu", 0.01))
    
    # Casting use-mixed-precision explicitly
    use_mixed_precision = context.run_config.get("use-mixed-precision", True)
    if isinstance(use_mixed_precision, str):
        use_mixed_precision = use_mixed_precision.lower() == 'true'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Determine if this client is Malicious
    attack_type = context.run_config.get("attack-type", "none")
    attack_ratio = float(context.run_config.get("attack-ratio", 0.0))
    num_malicious = int(num_partitions * attack_ratio)
    
    # Logic: First N clients are malicious
    is_attacker = (partition_id < num_malicious) and (attack_type != "none")
    
    # Reset attack type if client is benign
    if not is_attacker:
        attack_type = "none"
        
    # 3. Instantiate appropriate Client Class
    if attack_type == "none":
        return FlowerClient(
            net, trainloader, testloader, device, local_epochs, 
            learning_rate, use_mixed_precision, proximal_mu
        ).to_client()
    
    # --- Attack Clients ---
    # Note: These clients inherit from FlowerClient or implement fit(),
    # so the is_malicious flag logic will work.
    
    elif attack_type == "label_flip":
        flip_prob = float(context.run_config.get("flip-probability", 1.0))
        flip_type = context.run_config.get("flip-type", "reverse")
        return LabelFlippingClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            flip_probability=flip_prob, flip_type=flip_type
        ).to_client()
        
    elif attack_type == "gaussian_noise":
        noise_std = float(context.run_config.get("gaussian-noise-std", 0.1))
        return GaussianNoiseClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            noise_std=noise_std
        ).to_client()
        
    elif attack_type == "random_noise":
        scale = float(context.run_config.get("random-noise-scale", 0.5))
        return RandomNoiseClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            scale=scale
        ).to_client()
        
    elif attack_type == "backdoor":
        target_label = int(context.run_config.get("backdoor-label", 0))
        pixel_count = int(context.run_config.get("backdoor-pixel-count", 4))
        return BackdoorClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            target_label=target_label, pixel_count=pixel_count
        ).to_client()
        
    elif attack_type == "alie":
        z = float(context.run_config.get("alie-z", 1.5))
        return ALIEClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            z=z
        ).to_client()
        
    elif attack_type == "minmax":
        gamma = float(context.run_config.get("minmax-gamma", 10.0))
        return MinMaxClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            gamma=gamma
        ).to_client()
    
    elif attack_type == "sign_flip":
        # ByzantineClient usually handles simple gradient manipulation like sign flip
        return ByzantineClient(
            net, trainloader, testloader, device, local_epochs,
            learning_rate, use_mixed_precision, proximal_mu,
            mode="sign_flip"
        ).to_client()
        
    else:
        # Fallback to Benign if unknown attack type
        print(f"⚠️ Unknown attack type '{attack_type}', defaulting to benign.")
        return FlowerClient(
            net, trainloader, testloader, device, local_epochs, 
            learning_rate, use_mixed_precision, proximal_mu
        ).to_client()

app = ClientApp(client_fn=client_fn)