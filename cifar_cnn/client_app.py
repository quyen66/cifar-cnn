"""
Flower client implementation - Updated for 7 Attacks.
FIX V3: Gửi partition_id trong metrics để server có thể identify clients.

QUAN TRỌNG: File này PHẢI được sử dụng cùng với server_app.py đã fix!
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
    MinMaxClient,
    MinSumClient,
    OnOffAttackClient,
    SlowPoisoningClient,
)

os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

# ============================================
# FLOWER CLIENT (BENIGN) DEFINITION
# ============================================
class FlowerClient(NumPyClient):
    """Normal benign client with FedProx support."""
    
    def __init__(self, net, trainloader, testloader, device, local_epochs, 
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 partition_id=None):  # ===== FIX: Thêm partition_id =====
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.use_mixed_precision = use_mixed_precision
        self.proximal_mu = proximal_mu
        self.partition_id = partition_id  # ===== FIX =====
        
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
        
        # --- FIX: Gửi partition_id trong metrics ---
        is_malicious_flag = 0
        if self.__class__.__name__ != "FlowerClient":
             is_malicious_flag = 1
             
        metrics = {
            "train_loss": train_loss["train_loss"],
            "is_malicious": is_malicious_flag,
            "partition_id": self.partition_id  # ===== FIX =====
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
            learning_rate, use_mixed_precision, proximal_mu,
            partition_id=partition_id  # ===== FIX =====
        ).to_client()
    
    # --- Attack Clients ---
    # Tạo wrapper function để inject partition_id vào metrics
    
    def create_attack_client_with_partition_id(attack_client_class, **kwargs):
        """Helper to create attack client with partition_id in metrics."""
        client = attack_client_class(**kwargs)
        original_fit = client.fit
        
        def wrapped_fit(parameters, config):
            params, num_examples, metrics = original_fit(parameters, config)
            metrics["partition_id"] = partition_id  # Inject partition_id
            return params, num_examples, metrics
        
        client.fit = wrapped_fit
        return client
    
    if attack_type == "label_flip":
        flip_prob = float(context.run_config.get("flip-probability", 1.0))
        flip_type = context.run_config.get("flip-type", "reverse")
        client = create_attack_client_with_partition_id(
            LabelFlippingClient,
            net=net, trainloader=trainloader, testloader=testloader, 
            device=device, local_epochs=local_epochs,
            learning_rate=learning_rate, use_mixed_precision=use_mixed_precision, 
            proximal_mu=proximal_mu,
            flip_probability=flip_prob, flip_type=flip_type
        )
        return client.to_client()
        
    elif attack_type == "gaussian_noise":
        noise_std = float(context.run_config.get("gaussian-noise-std", 0.1))
        client = create_attack_client_with_partition_id(
            GaussianNoiseClient,
            net=net, trainloader=trainloader, testloader=testloader, 
            device=device, local_epochs=local_epochs,
            learning_rate=learning_rate, use_mixed_precision=use_mixed_precision, 
            proximal_mu=proximal_mu,
            noise_std=noise_std
        )
        return client.to_client()
        
    elif attack_type == "random_noise":
        scale = float(context.run_config.get("random-noise-scale", 0.5))
        client = create_attack_client_with_partition_id(
            RandomNoiseClient,
            net=net, trainloader=trainloader, testloader=testloader, 
            device=device, local_epochs=local_epochs,
            learning_rate=learning_rate, use_mixed_precision=use_mixed_precision, 
            proximal_mu=proximal_mu,
            scale=scale
        )
        return client.to_client()
        
    elif attack_type == "backdoor":
        #poison_ratio = float(context.run_config.get("attack-ratio", 0.1))  
        target_label = int(context.run_config.get("backdoor-label", 0))
        pixel_count = int(context.run_config.get("backdoor-pixel-count", 4))
        scaling_factor = float(context.run_config.get("backdoor-scaling-factor", 10.0))
        client = create_attack_client_with_partition_id(
            BackdoorClient,
            net=net, trainloader=trainloader, testloader=testloader, 
            device=device, local_epochs=local_epochs,
            learning_rate=learning_rate, use_mixed_precision=use_mixed_precision, 
            proximal_mu=proximal_mu,
            target_label=target_label, pixel_count=pixel_count,poison_ratio=0.05, scaling_factor=scaling_factor
        )
        return client.to_client()
        
    elif attack_type == "alie":
        z = float(context.run_config.get("alie-z", 1.5))
        n_clients = int(context.run_config.get("num-clients", 40))
        n_malicious = int(context.run_config.get("attack-ratio", 0.3)) * n_clients
        client = create_attack_client_with_partition_id(
            ALIEClient,
            net=net, trainloader=trainloader, testloader=testloader, 
            device=device, local_epochs=local_epochs,
            learning_rate=learning_rate, use_mixed_precision=use_mixed_precision, 
            proximal_mu=proximal_mu, z=z, n_clients=n_clients, n_malicious=n_malicious
        )
        return client.to_client()
        
    elif attack_type == "minmax":
        gamma = float(context.run_config.get("minmax-gamma", 10.0))
        client = create_attack_client_with_partition_id(
            MinMaxClient,
            net=net, trainloader=trainloader, testloader=testloader, 
            device=device, local_epochs=local_epochs,
            learning_rate=learning_rate, use_mixed_precision=use_mixed_precision, 
            proximal_mu=proximal_mu,
            gamma_init=gamma
        )
        return client.to_client()
    
    elif attack_type == "minsum":
        gamma = float(context.run_config.get("minmax-gamma", 10.0))
        client = create_attack_client_with_partition_id(
            MinSumClient,
            net=net, trainloader=trainloader, testloader=testloader, 
            device=device, local_epochs=local_epochs,
            learning_rate=learning_rate, use_mixed_precision=use_mixed_precision, 
            proximal_mu=proximal_mu,
            gamma_init=gamma
        )
        return client.to_client()
    
    elif attack_type == "sign_flip":
        client = create_attack_client_with_partition_id(
            ByzantineClient,
            net=net, trainloader=trainloader, testloader=testloader, 
            device=device, local_epochs=local_epochs,
            learning_rate=learning_rate, use_mixed_precision=use_mixed_precision, 
            proximal_mu=proximal_mu,
            mode="sign_flip"
        )
        return client.to_client()
    
    elif attack_type == "on_off":
        # On-Off Attack: Tấn công không liên tục
        # Highlight: Reputation EMA, Lambda Penalty, Soft Filter
        frequency = int(context.run_config.get("onoff-frequency", 5))
        onoff_type = context.run_config.get("onoff-attack", "sign_flip")
        onoff_scale = float(context.run_config.get("onoff-attack-scale", 2.0))
        
        client = create_attack_client_with_partition_id(
            OnOffAttackClient,
            net=net, trainloader=trainloader, testloader=testloader,
            device=device, local_epochs=local_epochs,
            learning_rate=learning_rate, use_mixed_precision=use_mixed_precision,
            proximal_mu=proximal_mu,
            attack_frequency=frequency,
            attack_type=onoff_type,
            attack_scale=onoff_scale
        )
        return client.to_client()
    
    elif attack_type == "slow_poison":
        # Slow Poisoning: Tấn công từ từ
        # Highlight: Baseline Tracking, Confidence Scoring, Hard Filter, Non-IID Handler
        poison_rate = float(context.run_config.get("slow-poison-rate", 0.1))
        poison_direction = context.run_config.get("slow-poison-direction", "negative")
        warmup = int(context.run_config.get("slow-poison-warmup", 0))
        
        client = create_attack_client_with_partition_id(
            SlowPoisoningClient,
            net=net, trainloader=trainloader, testloader=testloader,
            device=device, local_epochs=local_epochs,
            learning_rate=learning_rate, use_mixed_precision=use_mixed_precision,
            proximal_mu=proximal_mu,
            poison_rate=poison_rate,
            poison_direction=poison_direction,
            warmup_rounds=warmup
        )
        return client.to_client()
        
    else:
        # Fallback to Benign if unknown attack type
        print(f"⚠️ Unknown attack type '{attack_type}', defaulting to benign.")
        return FlowerClient(
            net, trainloader, testloader, device, local_epochs, 
            learning_rate, use_mixed_precision, proximal_mu,
            partition_id=partition_id
        ).to_client()

app = ClientApp(client_fn=client_fn)