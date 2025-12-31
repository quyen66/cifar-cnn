"""
On-Off Attack Implementation - FIXED VERSION V3
═══════════════════════════════════════════════════════════════════════════════

BUG FIX V3: Handle case where server_round is NOT passed in config

PROBLEM DISCOVERED:
- FIXED V2 assumed server_round is passed in config
- But Flower strategy may NOT pass server_round by default!
- config.get("server_round", 1) always returns 1
- 1 % 5 != 0 → attack NEVER triggers!

SOLUTION V3:
- Primary: Use server_round from config if available
- Fallback: Use client round counter if server_round not available
- Also: Support frequency=1 for testing (attack every round)

NOTE: The BEST fix is to modify server_app.py to pass server_round in config.
This is a workaround for when that's not possible.
"""

import numpy as np
from .base import AttackClient
from cifar_cnn.task import get_parameters


class OnOffAttackClient(AttackClient):
    """
    On-Off Attack: Tấn công không liên tục để exploit reputation system.
    
    FIXED V3: Handle missing server_round in config
    """
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 attack_frequency=5, attack_type="sign_flip", attack_scale=2.0):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        
        self.attack_frequency = attack_frequency
        # Normalize attack_type
        if attack_type in ["on_off", "onoff", "default", None, ""]:
            attack_type = "sign_flip"
        self.attack_type = attack_type
        self.attack_scale = attack_scale
        
        # Client round counter (fallback if server_round not in config)
        self.round_counter = 0
        self._warned_about_fallback = False
        
        print(f"   📊 On-Off Attack:")
        print(f"      Frequency: 1/{attack_frequency} rounds")
        print(f"      Attack Type: {self.attack_type}")
        print(f"      Scale: {attack_scale}")

    def fit(self, parameters, config):
        # Increment client counter
        self.round_counter += 1
        
        # ═══════════════════════════════════════════════════════════════════
        # FIX V3: Try server_round from config, fallback to client counter
        # ═══════════════════════════════════════════════════════════════════
        server_round = config.get("server_round", None)
        
        if server_round is None:
            # Fallback to client counter
            server_round = self.round_counter
            if not self._warned_about_fallback:
                print(f"   ⚠️ [On-Off] WARNING: server_round not in config!")
                print(f"      Using client round counter as fallback.")
                print(f"      For best results, add server_round to strategy config.")
                self._warned_about_fallback = True
        
        # Train bình thường để có benign gradient
        results = self.train_with_fedprox(parameters)
        trained_params = get_parameters(self.net)
        
        # Quyết định attack dựa trên round
        # Special case: frequency=1 means attack every round
        if self.attack_frequency == 1:
            is_attack_round = True
        else:
            is_attack_round = (server_round % self.attack_frequency == 0)
        
        # DEBUG LOG
        if is_attack_round:
            print(f"   ⚡ [On-Off] ROUND {server_round}: ATTACKING with {self.attack_type}!")
        
        if not is_attack_round:
            # ════════════════════════════════════════════════════════════
            # BENIGN ROUND: Return normal gradient
            # ════════════════════════════════════════════════════════════
            results["is_malicious"] = 1
            results["attack_active"] = 0
            results["attack_type"] = "on_off_benign"
            results["round_used"] = server_round
            return trained_params, len(self.trainloader.dataset), results
        
        # ════════════════════════════════════════════════════════════════
        # ATTACK ROUND: Apply malicious transformation
        # ════════════════════════════════════════════════════════════════
        malicious_params = []
        
        # For debugging: track cosine
        total_cosine = 0.0
        num_layers = 0
        
        for w_g, w_t in zip(parameters, trained_params):
            w_t = np.array(w_t, dtype=np.float32) if not isinstance(w_t, np.ndarray) else w_t
            w_g = np.array(w_g, dtype=np.float32) if not isinstance(w_g, np.ndarray) else w_g
            
            # Benign update vector
            u = w_t - w_g
            u_benign = u.copy()
            
            # Apply attack based on type
            if self.attack_type == "sign_flip":
                # Đảo ngược hoàn toàn hướng gradient
                u_mal = -u
                
            elif self.attack_type == "scale":
                # Scale gradient lên
                u_mal = self.attack_scale * u
                
            elif self.attack_type == "noise":
                # Thêm noise mạnh
                noise = np.random.normal(0, self.attack_scale * np.std(u), u.shape)
                u_mal = u + noise.astype(np.float32)
                
            elif self.attack_type == "zero":
                # Không đóng góp
                u_mal = np.zeros_like(u)
                
            else:
                # Default: sign flip
                u_mal = -u
            
            # DEBUG: Calculate cosine
            norm_benign = np.linalg.norm(u_benign)
            norm_mal = np.linalg.norm(u_mal)
            if norm_benign > 1e-10 and norm_mal > 1e-10:
                cos = np.dot(u_benign.flatten(), u_mal.flatten()) / (norm_benign * norm_mal)
                total_cosine += cos
                num_layers += 1
            
            w_mal = w_g + u_mal
            malicious_params.append(w_mal.astype(np.float32))
        
        # DEBUG: Log average cosine
        avg_cosine = total_cosine / max(num_layers, 1)
        print(f"   ⚡ [On-Off] Round {server_round}: Attack={self.attack_type}, "
              f"Cosine(benign,mal)={avg_cosine:.4f}")
        
        # Verify sign flip
        if self.attack_type == "sign_flip" and avg_cosine > -0.9:
            print(f"   ⚠️ WARNING: Sign flip may not work! Expected cos≈-1, got {avg_cosine:.4f}")
        
        results["is_malicious"] = 1
        results["attack_active"] = 1
        results["attack_type"] = f"on_off_{self.attack_type}"
        results["round_used"] = server_round
        results["attack_cosine"] = avg_cosine
        
        return malicious_params, len(self.trainloader.dataset), results


class PeriodicAttackClient(OnOffAttackClient):
    """Alias cho OnOffAttackClient."""
    pass


class IntermittentAttackClient(AttackClient):
    """
    Intermittent Attack: Random timing variant.
    """
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 attack_probability=0.2, attack_type="sign_flip", attack_scale=2.0):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        
        self.attack_probability = attack_probability
        self.round_counter = 0
        
        if attack_type in ["on_off", "onoff", "default", None, ""]:
            attack_type = "sign_flip"
        self.attack_type = attack_type
        self.attack_scale = attack_scale
        
        print(f"   📊 Intermittent Attack: prob={attack_probability}, type={attack_type}")

    def fit(self, parameters, config):
        self.round_counter += 1
        server_round = config.get("server_round", self.round_counter)
        
        results = self.train_with_fedprox(parameters)
        trained_params = get_parameters(self.net)
        
        # Random decision - use server round as seed for consistency across clients
        np.random.seed(server_round * 1000 + hash(str(id(self))) % 1000)
        is_attack_round = np.random.random() < self.attack_probability
        np.random.seed(None)  # Reset seed
        
        if not is_attack_round:
            results["is_malicious"] = 1
            results["attack_active"] = 0
            results["attack_type"] = "intermittent_benign"
            return trained_params, len(self.trainloader.dataset), results
        
        print(f"   ⚡ [Intermittent] Round {server_round}: ATTACKING!")
        
        # Attack logic
        malicious_params = []
        
        for w_g, w_t in zip(parameters, trained_params):
            w_t = np.array(w_t, dtype=np.float32) if not isinstance(w_t, np.ndarray) else w_t
            w_g = np.array(w_g, dtype=np.float32) if not isinstance(w_g, np.ndarray) else w_g
            
            u = w_t - w_g
            
            if self.attack_type == "sign_flip":
                u_mal = -u
            elif self.attack_type == "scale":
                u_mal = self.attack_scale * u
            elif self.attack_type == "noise":
                noise = np.random.normal(0, self.attack_scale * np.std(u), u.shape)
                u_mal = u + noise.astype(np.float32)
            else:
                u_mal = -u
            
            w_mal = w_g + u_mal
            malicious_params.append(w_mal.astype(np.float32))
        
        results["is_malicious"] = 1
        results["attack_active"] = 1
        results["attack_type"] = f"intermittent_{self.attack_type}"
        
        return malicious_params, len(self.trainloader.dataset), results