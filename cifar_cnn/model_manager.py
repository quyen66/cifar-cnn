"""Model saving and loading utilities."""

import torch
import json
from pathlib import Path


class ModelManager:
    """Quản lý việc lưu và load models."""
    
    def __init__(self, save_dir="saved_models"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_model(self, net, metadata, model_name=None):
        """
        Lưu model và metadata.
        
        Args:
            net: PyTorch model
            metadata: Dict chứa thông tin về model
            model_name: Tên folder (auto-generate nếu None)
        """
        if model_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{timestamp}"
        
        model_dir = self.save_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Lưu model weights
        model_path = model_dir / "model.pth"
        torch.save(net.state_dict(), model_path)
        
        # Lưu metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"MODEL SAVED")
        print(f"{'='*70}")
        print(f"  Path: {model_dir}")
        print(f"  Model: {model_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"{'='*70}\n")
        
        return str(model_dir)
    
    def load_model(self, net, model_name):
        """Load model từ saved directory."""
        model_dir = self.save_dir / model_name
        model_path = model_dir / "model.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        net.load_state_dict(torch.load(model_path, weights_only=True))
        
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        print(f"Model loaded from {model_path}")
        return net, metadata
    
    def list_models(self):
        """List tất cả saved models."""
        models = []
        for model_dir in self.save_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    models.append({
                        'name': model_dir.name,
                        'path': str(model_dir),
                        'metadata': metadata
                    })
        return models