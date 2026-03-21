"""
Attack Clients Module
═══════════════════════════════════════════════════════════════════════════════

Cung cấp các loại tấn công để test hệ thống phòng thủ.

PHÂN LOẠI 4 NHÓM:

1. NHÓM THÔ (Crude) - Target: Layer 1
   - RandomNoiseClient: Nhiễu ngẫu nhiên uniform
   - GaussianNoiseClient: Nhiễu Gaussian
   - ByzantineClient (sign_flip): Đảo dấu gradient

2. NHÓM TINH VI (Sophisticated) - Target: Layer 2
   - LabelFlippingClient: Đảo nhãn dữ liệu
   - BackdoorClient: Cấy trigger vào dữ liệu

3. NHÓM THỐNG KÊ (Statistical) - Target: Aggregation
   - MinMaxClient: Kéo lệch trung vị
   - MinSumClient: Tối thiểu hóa tổng

4. NHÓM THÍCH ỨNG (Adaptive) - Target: Reputation + Mode + Filtering
   - OnOffAttackClient: Tấn công không liên tục (NEW)
   - SlowPoisoningClient: Tấn công từ từ (NEW)
"""

# Base class
from .base import AttackClient

# Nhóm 1: Tấn công thô (Layer 1)
from .noise import RandomNoiseClient, GaussianNoiseClient
from .byzantine import ByzantineClient

# Nhóm 2: Tấn công tinh vi (Layer 2)
from .label_flip import LabelFlippingClient
from .backdoor import BackdoorClient

# Nhóm 3: Tấn công thống kê (Aggregation)
from .minmax import MinMaxClient, MinSumClient

# Nhóm 4: Tấn công thích ứng (Reputation + Mode + Filtering)
from .on_off import OnOffAttackClient, IntermittentAttackClient
from .slow_poison import SlowPoisoningClient, GradualPoisoningClient, StealthyPoisoningClient

# Legacy import (có thể bỏ sau)
try:
    from .alie import ALIEClient
except ImportError:
    ALIEClient = None 

__all__ = [
    # Base
    'AttackClient',
    
    # Nhóm 1: Thô
    'RandomNoiseClient',
    'GaussianNoiseClient', 
    'ByzantineClient',
    
    # Nhóm 2: Tinh vi
    'LabelFlippingClient',
    'BackdoorClient',
    
    # Nhóm 3: Thống kê
    'MinMaxClient',
    'MinSumClient',
    
    # Nhóm 4: Thích ứng 
    'OnOffAttackClient',
    'IntermittentAttackClient',
    'SlowPoisoningClient',
    'GradualPoisoningClient',
    'StealthyPoisoningClient',
    
    # Legacy
    'ALIEClient',
]

__version__ = '2.1.0'

print(f"✅ Attacks module loaded: v{__version__}")
print(f"   Nhóm 1 (Thô):      RandomNoise, GaussianNoise, SignFlip")
print(f"   Nhóm 2 (Tinh vi):  LabelFlip, Backdoor")
print(f"   Nhóm 3 (Thống kê): MinMax, MinSum")
print(f"   Nhóm 4 (Thích ứng): OnOff, SlowPoison [NEW]")