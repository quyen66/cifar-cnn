from .layer1_dbscan import Layer1Detector
from .layer2_detection import Layer2Detector
from .noniid_handler import NonIIDHandler  # FIXED
from .filtering import TwoStageFilter  # FIXED
from .reputation import ReputationSystem
from .mode_controller import ModeController
from .aggregation import (
    weighted_average_aggregation,
    trimmed_mean_aggregation,
    coordinate_median_aggregation,
    aggregate_by_mode
)

__all__ = [
    # Detection layers
    'Layer1Detector',
    'Layer2Detector',
    
    # Non-IID handling (FIXED)
    'NonIIDHandler',
    
    # Filtering (FIXED)
    'TwoStageFilter',
    
    # Reputation system
    'ReputationSystem',
    
    # Mode control
    'ModeController',
    
    # Aggregation methods
    'weighted_average_aggregation',
    'trimmed_mean_aggregation',
    'coordinate_median_aggregation',
    'aggregate_by_mode'
]

# Version info
__version__ = '1.0.1-fixed'
__status__ = 'Production (Fixed)'

print(f"✅ Defense package loaded: v{__version__} ({__status__})")
print(f"   - Layer1Detector: Enhanced DBSCAN")
print(f"   - Layer2Detector: Distance + Direction")
print(f"   - NonIIDHandler: WITH Baseline Tracking ✓")
print(f"   - TwoStageFilter: WITH Adaptive Thresholds ✓")
print(f"   - ReputationSystem: Asymmetric EMA")
print(f"   - ModeController: 3-mode switching")
print(f"   - Aggregation: Mode-adaptive methods")