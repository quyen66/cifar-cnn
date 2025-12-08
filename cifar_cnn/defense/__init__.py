from .layer1_dbscan import Layer1Detector
from .layer2_detection import Layer2Detector
from .noniid_handler import NonIIDHandler
from .scoring import ConfidenceScorer        
from .filtering import TwoStageFilter
from .reputation import ReputationSystem
from .mode_controller import ModeController
from .aggregation import (
    Aggregator,                            
    weighted_average_aggregation,
    trimmed_mean_aggregation,
    coordinate_median_aggregation,
    aggregate_by_mode
)

__all__ = [
    # 1. Detection Layers
    'Layer1Detector',
    'Layer2Detector',
    
    # 2. Analysis & Scoring
    'NonIIDHandler',
    'ConfidenceScorer', 
    
    # 3. Filtering
    'TwoStageFilter',
    
    # 4. Reputation & Mode
    'ReputationSystem',
    'ModeController',
    
    # 5. Aggregation
    'Aggregator',      
    'weighted_average_aggregation',
    'trimmed_mean_aggregation',
    'coordinate_median_aggregation',
    'aggregate_by_mode'
]

# Version info updated to match pyproject.toml
__version__ = '2.0.0'
__status__ = 'Production (Soft Pipeline V2)'

print(f"✅ Defense package loaded: v{__version__} ({__status__})")
print(f"   1. Layer1Detector: Enhanced DBSCAN (Hard Kill)")
print(f"   2. Layer2Detector: Distance + Direction (Redemption)")
print(f"   3. NonIIDHandler: Heterogeneity Analysis (H, δi)")
print(f"   4. ConfidenceScorer: Anomaly Scoring (ci calculation)")
print(f"   5. TwoStageFilter: Adaptive Thresholds (Hard/Soft)")
print(f"   6. ReputationSystem: Asymmetric EMA & Adaptive Penalty")
print(f"   7. ModeController: 3-mode Switching (Normal/Alert/Defense)")
print(f"   8. Aggregator: Mode-adaptive Aggregation")