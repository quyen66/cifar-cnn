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
from .adaptive_reference import AdaptiveReferenceTracker

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
    'aggregate_by_mode',  
    
    # 6. Adaptive Reference (NEW)
    'AdaptiveReferenceTracker',
    
]

# Version info updated to match pyproject.toml
__version__ = '2.1.0'
__status__ = 'Production (Soft Pipeline V2 + Adaptive Reference)'