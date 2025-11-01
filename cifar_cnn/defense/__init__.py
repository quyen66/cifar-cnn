"""Defense module - Week 2-3 implementation."""

# Week 2: Layer 1 Detection
from .layer1_dbscan import Layer1Detector

# Week 3: Layer 2 Detection + Reputation
from .layer2_detection import Layer2Detector
from .reputation import ReputationSystem

# Placeholders for future weeks
# from .filtering import TwoStageFilter  # Week 4
# from .aggregation import ModeAdaptiveAggregation  # Week 4
# from .mode_controller import ModeController  # Week 4

__all__ = [
    'Layer1Detector',
    'Layer2Detector',
    'ReputationSystem',
    # Will add more as we implement
]