"""
Defense Package
===============
Full defense pipeline với configurable parameters.
"""

from .layer1_dbscan import Layer1Detector
from .layer2_detection import Layer2Detector
from .noniid_handler import NonIIDHandler
from .filtering import TwoStageFilter
from .reputation import ReputationSystem
from .mode_controller import ModeController
from .aggregation import (
    weighted_average_aggregation,
    trimmed_mean_aggregation,
    coordinate_median_aggregation,
    aggregate_by_mode
)

__all__ = [
    'Layer1Detector',
    'Layer2Detector',
    'NonIIDHandler',
    'TwoStageFilter',
    'ReputationSystem',
    'ModeController',
    'weighted_average_aggregation',
    'trimmed_mean_aggregation',
    'coordinate_median_aggregation',
    'aggregate_by_mode'
]