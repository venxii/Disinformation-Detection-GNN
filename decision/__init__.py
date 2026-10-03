"""Evidence-aware decision utilities kept separate from the GNN."""

from .calibration import TemperatureScaler
from .controller import ControllerConfig, DecisionController, DecisionState
from .contradiction import ContradictionConfig, contradiction_gate
from .trajectory import build_trajectory, trajectory_snapshots
from .metrics import brier_score, classification_metrics, expected_calibration_error, selective_metrics

__all__ = ["TemperatureScaler", "ControllerConfig", "DecisionController", "DecisionState", "ContradictionConfig", "contradiction_gate", "build_trajectory", "trajectory_snapshots", "classification_metrics", "brier_score", "expected_calibration_error", "selective_metrics"]
