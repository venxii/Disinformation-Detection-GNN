"""Transparent, revisable WAIT/PROVISIONAL/CONFIRMED state machine."""

from __future__ import annotations

from dataclasses import dataclass, field

from .contradiction import ContradictionConfig, contradiction_gate


@dataclass(frozen=True)
class ControllerConfig:
    provisional_confidence: float = 0.85
    confirm_confidence: float = 0.90
    min_nodes_provisional: int = 3
    min_nodes_confirm: int = 5
    stable_observations: int = 2
    revisit_confirmed: bool = True
    contradiction: ContradictionConfig = field(default_factory=ContradictionConfig)


@dataclass
class DecisionState:
    status: str = "UNDECIDED"
    label: int | None = None
    history: list[dict] = field(default_factory=list)


class DecisionController:
    def __init__(self, config: ControllerConfig = ControllerConfig()):
        self.config = config

    def update(self, state: DecisionState, *, prediction: int, confidence: float, num_nodes: int, stable_count: int, contradiction_score: float = 0.0, new_evidence: int = 0, contradictory_evidence: int = 0):
        material = contradiction_gate(score=contradiction_score, new_evidence=new_evidence, contradictory_evidence=contradictory_evidence, config=self.config.contradiction)
        previous_status = state.status
        if material and state.status in {"PROVISIONAL", "CONFIRMED"} and (state.status != "CONFIRMED" or self.config.revisit_confirmed):
            state.status, state.label = "REOPENED", None
            state.history.append({"previous_status": previous_status, "status": state.status, "label": state.label, "prediction": prediction, "confidence": confidence, "material_contradiction": material})
            return state
        if state.status == "REOPENED":
            state.status = "WAIT"
        if state.status in {"UNDECIDED", "WAIT"}:
            if confidence >= self.config.provisional_confidence and num_nodes >= self.config.min_nodes_provisional and stable_count >= self.config.stable_observations:
                state.status, state.label = "PROVISIONAL", prediction
            else:
                state.status = "WAIT"
        elif state.status == "PROVISIONAL":
            if prediction != state.label:
                state.status, state.label = "WAIT", None
            elif confidence >= self.config.confirm_confidence and num_nodes >= self.config.min_nodes_confirm and stable_count >= self.config.stable_observations:
                state.status = "CONFIRMED"
        elif state.status == "CONFIRMED" and prediction != state.label:
            state.status, state.label = "WAIT", None
        state.history.append({"previous_status": previous_status, "status": state.status, "label": state.label, "prediction": prediction, "confidence": confidence, "material_contradiction": material})
        return state
