import unittest

from decision.calibration import TemperatureScaler
from decision.controller import ControllerConfig, DecisionController, DecisionState
from decision.contradiction import ContradictionConfig, contradiction_gate


class DecisionTests(unittest.TestCase):
    def controller(self):
        return DecisionController(ControllerConfig(provisional_confidence=.8, confirm_confidence=.9, min_nodes_provisional=2, min_nodes_confirm=3, stable_observations=2, contradiction=ContradictionConfig(score_threshold=.7, min_new_evidence=2, min_contradictory_evidence=2)))

    def test_wait_then_provisional_then_confirmed(self):
        c, s = self.controller(), DecisionState()
        c.update(s, prediction=0, confidence=.95, num_nodes=2, stable_count=1)
        self.assertEqual(s.status, "WAIT")
        c.update(s, prediction=0, confidence=.95, num_nodes=2, stable_count=2)
        self.assertEqual(s.status, "PROVISIONAL")
        c.update(s, prediction=0, confidence=.95, num_nodes=3, stable_count=2)
        self.assertEqual(s.status, "CONFIRMED")

    def test_weak_contradiction_does_not_reopen(self):
        c, s = self.controller(), DecisionState("PROVISIONAL", 0)
        c.update(s, prediction=0, confidence=.95, num_nodes=4, stable_count=2, contradiction_score=.9, new_evidence=1, contradictory_evidence=1)
        self.assertEqual(s.status, "CONFIRMED")

    def test_late_material_contradiction_reopens_confirmed(self):
        c, s = self.controller(), DecisionState("CONFIRMED", 0)
        c.update(s, prediction=0, confidence=.95, num_nodes=10, stable_count=3, contradiction_score=.9, new_evidence=4, contradictory_evidence=3)
        self.assertEqual(s.history[-1]["material_contradiction"], True)
        self.assertEqual(s.status, "REOPENED")
        c.update(s, prediction=1, confidence=.91, num_nodes=12, stable_count=2)
        self.assertIn(s.status, {"WAIT", "PROVISIONAL"})

    def test_prediction_change_is_not_contradiction(self):
        self.assertFalse(contradiction_gate(score=0.0, new_evidence=0, contradictory_evidence=0))

    def test_temperature_scaler_round_trip(self):
        import tempfile
        import torch
        logits = torch.tensor([[4., 0.], [0., 4.], [3., 1.], [1., 3.]])
        labels = torch.tensor([0, 1, 0, 1])
        scaler = TemperatureScaler().fit(logits, labels)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            scaler.save(f.name)
            loaded = TemperatureScaler.load(f.name)
            self.assertAlmostEqual(float(scaler.temperature), float(loaded.temperature), places=5)


if __name__ == "__main__":
    unittest.main()
