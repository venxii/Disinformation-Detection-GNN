"""Post-hoc temperature scaling fitted on validation logits only."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class TemperatureScaler(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, logits):
        return logits / self.temperature.clamp_min(1e-6)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, *, max_iter: int = 100):
        if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.size(0):
            raise ValueError("logits must be [N,C] and labels must be [N]")
        self.eval()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.temperature.clamp_(min=1e-3)
        return self

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"temperature": float(self.temperature.detach())}, path)

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(payload["temperature"])
