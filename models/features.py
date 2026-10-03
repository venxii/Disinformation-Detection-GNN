"""Train-only feature statistics for optional text+graph experiments."""

from __future__ import annotations

import torch


class FeatureNormalizer:
    """Standardize node features using statistics fitted on training graphs only."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)

    @classmethod
    def fit(cls, data_objects):
        if not data_objects:
            raise ValueError("cannot fit feature normalizer on an empty training set")
        features = torch.cat([data.x.detach().float() for data in data_objects], dim=0)
        return cls(features.mean(dim=0), features.std(dim=0, unbiased=False))

    def transform(self, data):
        if data.x.size(1) != self.mean.numel():
            raise ValueError("feature dimension does not match fitted normalizer")
        data.x = (data.x.float() - self.mean) / self.std
        return data

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["mean"], state["std"])
