"""Official-SAFE-style independent MLP and sequential LSTM predictors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


try:
    import torch
    from torch import nn
except ImportError:  # Keep dataset and CLI help usable without the training extra.
    torch = None
    nn = None


@dataclass
class ModelConfig:
    model_type: str
    input_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    cumulative_mlp: bool = False


def require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for SAFE model training. Install the project "
            "training extra or `pip install torch`."
        )


if nn is not None:

    class SafeMLP(nn.Module):
        """SAFE `indep` predictor: a per-inference MLP with sigmoid score."""

        def __init__(self, input_dim, hidden_dim=256, num_layers=2, cumulative=False):
            super().__init__()
            layers = []
            if num_layers <= 1:
                layers.append(nn.Linear(input_dim, 1))
            else:
                layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
                for _ in range(num_layers - 2):
                    layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
                layers.append(nn.Linear(hidden_dim, 1))
            layers.append(nn.Sigmoid())
            self.projector = nn.Sequential(*layers)
            self.cumulative = bool(cumulative)

        def forward(self, features, lengths=None):
            scores = self.projector(features).squeeze(-1)
            if self.cumulative:
                scores = torch.cumsum(scores, dim=1)
            return scores


    class SafeLSTM(nn.Module):
        """SAFE LSTM predictor returning a failure probability at every inference."""

        def __init__(self, input_dim, hidden_dim=256, num_layers=1, dropout=0.0):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, features, lengths=None):
            if lengths is None:
                output, _ = self.lstm(features)
            else:
                packed = nn.utils.rnn.pack_padded_sequence(
                    features,
                    lengths.detach().cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed_output, _ = self.lstm(packed)
                output, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_output,
                    batch_first=True,
                    total_length=features.shape[1],
                )
            return torch.sigmoid(self.fc(self.dropout(output))).squeeze(-1)


else:

    class SafeMLP:  # pragma: no cover - exercised only in dependency-poor envs.
        def __init__(self, *args, **kwargs):
            require_torch()


    class SafeLSTM:
        def __init__(self, *args, **kwargs):
            require_torch()


def make_model(config: ModelConfig):
    require_torch()
    if config.model_type == "mlp":
        return SafeMLP(
            config.input_dim,
            config.hidden_dim,
            config.num_layers,
            config.cumulative_mlp,
        )
    if config.model_type == "lstm":
        return SafeLSTM(
            config.input_dim,
            config.hidden_dim,
            config.num_layers,
            config.dropout,
        )
    raise ValueError(f"Unknown SAFE model type {config.model_type!r}")


def masked_binary_cross_entropy(scores, labels, valid_mask, class_weights=None):
    """Apply one binary rollout label to every valid inference timestep."""
    require_torch()
    targets = labels.float().unsqueeze(1).expand_as(scores)
    losses = nn.functional.binary_cross_entropy(scores, targets, reduction="none")
    if class_weights is not None:
        success_weight, failure_weight = class_weights
        weights = torch.where(targets > 0.5, failure_weight, success_weight)
        losses = losses * weights
    valid_mask = valid_mask.to(dtype=losses.dtype)
    return (losses * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def save_checkpoint(path: str | Path, model, config: ModelConfig, extra=None):
    require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "config": asdict(config), "extra": extra or {}},
        path,
    )
    return path


def load_checkpoint(path: str | Path, device="cpu"):
    require_torch()
    payload = torch.load(path, map_location=device)
    config = ModelConfig(**payload["config"])
    model = make_model(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, payload.get("extra", {})
