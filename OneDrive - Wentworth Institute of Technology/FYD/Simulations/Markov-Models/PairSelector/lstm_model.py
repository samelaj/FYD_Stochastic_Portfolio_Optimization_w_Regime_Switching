"""
lstm_model.py
─────────────
Two-head LSTM for pair scoring. CPU-optimized, ~25k parameters.

Input:  (batch, seq_len=20, n_features=12)
Output: (batch, 2) — RAW LOGITS (not probabilities)
  output[:,0] = logit for P(mean reversion within 5 days)   — reversion head
  output[:,1] = logit for P(cointegration within 63 days)   — emergence head

IMPORTANT: output is raw logits, not sigmoid probabilities.
  - During training: use BCEWithLogitsLoss (applies sigmoid internally)
  - During inference: apply torch.sigmoid() to get probabilities

Architecture: 1-layer LSTM → shared FC → 2 output heads
  hidden_dim=64, shared_dim=32, dropout=0.3
  ~25k params → trains in <2 min on CPU
"""

import torch
import torch.nn as nn


class PairScorerLSTM(nn.Module):
    """
    Two-head LSTM for pair scoring.

    Outputs raw logits (not probabilities). Apply torch.sigmoid() at inference.
    Use BCEWithLogitsLoss during training for numerical stability.
    """
    def __init__(self, n_features: int = 12, hidden_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,   # dropout not applied with num_layers=1
        )
        self.dropout = nn.Dropout(dropout)

        # Shared representation after LSTM
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Head 1: reversion (outputs raw logit)
        self.reversion_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            # NO Sigmoid here — BCEWithLogitsLoss handles it during training
            # Apply torch.sigmoid() manually at inference
        )

        # Head 2: emergence (outputs raw logit)
        self.emergence_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        Returns: (batch, 2) — raw logits
        """
        lstm_out, _ = self.lstm(x)
        last         = self.dropout(lstm_out[:, -1, :])   # last timestep
        shared       = self.shared_fc(last)

        logit_rev  = self.reversion_head(shared)   # (batch, 1)
        logit_emer = self.emergence_head(shared)   # (batch, 1)

        return torch.cat([logit_rev, logit_emer], dim=1)   # (batch, 2)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time: returns sigmoid probabilities (batch, 2)."""
        with torch.no_grad():
            logits = self(x)
        return torch.sigmoid(logits)

    def predict_pair(self, sequence: torch.Tensor) -> tuple:
        """
        Score a single sequence (1, seq_len, n_features).
        Returns (p_reversion, p_emergence) as floats.
        """
        proba = self.predict_proba(sequence.unsqueeze(0) if sequence.dim() == 2
                                   else sequence)
        return float(proba[0, 0]), float(proba[0, 1])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import torch

    model = PairScorerLSTM(n_features=12, hidden_dim=64, dropout=0.3)
    print(f"Model parameters: {count_parameters(model):,}")

    # Shape test
    x     = torch.randn(4, 20, 12)
    logit = model(x)
    prob  = torch.sigmoid(logit)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape (logits): {logit.shape}")
    print(f"Output shape (proba):  {prob.shape}")
    assert logit.shape == (4, 2), f"Expected (4,2), got {logit.shape}"
    print("Shape assertion passed.")

    # Check probabilities are in [0,1]
    assert prob.min() >= 0.0 and prob.max() <= 1.0, "Probs out of range"
    print("Probability range check passed.")
