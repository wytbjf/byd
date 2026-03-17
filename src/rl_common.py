from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUPolicy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, h=None):
        y, h = self.gru(x, h)
        return self.head(y[:, -1]), h


@dataclass
class Batch:
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    s2: torch.Tensor
    d: torch.Tensor


class ReplayBuffer:
    def __init__(self, size: int):
        self.buf: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = deque(maxlen=size)

    def add(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        arr = [self.buf[i] for i in idx]
        s, a, r, s2, d = [np.array(x) for x in zip(*arr)]
        return Batch(
            s=torch.tensor(s, dtype=torch.float32, device=device),
            a=torch.tensor(a, dtype=torch.float32, device=device),
            r=torch.tensor(r, dtype=torch.float32, device=device),
            s2=torch.tensor(s2, dtype=torch.float32, device=device),
            d=torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buf)


def soft_update(src: nn.Module, dst: nn.Module, tau: float) -> None:
    for p, q in zip(src.parameters(), dst.parameters()):
        q.data.mul_(1 - tau).add_(tau * p.data)
