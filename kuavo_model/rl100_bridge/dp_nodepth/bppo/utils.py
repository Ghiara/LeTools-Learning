"""Small import-safe subset of RL-100 utilities used by the BPPO bridge."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Distribution


CONST_EPS = 1e-10


def orthogonal_initWeights(net: nn.Module) -> None:
    """RL-100's original orthogonal initializer."""
    for parameter in net.parameters():
        if len(parameter.size()) >= 2:
            nn.init.orthogonal_(parameter)


def log_prob_func(dist: Distribution, action: torch.Tensor) -> torch.Tensor:
    """RL-100's original reduction for action log probabilities."""
    log_prob = dist.log_prob(action)
    if len(log_prob.shape) == 1:
        return log_prob
    return log_prob.sum(-1, keepdim=True)


class AdaptiveScheduler:
    """RL-100's adaptive learning-rate helper without unrelated Gym utilities."""

    def __init__(self, kl_threshold, min_lr, max_lr, init_lr):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.kl_threshold = kl_threshold
        self.current_lr = init_lr

    def update(self, kl_dist):
        lr = self.current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(self.current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(self.current_lr * 1.5, self.max_lr)
        self.current_lr = lr
        return lr
