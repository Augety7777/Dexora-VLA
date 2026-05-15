"""
Discriminator score -> per-sample weight (w_i) for data-quality-aware
post-training (Dexora §III-D, Eq.(8)).

We follow the DWBC mapping from Xu et al. ICML 2022 ("Discriminator-Weighted
Offline Imitation Learning From Suboptimal Demonstrations", ref. [41] in the
Dexora paper):

    w_i = clip( d / (eta * (1 - d) + d) , w_min, w_max )

where ``d = d(C_i)`` is the discriminator output in (0, 1] and ``eta`` is the
PU-loss positive weight used at discriminator training time (paper: 0.5).
The mapping is monotonically increasing in ``d``, equals 1 when ``d -> 1`` and
``eta -> 0``, and pushes ``w -> 0`` as ``d -> 0``.

A short linear warm-up is applied during the first ``warmup_steps`` post-training
steps. The warm-up interpolates each sample's weight between 1.0 and its
DWBC-computed value, so the policy is initially trained with vanilla diffusion
loss and gradually transitions to the quality-weighted objective.
"""

from __future__ import annotations

from typing import Optional

import torch


def dwbc_score_to_weight(
    scores: torch.Tensor,
    eta: float = 0.5,
    w_min: float = 0.0,
    w_max: float = 5.0,
) -> torch.Tensor:
    """
    Convert calibrated discriminator scores into per-sample weights via DWBC.

    Args:
        scores: Tensor of discriminator outputs in (0, 1]. Any shape; weights
            come out with the same shape.
        eta: PU positive weight used at discriminator training time. The paper
            uses 0.5.
        w_min: Clamp weights from below to avoid exact zeros.
        w_max: Clamp weights from above so a few high-d outliers cannot
            dominate the loss.

    Returns:
        Weights tensor with the same shape & device as ``scores``.
    """
    # Run in float32 for numerical stability under bf16 / fp16 training.
    s = scores.detach().to(dtype=torch.float32)
    s = torch.clamp(s, 1e-6, 1.0 - 1e-6)
    weights = s / (eta * (1.0 - s) + s)
    weights = torch.clamp(weights, min=w_min, max=w_max)
    return weights.to(dtype=scores.dtype)


def warmup_weights(
    weights: torch.Tensor,
    global_step: int,
    warmup_steps: int = 1000,
) -> torch.Tensor:
    """
    Linearly interpolate per-sample weights between 1.0 and ``weights``
    during the first ``warmup_steps`` steps of post-training.

    Returns ``weights`` unchanged once ``global_step >= warmup_steps``.
    """
    if warmup_steps <= 0:
        return weights
    progress = min(max(global_step / float(warmup_steps), 0.0), 1.0)
    ones = torch.ones_like(weights)
    return ones + progress * (weights - ones)


def scores_to_train_weights(
    scores: Optional[torch.Tensor],
    *,
    eta: float = 0.5,
    w_min: float = 0.0,
    w_max: float = 5.0,
    warmup_steps: int = 1000,
    global_step: int = 0,
    fallback_shape: Optional[torch.Size] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convenience wrapper: discriminator scores -> warmed-up DWBC weights.

    If ``scores`` is None we fall back to all-ones weights of ``fallback_shape``,
    which means the policy is trained with the vanilla (unweighted) diffusion
    loss. This is the right behaviour for stage-1 (sim pretrain) and for
    sanity-check runs.
    """
    if scores is None:
        assert fallback_shape is not None, "Either scores or fallback_shape must be provided."
        return torch.ones(fallback_shape, device=device, dtype=dtype)

    w = dwbc_score_to_weight(scores, eta=eta, w_min=w_min, w_max=w_max)
    w = warmup_weights(w, global_step=global_step, warmup_steps=warmup_steps)
    return w
