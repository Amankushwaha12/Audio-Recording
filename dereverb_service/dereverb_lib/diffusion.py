"""
Cosine noise schedule (4000 steps), x0-prediction diffusion training loss (Eq. 9),
and Exponential Moving Average (EMA 0.9999) helper.

Paper specification (Saito et al., ICASSP 2023):
  - Loss (Eq. 9): E_{t, q} [ gamma_t * || x_0 - f_theta^(t)(x_t) ||_2^2 ]
  - 4000 diffusion steps, cosine schedule (Nichol & Dhariwal, Improved DDPM)
  - EMA rate: 0.9999
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """
    Improved DDPM cosine beta schedule (Nichol & Dhariwal, 2021).
    Returns betas of shape (num_timesteps,) in float64.
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps, dtype=torch.float64) / num_timesteps
    alphas_cumprod = torch.cos((t + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-8, max=max_beta)


class GaussianDiffusionPrior:
    """
    Variance-preserving Gaussian diffusion process with x0 prediction (Eq. 9).

    ASSUMPTION on gamma_t weighting in Eq. 9:
      In standard x0-prediction DDPM (equivalent to uniform epsilon-loss),
      gamma_t = alpha_bar_t / (1 - alpha_bar_t) = SNR(t). For stability across
      t -> 0 where SNR(t) grows large, we support both:
        - "snr_clamped" (default, Min-SNR-5 / clamped SNR weighting: min(SNR(t), 5.0))
        - "uniform" (gamma_t = 1.0, unweighted x0 MSE)
        - "snr" (exact alpha_bar_t / (1 - alpha_bar_t), clamped at 100.0)
      The paper writes gamma_t in Eq. 9 without tabulating a custom weighting table;
      this choice is explicitly documented as ASSUMPTION.
    """

    def __init__(
        self,
        num_timesteps: int = 4000,
        schedule: str = "cosine",
        gamma_mode: str = "snr_clamped",
        device: Optional[torch.device] = None,
    ):
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.gamma_mode = gamma_mode

        if schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unsupported noise schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()
        # VE-equivalent noise levels for DDRM: sigma_t = sqrt((1 - alpha_bar_t) / alpha_bar_t)
        self.ddrm_sigmas = torch.sqrt((1.0 - alphas_cumprod) / torch.clamp(alphas_cumprod, min=1e-12)).float()

        snr = alphas_cumprod / torch.clamp(1.0 - alphas_cumprod, min=1e-8)
        if gamma_mode == "uniform":
            gamma = torch.ones_like(snr)
        elif gamma_mode == "snr":
            gamma = torch.clamp(snr, max=100.0)
        elif gamma_mode == "snr_clamped":
            gamma = torch.clamp(snr, max=5.0)
        else:
            raise ValueError(f"Unknown gamma_mode: {gamma_mode}")
        self.gamma_t = gamma.float()

        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> "GaussianDiffusionPrior":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.ddrm_sigmas = self.ddrm_sigmas.to(device)
        self.gamma_t = self.gamma_t.to(device)
        return self

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward noising q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_omab * noise

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Eq. 9 weighted x0-prediction MSE loss:
            L = E_{t, q} [ gamma_t * || x_0 - f_theta^(t)(x_t) ||_2^2 ]
        """
        batch_size = x0.shape[0]
        device = x0.device
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)
        x0_pred = model(x_t, t)

        # Per-sample MSE over (2, 512, 512)
        per_sample_mse = (x0 - x0_pred).float().pow(2).mean(dim=(1, 2, 3))
        weights = self.gamma_t[t].to(device)
        loss = (weights * per_sample_mse).mean()

        metrics = {
            "loss": float(loss.detach().item()),
            "unweighted_mse": float(per_sample_mse.detach().mean().item()),
        }
        return loss, metrics


class EMAHelper:
    """Exponential Moving Average (EMA) parameter tracker (decay = 0.9999)."""

    def __init__(self, model: nn.Module, rate: float = 0.9999):
        self.rate = rate
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for p_ema, p_src in zip(self.ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(self.rate).add_(p_src.data, alpha=1.0 - self.rate)
        for b_ema, b_src in zip(self.ema_model.buffers(), model.buffers()):
            b_ema.data.copy_(b_src.data)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.ema_model.load_state_dict(state_dict)
