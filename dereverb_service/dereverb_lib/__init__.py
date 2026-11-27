"""
Unsupervised Vocal Dereverberation with Diffusion-based Generative Models
(Saito et al., Sony, ICASSP 2023, arXiv:2211.04124)

Research prototype implementing:
  - Complex STFT (1024 window, 256 hop, Hann, DC removed -> 512 x 512 x 2)
  - Modified OpenAI guided-diffusion / DDRM U-Net with AdaGN & time-distributed FC
  - Unsupervised dry-only diffusion prior training (Eq. 9 weighted x0 MSE, 4000 steps cosine)
  - Per-frequency WPE filter estimation (L=150, D=8, 1 iter; baseline L=200, D=8, 3 iters)
  - Differentiable Toeplitz dereverberation operator (Eq. 3-6) and batched per-frequency SVD
  - DDRM reverse sampling (Kawar et al. 2022, Eq. 8) + autograd filter refinement (Algorithm 1)
  - BS.1770 loudness matching (+/- 0.5 LU), -1.5 dBTP true-peak limiter, dry/wet blend
"""

from .config import (
    DDRMConfig,
    DereverbPipelineConfig,
    PostProcessConfig,
    RefinementConfig,
    STFTConfig,
    TrainingConfig,
    UNetConfig,
    WPEConfig,
)

__all__ = [
    "STFTConfig",
    "UNetConfig",
    "TrainingConfig",
    "WPEConfig",
    "DDRMConfig",
    "RefinementConfig",
    "PostProcessConfig",
    "DereverbPipelineConfig",
]
