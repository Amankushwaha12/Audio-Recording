"""
Configuration dataclasses matching Saito et al. (ICASSP 2023, arXiv:2211.04124)
Table 2 and Section 4 experimental settings.

Every parameter not explicitly stated in the paper is marked ASSUMPTION.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class STFTConfig:
    """Complex STFT configuration from Saito et al. Section 4."""

    sample_rate: int = 44100  # Configurable; paper uses 44.1 kHz mono vocals (NHSS / NUS-48E)
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    patch_frames: int = 512  # 512 x 512 complex STFT after dropping DC bin
    drop_dc: bool = True
    # ASSUMPTION: PyTorch center=True + normalized=True (1/sqrt(n_fft)) with optional
    # scalar stft_scale saved in checkpoint metadata so train/infer variance stays matched.
    center: bool = True
    normalized: bool = True
    stft_scale: float = 1.0
    # ASSUMPTION: Overlapping window hop for files longer than patch_frames (50% overlap,
    # raised-cosine crossfade in time domain after ISTFT).
    window_hop_frames: int = 256


@dataclass
class UNetConfig:
    """Modified guided-diffusion / DDRM U-Net hyperparameters (Paper Table 2)."""

    in_channels: int = 2  # Real & imaginary components of complex STFT
    out_channels: int = 2
    model_channels: int = 64  # Base channels = 64
    num_res_blocks: int = 2  # Depth = 2 residual blocks per resolution
    channel_mult: Tuple[int, ...] = (1, 1, 2, 2, 4, 4)
    attention_resolutions: Tuple[int, ...] = (32, 16)
    num_heads: int = 2
    dropout: float = 0.0
    resblock_updown: bool = True  # BigGAN-style up/downsampling residual blocks
    use_scale_shift_norm: bool = True  # Adaptive Group Normalization (AdaGN)
    use_time_distributed_fc: bool = True  # Time-distributed FC at end of every ResBlock
    # ASSUMPTION: Frequency dimension of the input spectrogram (n_fft // 2 when DC removed).
    input_freq_bins: int = 512
    # ASSUMPTION: GroupNorm groups = 32 (standard in guided-diffusion; divides 64, 128, 256).
    norm_groups: int = 32


@dataclass
class TrainingConfig:
    """Unsupervised dry-only diffusion prior training settings (Paper Table 2)."""

    diffusion_steps: int = 4000
    noise_schedule: str = "cosine"
    batch_size: int = 6
    max_iters: int = 370_000
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0  # ASSUMPTION: Standard guided-diffusion AdamW weight_decay=0.0
    use_amp: bool = True  # 16-bit mixed precision
    ema_rate: float = 0.9999
    grad_clip_norm: float = 1.0  # ASSUMPTION: Gradient norm clip for fp16 stability
    log_every: int = 100
    save_every: int = 10_000
    num_workers: int = 4
    seed: int = 42


@dataclass
class WPEConfig:
    """Per-frequency WPE filter estimation settings."""

    taps: int = 150  # L = 150 for proposed Algorithm 1 initialization
    delay: int = 8  # D = 8 frames
    iterations: int = 1  # 1 iteration for Algorithm 1 initialization
    # ASSUMPTION: Diagonal ridge regularization for numerical stability of R_k^{-1} p_k
    ridge_eps: float = 1e-6
    # ASSUMPTION: Variance floor preventing division by zero in silent STFT frames
    psd_floor: float = 1e-10
    # ASSUMPTION: Half-width of temporal context for WPE PSD smoothing (0 = per-frame |x|^2)
    psd_context: int = 0

    @classmethod
    def baseline(cls) -> "WPEConfig":
        """Paper baseline: WPE alone with L=200, D=8, 3 iterations."""
        return cls(taps=200, delay=8, iterations=3)


@dataclass
class DDRMConfig:
    """DDRM inference settings (Paper Section 4 / Kawar et al. 2022 Eq. 8)."""

    eta: float = 0.7
    eta_b: float = 0.2
    sigma_y: float = 1.0e-6
    sampling_steps: int = 20
    # ASSUMPTION: Singular value clamp floor when inverting Sigma_A -> Sigma_H = 1 / Sigma_A
    svd_eps: float = 1e-5
    # ASSUMPTION: Frequency chunk size during batched SVD to bound GPU/CPU memory
    svd_freq_chunk: int = 32


@dataclass
class RefinementConfig:
    """Algorithm 1 Step 3 filter refinement via PyTorch autograd."""

    alpha: float = 1.0e-6  # Gradient descent step size
    lam: float = 1.0  # L2 penalty lambda * ||g_k||^2
    n_refine: int = 10_000  # Paper default: 10,000 iterations
    # ASSUMPTION: --fast mode uses fewer refinement iterations (and optionally 10 DDRM steps)
    # for interactive experimentation; clearly trade-offs refinement convergence for runtime.
    fast_n_refine: int = 500
    fast_ddrm_steps: int = 10
    # ASSUMPTION: Optional Adam vs plain GD; paper Eq. in Algorithm 1 specifies plain GD:
    # g_k <- g_k - alpha * grad_{g_k} L(g_k), so default is exact plain gradient descent.
    use_plain_gd: bool = True


@dataclass
class PostProcessConfig:
    """Safety & loudness-matching post-processing rules."""

    match_loudness: bool = True
    loudness_tolerance_lu: float = 0.5  # ITU-R BS.1770 match within +/- 0.5 LU
    true_peak_ceiling_dbtp: float = -1.5  # -1.5 dBTP true-peak limiter
    dry_mix: float = 0.15  # 10-20% original dry/wet blend (default 15%)
    oversample_factor: int = 4  # 4x oversampled true-peak estimation


@dataclass
class DereverbPipelineConfig:
    """Top-level configuration aggregating all stages of Algorithm 1."""

    stft: STFTConfig = field(default_factory=STFTConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wpe: WPEConfig = field(default_factory=WPEConfig)
    ddrm: DDRMConfig = field(default_factory=DDRMConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    post: PostProcessConfig = field(default_factory=PostProcessConfig)
    fast: bool = False
    skip_refinement: bool = False  # Enables ablation: Proposed (no refinement)
    baseline_wpe_only: bool = False  # Enables baseline: WPE alone (L=200, D=8, 3 iters)
    seed: int = 42
