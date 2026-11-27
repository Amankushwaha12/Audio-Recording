"""
DDRM Sampler (Kawar et al. 2022, Eq. 8) & Algorithm 1 Orchestrator
(Saito et al., Sony, ICASSP 2023, arXiv:2211.04124).

Algorithm 1:
  1. Estimate per-frequency WPE filter g_k (L=150, D=8, 1 iteration).
  2. Estimate dry signal x_bar_dry via DDRM (Eq. 8) using the pre-trained
     diffusion prior p_theta(x_dry) and linear operator H_k = (I~ - G_k)^dagger.
  3. Repeat N_refine times: refine g_k via autograd on
         || x_bar_dry - (I~ - G_k) y_wet ||_2^2 + lambda * || g_k ||_2^2.
  4. Run DDRM again with g_k_refined to get final dry estimate x_tilde_dry.

Also supports:
  - Baseline mode: WPE alone (L=200, D=8, 3 iterations)
  - Ablation mode: Proposed without refinement (stops after Step 2)
  - Oracle evaluation mode: DDRM given an oracle operator g_k* estimated from (dry, wet)
  - Long-file processing via overlapping 512-frame windows + raised-cosine crossfades
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import DDRMConfig, DereverbPipelineConfig, WPEConfig
from .diffusion import GaussianDiffusionPrior
from .operator import (
    PerFreqOperatorSVD,
    apply_dereverb_operator,
    compute_operator_svd,
    pack_wet_vector,
    refine_wpe_filter,
    unpack_dry_vector,
)
from .stft import (
    ch2_to_complex,
    complex_to_2ch,
    compute_istft,
    compute_stft,
    iter_overlapping_frame_slices,
    overlap_add_waveforms,
    postprocess_dereverb_output,
)
from .wpe import estimate_wpe_filter


def _build_ddrm_timesteps(total_steps: int, sampling_steps: int) -> List[int]:
    """
    Build descending schedule of `sampling_steps` integer timesteps from T-1 down to 0
    (matching bahjat-kawar/ddrm uniform stride sampling).
    """
    if sampling_steps >= total_steps:
        return list(range(total_steps - 1, -1, -1))
    seq = np.linspace(0, total_steps - 1, sampling_steps, dtype=np.int64)
    return [int(t) for t in seq[::-1]]


@torch.no_grad()
def ddrm_sample_patch(
    model: nn.Module,
    diffusion: GaussianDiffusionPrior,
    svd: PerFreqOperatorSVD,
    ddrm_cfg: DDRMConfig,
    sampling_steps: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Run DDRM reverse sampling (Kawar et al. 2022, Eq. 8) for a single 512 x 512
    complex STFT patch using per-frequency SVD of H_k = (I~ - G_k)^dagger.

    Coordinate convention:
      - Inside SVD space, time axis is in newest-to-oldest order matching (I~ - G_k).
      - Before calling the U-Net prior f_theta(x_t, t), we flip back to chronological
        order (unpack_dry_vector) so the U-Net always sees standard left-to-right time!
      - After f_theta predicts x_0_hat (chronological), we flip to newest-to-oldest
        before projecting by V_H^H.

    Returns:
        x_dry_chron: Complex tensor (F, m) in chronological time order.
    """
    device = svd.V_H.device
    F_bins, m = svd.y_bar.shape
    steps = ddrm_cfg.sampling_steps if sampling_steps is None else int(sampling_steps)
    eta = float(ddrm_cfg.eta)
    eta_b = float(ddrm_cfg.eta_b)
    sigma_y = float(ddrm_cfg.sigma_y)

    timesteps = _build_ddrm_timesteps(diffusion.num_timesteps, steps)
    sigmas = diffusion.ddrm_sigmas.to(device)
    sqrt_ab = diffusion.sqrt_alphas_cumprod.to(device)

    V_H = svd.V_H  # (F, m, m)
    V_H_conj_T = V_H.conj().transpose(-1, -2)  # (F, m, m) = V_H^H
    s_H = svd.s_H  # (F, m)
    y_bar = svd.y_bar  # (F, m)

    # Effective per-coordinate observation noise: sigma_obs = sigma_y / s_H
    nonzero_mask = s_H > 0.0
    sigma_obs = torch.where(nonzero_mask, sigma_y / torch.clamp(s_H, min=1e-12), torch.full_like(s_H, float("inf")))

    def _randn_complex(shape: Tuple[int, ...]) -> torch.Tensor:
        # Each real & imag channel has variance 1.0 to match 2-channel real DDPM training
        # (Unitary V_H preserves i.i.d. N(0, 1) across real & imag channels.)
        re = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        im = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        return torch.complex(re, im)

    # Initialization at t = timesteps[0] (DDRM Eq. 8 initialization)
    t_init = timesteps[0]
    sigma_init = sigmas[t_init]
    init_noise = _randn_complex((F_bins, m))

    # For s_H > 0: x_bar_T ~ N(y_bar, max(sigma_T^2 - sigma_obs^2, 0) I)
    # For s_H == 0: x_bar_T ~ N(0, sigma_T^2 I)
    init_std_obs = torch.sqrt(torch.clamp(sigma_init * sigma_init - sigma_obs * sigma_obs, min=0.0))
    x_bar = torch.where(
        nonzero_mask,
        y_bar + init_std_obs.to(y_bar.dtype) * init_noise,
        sigma_init.to(y_bar.dtype) * init_noise,
    )
    # Map spectral state back to signal space (newest-to-oldest order)
    x_ve = torch.bmm(V_H, x_bar.unsqueeze(-1)).squeeze(-1)  # (F, m)

    x0_chron = unpack_dry_vector(x_ve)
    model.eval()

    for idx, t_curr in enumerate(timesteps):
        t_next = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
        sigma_curr = sigmas[t_curr]
        sigma_next = sigmas[t_next] if t_next >= 0 else torch.tensor(0.0, device=device)

        # Convert VE state -> VP input in chronological order for U-Net prior
        x_vp_chron = unpack_dry_vector(x_ve) * sqrt_ab[t_curr]
        x_in = complex_to_2ch(x_vp_chron).unsqueeze(0)  # (1, 2, F, m)
        t_tensor = torch.tensor([t_curr], device=device, dtype=torch.long)

        # U-Net predicts clean dry x_0 directly (Eq. 9)
        x0_pred_2ch = model(x_in, t_tensor).squeeze(0)  # (2, F, m)
        x0_chron = ch2_to_complex(x0_pred_2ch)

        if t_next < 0:
            # Final step (sigma_next = 0): apply noiseless data-consistency projection if sigma_obs == 0,
            # or exact Eq. 8 limit as sigma_next -> 0:
            # Case B (0 < sigma_obs): x_bar_0 = x_0_bar
            # Case C (0 >= sigma_obs): (1 - eta_b) x_0_bar + eta_b y_bar
            x0_vec = torch.flip(x0_chron, dims=[-1])
            x0_bar = torch.bmm(V_H_conj_T, x0_vec.unsqueeze(-1)).squeeze(-1)
            use_c = nonzero_mask & (sigma_obs <= 1e-12)
            x_bar = torch.where(use_c, (1.0 - eta_b) * x0_bar + eta_b * y_bar, x0_bar)
            x_ve = torch.bmm(V_H, x_bar.unsqueeze(-1)).squeeze(-1)
            x0_chron = unpack_dry_vector(x_ve)
            break

        # Project x_0_hat and x_t into V_H spectral coordinates
        x0_vec = torch.flip(x0_chron, dims=[-1])
        x0_bar = torch.bmm(V_H_conj_T, x0_vec.unsqueeze(-1)).squeeze(-1)
        x_t_bar = torch.bmm(V_H_conj_T, x_ve.unsqueeze(-1)).squeeze(-1)

        eps = _randn_complex((F_bins, m))

        # Exact Kawar et al. (2022) Eq. 8 three cases:
        # Case A: s_H == 0
        sqrt_1_m_eta2 = math.sqrt(max(0.0, 1.0 - eta * eta))
        case_a = (
            x0_bar
            + sqrt_1_m_eta2 * (sigma_next / torch.clamp(sigma_curr, min=1e-12)) * (x_t_bar - x0_bar)
            + eta * sigma_next * eps
        )

        # Case B: s_H > 0 and sigma_next < sigma_obs (= sigma_y / s_H)
        case_b = (
            x0_bar
            + sqrt_1_m_eta2 * (sigma_next / torch.clamp(sigma_obs, min=1e-12)) * (y_bar - x0_bar)
            + eta * sigma_next * eps
        )

        # Case C: s_H > 0 and sigma_next >= sigma_obs
        c_std = torch.sqrt(torch.clamp(sigma_next * sigma_next - (eta_b * sigma_obs) ** 2, min=0.0))
        case_c = (1.0 - eta_b) * x0_bar + eta_b * y_bar + c_std.to(y_bar.dtype) * eps

        is_case_b = nonzero_mask & (sigma_next < sigma_obs)
        is_case_c = nonzero_mask & (sigma_next >= sigma_obs)

        x_next_bar = torch.where(is_case_b, case_b, torch.where(is_case_c, case_c, case_a))
        x_ve = torch.bmm(V_H, x_next_bar.unsqueeze(-1)).squeeze(-1)

    return x0_chron


def run_algorithm1_patch(
    y_full_spec: torch.Tensor,
    start_frame: int,
    model: nn.Module,
    diffusion: GaussianDiffusionPrior,
    cfg: DereverbPipelineConfig,
    g_init_override: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Execute Algorithm 1 (Saito et al., ICASSP 2023) on a single 512-frame STFT patch.

    Steps:
      1. Estimate per-frequency WPE filter g_k (L=150, D=8, 1 iter) — or use override
         (e.g., oracle g_k* during evaluation).
      2. Compute SVD of (I~ - G_k) and run DDRM (Eq. 8) -> x_bar_dry.
      3. Refine g_k for N_refine iterations via autograd (unless skip_refinement=True).
      4. Recompute SVD of (I~ - G_k_refined) and run DDRM -> x_tilde_dry.
    """
    m = cfg.stft.patch_frames
    D = cfg.wpe.delay
    L = cfg.wpe.taps
    ddrm_steps = cfg.refinement.fast_ddrm_steps if cfg.fast else cfg.ddrm.sampling_steps
    n_refine = cfg.refinement.fast_n_refine if cfg.fast else cfg.refinement.n_refine
    if cfg.skip_refinement:
        n_refine = 0

    timings: Dict[str, float] = {}

    # Extract causal wet observation vector y_vec (F, m + D + L - 1)
    y_vec = pack_wet_vector(y_full_spec, start_frame=start_frame, m=m, delay=D, taps=L)

    # Step 1: Per-frequency WPE filter estimation
    t0 = time.perf_counter()
    if g_init_override is not None:
        g_k = g_init_override.to(y_full_spec.device)
    else:
        # Estimate WPE filter on this window (plus left context if available)
        ctx_start = max(0, start_frame - (D + L - 1))
        ctx_end = min(y_full_spec.shape[1], start_frame + m)
        y_wpe_slice = y_full_spec[:, ctx_start:ctx_end]
        if y_wpe_slice.shape[1] < m:
            y_wpe_slice = torch.nn.functional.pad(y_wpe_slice, (0, m - y_wpe_slice.shape[1]))
        g_k, _ = estimate_wpe_filter(y_wpe_slice, cfg.wpe)
    timings["wpe_init_sec"] = round(time.perf_counter() - t0, 3)

    # Step 2: SVD + First DDRM pass -> x_bar_dry
    t1 = time.perf_counter()
    svd1 = compute_operator_svd(
        g_k,
        y_vec,
        m=m,
        delay=D,
        svd_eps=cfg.ddrm.svd_eps,
        freq_chunk=cfg.ddrm.svd_freq_chunk,
    )
    timings["svd1_sec"] = round(time.perf_counter() - t1, 3)

    t2 = time.perf_counter()
    x_bar_dry = ddrm_sample_patch(
        model=model,
        diffusion=diffusion,
        svd=svd1,
        ddrm_cfg=cfg.ddrm,
        sampling_steps=ddrm_steps,
        generator=generator,
    )
    timings["ddrm1_sec"] = round(time.perf_counter() - t2, 3)

    if n_refine <= 0:
        return x_bar_dry, {"timings": timings, "refinement": {"iters": 0}}

    # Step 3: Autograd refinement of g_k against x_bar_dry
    t3 = time.perf_counter()
    g_refined, ref_stats = refine_wpe_filter(
        g_init=g_k,
        x_bar_dry_chron=x_bar_dry,
        y_vec=y_vec,
        m=m,
        delay=D,
        cfg=cfg.refinement,
        n_iters=n_refine,
    )
    timings["refine_sec"] = round(time.perf_counter() - t3, 3)

    # Step 4: SVD + Second DDRM pass with g_refined -> x_tilde_dry
    t4 = time.perf_counter()
    svd2 = compute_operator_svd(
        g_refined,
        y_vec,
        m=m,
        delay=D,
        svd_eps=cfg.ddrm.svd_eps,
        freq_chunk=cfg.ddrm.svd_freq_chunk,
    )
    timings["svd2_sec"] = round(time.perf_counter() - t4, 3)

    t5 = time.perf_counter()
    x_tilde_dry = ddrm_sample_patch(
        model=model,
        diffusion=diffusion,
        svd=svd2,
        ddrm_cfg=cfg.ddrm,
        sampling_steps=ddrm_steps,
        generator=generator,
    )
    timings["ddrm2_sec"] = round(time.perf_counter() - t5, 3)

    return x_tilde_dry, {"timings": timings, "refinement": ref_stats}


def dereverb_waveform(
    wet_mono: np.ndarray,
    sample_rate: int,
    cfg: DereverbPipelineConfig,
    model: Optional[nn.Module] = None,
    diffusion: Optional[GaussianDiffusionPrior] = None,
    device: Optional[torch.device] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    End-to-end dereverberation for arbitrary-length mono audio.

    Handles:
      - Automatic GPU selection with explicit warning if running on CPU
      - Resampling to cfg.stft.sample_rate if needed (and back to input sample_rate)
      - Overlapping 512-frame STFT windows with raised-cosine time-domain crossfades
        ASSUMPTION: For files longer than 512 STFT frames (~2.97 s @ 44.1 kHz), we
        estimate and refine a fresh per-window WPE filter g_k using left-context frames
        from the full wet STFT, then crossfade reconstructed windows in the time domain.
      - Baseline WPE-only mode (L=200, D=8, 3 iterations) when cfg.baseline_wpe_only=True
      - Mandatory safety post-processing: dry/wet blend (10-20%), BS.1770 loudness match
        within +/- 0.5 LU, and -1.5 dBTP true-peak limiter.
    """
    t_total_start = time.perf_counter()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cpu_warning = None
    if device.type != "cuda":
        cpu_warning = (
            "WARNING: Running on CPU (no CUDA GPU detected). Diffusion + per-frequency SVD "
            "inference will be significantly slower than GPU; consider --fast or --baseline-wpe."
        )
        warnings.warn(cpu_warning, RuntimeWarning)

    wet_in = np.asarray(wet_mono, dtype=np.float32)
    if wet_in.ndim > 1:
        wet_in = np.mean(wet_in, axis=-1).astype(np.float32)
    orig_len = len(wet_in)
    audio_duration_sec = orig_len / max(1, sample_rate)

    # Resample to model sample rate if necessary
    target_sr = cfg.stft.sample_rate
    if sample_rate != target_sr:
        from scipy.signal import resample_poly

        gcd = math.gcd(sample_rate, target_sr)
        wet_work = resample_poly(wet_in, target_sr // gcd, sample_rate // gcd).astype(np.float32)
    else:
        wet_work = wet_in

    work_len = len(wet_work)
    wet_tensor = torch.from_numpy(wet_work).to(device)
    y_full_spec = compute_stft(wet_tensor, cfg.stft)  # (512, T_frames) complex
    total_frames = y_full_spec.shape[1]

    window_reports: List[Dict] = []

    if cfg.baseline_wpe_only:
        if progress_cb:
            progress_cb(0.2, "Running baseline WPE (L=200, D=8, 3 iters)...")
        wpe_base_cfg = WPEConfig.baseline()
        _, x_hat_spec = estimate_wpe_filter(y_full_spec, wpe_base_cfg)
        dry_work = compute_istft(x_hat_spec, cfg.stft, length=work_len).detach().cpu().numpy()
        mode_name = "baseline_wpe_only"
    else:
        if model is None or diffusion is None:
            raise ValueError(
                "A trained diffusion model and GaussianDiffusionPrior are required unless "
                "baseline_wpe_only=True."
            )
        model = model.to(device).eval()
        diffusion = diffusion.to(device)

        slices = iter_overlapping_frame_slices(
            total_frames=total_frames,
            patch_frames=cfg.stft.patch_frames,
            hop_frames=cfg.stft.window_hop_frames,
        )
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.seed)

        reconstructed_chunks: List[Tuple[int, np.ndarray]] = []
        for idx, (f_start, f_end) in enumerate(slices):
            if progress_cb:
                pct = 0.05 + 0.85 * (idx / max(1, len(slices)))
                progress_cb(pct, f"Processing window {idx + 1}/{len(slices)} (Algorithm 1)...")

            x_patch_spec, rep = run_algorithm1_patch(
                y_full_spec=y_full_spec,
                start_frame=f_start,
                model=model,
                diffusion=diffusion,
                cfg=cfg,
                generator=gen,
            )
            window_reports.append(rep)

            valid_frames = f_end - f_start
            x_valid_spec = x_patch_spec[:, :valid_frames]
            chunk_samples = min(
                work_len - f_start * cfg.stft.hop_length,
                (valid_frames - 1) * cfg.stft.hop_length + cfg.stft.win_length,
            )
            chunk_wav = (
                compute_istft(x_valid_spec, cfg.stft, length=max(1, chunk_samples))
                .detach()
                .cpu()
                .numpy()
            )
            reconstructed_chunks.append((f_start * cfg.stft.hop_length, chunk_wav))

        dry_work = overlap_add_waveforms(reconstructed_chunks, total_samples=work_len)
        mode_name = (
            "proposed_no_refinement"
            if cfg.skip_refinement
            else ("proposed_fast" if cfg.fast else "proposed_full")
        )

    # Resample back to input sample rate if resampled
    if sample_rate != target_sr:
        from scipy.signal import resample_poly

        gcd = math.gcd(sample_rate, target_sr)
        dry_out_raw = resample_poly(dry_work, sample_rate // gcd, target_sr // gcd).astype(np.float32)
        if len(dry_out_raw) < orig_len:
            dry_out_raw = np.pad(dry_out_raw, (0, orig_len - len(dry_out_raw)))
        else:
            dry_out_raw = dry_out_raw[:orig_len]
    else:
        dry_out_raw = dry_work[:orig_len]

    if progress_cb:
        progress_cb(0.94, "Matching loudness (+/- 0.5 LU) & applying -1.5 dBTP limiter...")

    final_out, post_metrics = postprocess_dereverb_output(
        dry_est=dry_out_raw,
        wet_ref=wet_in,
        sample_rate=sample_rate,
        cfg=cfg.post,
    )

    elapsed_sec = time.perf_counter() - t_total_start
    sec_per_min_audio = elapsed_sec / max(audio_duration_sec / 60.0, 1e-6)

    if progress_cb:
        progress_cb(1.0, "Complete")

    report = {
        "mode": mode_name,
        "device": str(device),
        "cpu_warning": cpu_warning,
        "audio_duration_sec": round(audio_duration_sec, 3),
        "sample_rate": sample_rate,
        "model_sample_rate": target_sr,
        "num_windows": len(window_reports) if window_reports else 1,
        "elapsed_sec": round(elapsed_sec, 3),
        "sec_per_minute_audio": round(sec_per_min_audio, 2),
        "postprocess": post_metrics,
        "windows": window_reports,
    }
    return final_out, report
