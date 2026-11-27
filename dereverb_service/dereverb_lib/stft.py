"""
Complex STFT pipeline, overlapping long-file windowing, ITU-R BS.1770 loudness
matching (+/- 0.5 LU), 4x oversampled -1.5 dBTP true-peak limiter, and dry/wet
blending.

Paper specification (Saito et al., ICASSP 2023):
  - Complex STFT, window 1024, hop 256, Hann window, remove DC component
  - 2 channels (real, imaginary) of size 512 x 512
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import PostProcessConfig, STFTConfig


def compute_stft(
    waveform: torch.Tensor,
    cfg: STFTConfig,
) -> torch.Tensor:
    """
    Compute complex STFT of a 1D mono waveform (T_samples,) or batched (B, T_samples).
    Returns complex tensor of shape (..., F, T_frames) where F = 512 when drop_dc=True.
    """
    squeeze_batch = waveform.ndim == 1
    if squeeze_batch:
        waveform = waveform.unsqueeze(0)

    window = torch.hann_window(cfg.win_length, device=waveform.device, dtype=waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=cfg.center,
        normalized=cfg.normalized,
        return_complex=True,
    )
    if cfg.drop_dc:
        spec = spec[:, 1:, :]  # Drop k=0 DC bin -> 512 bins (k=1..512)
    if cfg.stft_scale != 1.0:
        spec = spec * cfg.stft_scale
    return spec.squeeze(0) if squeeze_batch else spec


def compute_istft(
    spec: torch.Tensor,
    cfg: STFTConfig,
    length: Optional[int] = None,
) -> torch.Tensor:
    """
    Inverse complex STFT restoring a zero DC bin when drop_dc=True.
    Input shape: (F, T_frames) or (B, F, T_frames).
    """
    squeeze_batch = spec.ndim == 2
    if squeeze_batch:
        spec = spec.unsqueeze(0)

    if cfg.stft_scale != 1.0:
        spec = spec / cfg.stft_scale

    if cfg.drop_dc:
        dc = torch.zeros(
            (spec.shape[0], 1, spec.shape[2]),
            dtype=spec.dtype,
            device=spec.device,
        )
        spec_full = torch.cat([dc, spec], dim=1)  # Restore 513 bins
    else:
        spec_full = spec

    real_dtype = torch.float32 if spec.dtype == torch.complex64 else torch.float64
    window = torch.hann_window(cfg.win_length, device=spec.device, dtype=real_dtype)
    wav = torch.istft(
        spec_full,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=cfg.center,
        normalized=cfg.normalized,
        length=length,
    )
    return wav.squeeze(0) if squeeze_batch else wav


def complex_to_2ch(spec: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor (..., F, T) to real 2-channel tensor (..., 2, F, T)."""
    return torch.stack([spec.real, spec.imag], dim=-3)


def ch2_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert real 2-channel tensor (..., 2, F, T) to complex tensor (..., F, T)."""
    return torch.complex(x[..., 0, :, :].float(), x[..., 1, :, :].float())


def iter_overlapping_frame_slices(
    total_frames: int,
    patch_frames: int,
    hop_frames: int,
) -> List[Tuple[int, int]]:
    """
    Generate overlapping [start, end) frame windows covering [0, total_frames).
    ASSUMPTION: If total_frames <= patch_frames, returns a single padded window [0, patch_frames).
    For longer files, slides by hop_frames (default 256 = 50% overlap) and snaps the
    final window to end at total_frames so no trailing audio is lost.
    """
    if total_frames <= patch_frames:
        return [(0, total_frames)]

    starts = list(range(0, total_frames - patch_frames + 1, hop_frames))
    if starts[-1] + patch_frames < total_frames:
        starts.append(total_frames - patch_frames)
    return [(s, s + patch_frames) for s in starts]


def overlap_add_waveforms(
    chunks: List[Tuple[int, np.ndarray]],
    total_samples: int,
) -> np.ndarray:
    """
    Overlap-add dereverbed time-domain chunks with smooth raised-cosine crossfades.
    ASSUMPTION: Crossfading reconstructed time-domain chunks avoids phase discontinuities
    that can occur when stitching independent per-window WPE/DDRM spectral estimates.
    """
    if len(chunks) == 1:
        start, wav = chunks[0]
        out = np.zeros(total_samples, dtype=np.float64)
        end = min(total_samples, start + len(wav))
        out[start:end] = wav[: end - start]
        return out.astype(np.float32)

    acc = np.zeros(total_samples, dtype=np.float64)
    weight = np.zeros(total_samples, dtype=np.float64)

    for idx, (start, wav) in enumerate(chunks):
        n = len(wav)
        end = min(total_samples, start + n)
        valid_n = end - start
        if valid_n <= 0:
            continue

        # Raised-cosine fade on internal overlap boundaries; flat at file start/end
        w = np.ones(valid_n, dtype=np.float64)
        fade_len = min(valid_n // 3, int(0.35 * valid_n))
        if idx > 0 and fade_len > 1:
            t = np.linspace(0.0, math.pi, fade_len, endpoint=False)
            w[:fade_len] *= 0.5 * (1.0 - np.cos(t))
        if idx < len(chunks) - 1 and fade_len > 1:
            t = np.linspace(0.0, math.pi, fade_len, endpoint=False)
            w[-fade_len:] *= 0.5 * (1.0 + np.cos(t))

        acc[start:end] += wav[:valid_n] * w
        weight[start:end] += w

    safe_w = np.maximum(weight, 1e-8)
    return (acc / safe_w).astype(np.float32)


# ---------------------------------------------------------------------------
# ITU-R BS.1770-4 Loudness Measurement, Loudness Matching & True-Peak Limiter
# ---------------------------------------------------------------------------

def _k_weighting_filter(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    ITU-R BS.1770-4 K-weighting pre-filter (high-shelf + RLB high-pass).
    Uses pyloudnorm if available, otherwise falls back to scipy.signal bilinear
    filter coefficients matching BS.1770-4 for arbitrary sample rates.
    """
    try:
        from scipy.signal import bilinear, lfilter
    except ImportError:
        return audio.astype(np.float64)

    x = audio.astype(np.float64)

    # Stage 1: Acoustic head-effect high-shelf (+4 dB above ~1.68 kHz)
    f0 = 1681.974450955533
    G = 3.999843853973347
    Q = 0.7071752369554196
    K = math.tan(math.pi * f0 / sample_rate)
    Vh = math.pow(10.0, G / 20.0)
    Vb = math.pow(Vh, 0.4996667741545416)
    a0 = 1.0 + K / Q + K * K
    b_hs = np.array([
        (Vh + Vb * K / Q + K * K) / a0,
        2.0 * (K * K - Vh) / a0,
        (Vh - Vb * K / Q + K * K) / a0,
    ])
    a_hs = np.array([
        1.0,
        2.0 * (K * K - 1.0) / a0,
        (1.0 - K / Q + K * K) / a0,
    ])
    x = lfilter(b_hs, a_hs, x)

    # Stage 2: RLB 2nd-order high-pass filter (~38.14 Hz)
    f0_hp = 38.13547087602444
    Q_hp = 0.5003270373238773
    K_hp = math.tan(math.pi * f0_hp / sample_rate)
    a0_hp = 1.0 + K_hp / Q_hp + K_hp * K_hp
    b_hp = np.array([1.0, -2.0, 1.0]) / a0_hp
    a_hp = np.array([
        1.0,
        2.0 * (K_hp * K_hp - 1.0) / a0_hp,
        (1.0 - K_hp / Q_hp + K_hp * K_hp) / a0_hp,
    ])
    x = lfilter(b_hp, a_hp, x)
    return x


def measure_integrated_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """Measure integrated loudness (LUFS) per ITU-R BS.1770-4."""
    try:
        import pyloudnorm as pyln  # type: ignore

        meter = pyln.Meter(sample_rate)
        if audio.ndim == 1 and len(audio) < int(0.4 * sample_rate):
            pad = np.zeros(int(0.45 * sample_rate), dtype=np.float64)
            pad[: len(audio)] = audio
            return float(meter.integrated_loudness(pad))
        return float(meter.integrated_loudness(audio.astype(np.float64)))
    except Exception:
        pass

    # Pure NumPy/SciPy ITU-R BS.1770-4 gated loudness fallback
    mono = audio if audio.ndim == 1 else np.mean(audio, axis=-1)
    if len(mono) == 0:
        return -70.0
    y = _k_weighting_filter(mono, sample_rate)
    block_len = max(1, int(round(0.4 * sample_rate)))
    step = max(1, int(round(0.1 * sample_rate)))
    if len(y) < block_len:
        ms = float(np.mean(y * y))
        return float(-0.691 + 10.0 * math.log10(max(ms, 1e-12)))

    energies = []
    for start in range(0, len(y) - block_len + 1, step):
        seg = y[start : start + block_len]
        energies.append(float(np.mean(seg * seg)))
    energies_arr = np.asarray(energies, dtype=np.float64)
    lufs_blocks = -0.691 + 10.0 * np.log10(np.maximum(energies_arr, 1e-12))

    # Absolute gate (-70 LUFS)
    abs_mask = lufs_blocks > -70.0
    if not np.any(abs_mask):
        return -70.0
    mean_abs = float(np.mean(energies_arr[abs_mask]))
    rel_thresh = -0.691 + 10.0 * math.log10(max(mean_abs, 1e-12)) - 10.0

    # Relative gate (-10 LU)
    rel_mask = abs_mask & (lufs_blocks > rel_thresh)
    if not np.any(rel_mask):
        return float(-0.691 + 10.0 * math.log10(max(mean_abs, 1e-12)))
    gated_mean = float(np.mean(energies_arr[rel_mask]))
    return float(-0.691 + 10.0 * math.log10(max(gated_mean, 1e-12)))


def measure_true_peak_dbtp(audio: np.ndarray, oversample: int = 4) -> float:
    """Estimate inter-sample true peak in dBTP using polyphase / Catmull-Rom 4x oversampling."""
    x = np.asarray(audio, dtype=np.float64)
    if x.size == 0:
        return -100.0
    peak = float(np.max(np.abs(x)))
    if oversample > 1 and x.shape[0] >= 4:
        try:
            from scipy.signal import resample_poly

            up = resample_poly(x, oversample, 1, axis=0)
            peak = max(peak, float(np.max(np.abs(up))))
        except Exception:
            # 4x cubic Catmull-Rom fallback
            y0, y1, y2, y3 = x[:-3], x[1:-2], x[2:-1], x[3:]
            for t in (0.25, 0.5, 0.75):
                t2 = t * t
                t3 = t2 * t
                interp = 0.5 * (
                    2.0 * y1
                    + (-y0 + y2) * t
                    + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2
                    + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3
                )
                peak = max(peak, float(np.max(np.abs(interp))))
    return float(20.0 * math.log10(max(peak, 1e-10)))


def apply_true_peak_limiter(
    audio: np.ndarray,
    sample_rate: int,
    ceiling_dbtp: float = -1.5,
    oversample: int = 4,
) -> Tuple[np.ndarray, float]:
    """
    Smooth lookahead true-peak limiter guaranteeing output <= ceiling_dbtp.
    Returns (limited_audio, max_gain_reduction_db).
    """
    x = np.asarray(audio, dtype=np.float64).copy()
    if x.size == 0:
        return x.astype(np.float32), 0.0

    ceiling_lin = 10.0 ** (ceiling_dbtp / 20.0)
    n = x.shape[0]
    abs_env = np.abs(x) if x.ndim == 1 else np.max(np.abs(x), axis=-1)

    # 4x oversampled envelope detection
    if oversample > 1 and n >= 4:
        try:
            from scipy.signal import resample_poly

            up = resample_poly(x, oversample, 1, axis=0)
            up_abs = np.abs(up) if up.ndim == 1 else np.max(np.abs(up), axis=-1)
            pad_len = (-len(up_abs)) % oversample
            if pad_len:
                up_abs = np.pad(up_abs, (0, pad_len))
            peak_env = np.max(up_abs.reshape(-1, oversample), axis=1)[:n]
            peak_env = np.maximum(peak_env, abs_env)
        except Exception:
            peak_env = abs_env.copy()
    else:
        peak_env = abs_env.copy()

    # 5 ms lookahead max window
    lookahead = max(1, int(round(0.005 * sample_rate)))
    try:
        from scipy.ndimage import maximum_filter1d

        # Causal lookahead: max over [i, i + lookahead]
        padded = np.pad(peak_env, (0, lookahead), mode="edge")
        win_peak = maximum_filter1d(padded, size=lookahead + 1, origin=-(lookahead // 2))[:n]
    except Exception:
        win_peak = peak_env.copy()
        for k in range(1, min(lookahead + 1, n)):
            win_peak[:-k] = np.maximum(win_peak[:-k], peak_env[k:])

    desired_gain = np.where(win_peak > ceiling_lin, ceiling_lin / np.maximum(win_peak, 1e-12), 1.0)

    # Smooth attack (1.5 ms) & release (65 ms)
    attack_c = math.exp(-1.0 / max(1.0, 0.0015 * sample_rate))
    release_c = math.exp(-1.0 / max(1.0, 0.065 * sample_rate))
    gain = np.ones(n, dtype=np.float64)
    g = 1.0
    for i in range(n):
        target = desired_gain[i]
        if target < g:
            g = min(target, attack_c * g + (1.0 - attack_c) * target)
        else:
            g = release_c * g + (1.0 - release_c) * target
        gain[i] = g

    out = x * gain if x.ndim == 1 else x * gain[:, None]
    # Hard safety trim if inter-sample peak still slightly exceeds ceiling
    tp_after = measure_true_peak_dbtp(out, oversample=oversample)
    if tp_after > ceiling_dbtp + 0.02:
        trim = 10.0 ** ((ceiling_dbtp - tp_after) / 20.0)
        out = out * trim
        min_gain = float(np.min(gain)) * trim
    else:
        min_gain = float(np.min(gain))

    max_gr_db = max(0.0, -20.0 * math.log10(max(min_gain, 1e-8)))
    return out.astype(np.float32), max_gr_db


def postprocess_dereverb_output(
    dry_est: np.ndarray,
    wet_ref: np.ndarray,
    sample_rate: int,
    cfg: PostProcessConfig,
) -> Tuple[np.ndarray, dict]:
    """
    Apply safety post-processing:
      1. Optional dry/wet blend (default 15% original wet signal to prevent robotic artifacts)
      2. Loudness match to input within +/- 0.5 LU (ITU-R BS.1770)
      3. -1.5 dBTP true-peak limiter (with 2nd pass check to keep LUFS within +/- 0.5 LU)
    """
    min_len = min(len(dry_est), len(wet_ref))
    dry_est = np.asarray(dry_est[:min_len], dtype=np.float64)
    wet_ref = np.asarray(wet_ref[:min_len], dtype=np.float64)

    dry_mix = float(np.clip(cfg.dry_mix, 0.0, 0.5))
    blended = (1.0 - dry_mix) * dry_est + dry_mix * wet_ref

    in_lufs = measure_integrated_lufs(wet_ref, sample_rate)
    out_lufs_raw = measure_integrated_lufs(blended, sample_rate)

    makeup_db = 0.0
    if cfg.match_loudness and in_lufs > -65.0 and out_lufs_raw > -65.0:
        makeup_db = float(np.clip(in_lufs - out_lufs_raw, -18.0, 18.0))
        blended = blended * (10.0 ** (makeup_db / 20.0))

    limited, max_gr_db = apply_true_peak_limiter(
        blended,
        sample_rate=sample_rate,
        ceiling_dbtp=cfg.true_peak_ceiling_dbtp,
        oversample=cfg.oversample_factor,
    )

    # Second pass trim if limiter pulled LUFS slightly outside +/- tolerance
    out_lufs = measure_integrated_lufs(limited, sample_rate)
    if (
        cfg.match_loudness
        and in_lufs > -65.0
        and out_lufs > -65.0
        and abs(out_lufs - in_lufs) > cfg.loudness_tolerance_lu
    ):
        correction_db = float(np.clip(in_lufs - out_lufs, -4.0, 4.0))
        limited = limited * (10.0 ** (correction_db / 20.0))
        limited, gr2 = apply_true_peak_limiter(
            limited,
            sample_rate=sample_rate,
            ceiling_dbtp=cfg.true_peak_ceiling_dbtp,
            oversample=cfg.oversample_factor,
        )
        max_gr_db = max(max_gr_db, gr2)
        makeup_db += correction_db
        out_lufs = measure_integrated_lufs(limited, sample_rate)

    in_tp = measure_true_peak_dbtp(wet_ref, oversample=cfg.oversample_factor)
    out_tp = measure_true_peak_dbtp(limited, oversample=cfg.oversample_factor)

    metrics = {
        "input_lufs": round(in_lufs, 2),
        "output_lufs": round(out_lufs, 2),
        "loudness_delta_lu": round(out_lufs - in_lufs, 2),
        "input_true_peak_dbtp": round(in_tp, 2),
        "output_true_peak_dbtp": round(out_tp, 2),
        "makeup_gain_db": round(makeup_db, 2),
        "limiter_max_gr_db": round(max_gr_db, 2),
        "dry_mix": round(dry_mix, 3),
    }
    return limited.astype(np.float32), metrics
