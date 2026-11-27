"""
Per-frequency Weighted Prediction Error (WPE) dereverberation filter estimation.

Implements single-channel per-frequency WPE in pure PyTorch (GPU/CPU) so we can
directly extract the exact complex filter taps g_k of shape (F, L) needed by
Algorithm 1 of Saito et al. (ICASSP 2023):
  - Proposed Algorithm 1 init: L = 150, D = 8, 1 iteration
  - Paper baseline (WPE alone): L = 200, D = 8, 3 iterations

Signal model per frequency bin k:
    x_k[t] = y_k[t] - sum_{l=1}^{L} g_{l,k}^* y_k[t - D - l + 1]
           = y_k[t] - g_k^H y_tilde_k[t]
"""

from __future__ import annotations

from typing import Tuple

import torch

from .config import WPEConfig


def _build_delayed_matrix(
    y: torch.Tensor,
    taps: int,
    delay: int,
) -> torch.Tensor:
    """
    Build delayed observation buffer Y_tilde of shape (F, T, L) where
    Y_tilde[k, t, l] = y[k, t - delay - l] (zero-padded for negative time indices).
    """
    F, T = y.shape
    pad_len = delay + taps - 1
    y_padded = torch.nn.functional.pad(y, (pad_len, 0))  # (F, T + D + L - 1)
    # Using unfold on time axis: windows of length `taps` ending at index t - delay
    # For each t in 0..T-1, y_padded index for frame t is (pad_len + t).
    # We want indices: (pad_len + t - delay), (pad_len + t - delay - 1), ..., (pad_len + t - delay - taps + 1)
    # That is the reverse of slice [t : t + taps] in y_padded!
    # Check: when t=0, slice [0 : taps] has indices 0..taps-1; reversed -> taps-1..0 = pad_len-delay .. pad_len-delay-taps+1.
    windows = y_padded[:, : T + taps - 1].unfold(dimension=1, size=taps, step=1)  # (F, T, L)
    return torch.flip(windows, dims=[-1]).contiguous()


def estimate_wpe_filter(
    y_spec: torch.Tensor,
    cfg: WPEConfig,
    freq_chunk: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate per-frequency complex WPE filter taps g_k and dereverbed STFT x_hat.

    Args:
        y_spec: Complex STFT of wet signal, shape (F, T).
        cfg: WPEConfig (taps L, delay D, iterations).
        freq_chunk: Chunk size along frequency axis to bound GPU memory during
                    (L x L) normal equation solves.

    Returns:
        g_k: Complex filter taps of shape (F, L) in complex64/complex128 matching y_spec.
        x_hat: Dereverbed complex STFT of shape (F, T).
    """
    if not torch.is_complex(y_spec):
        raise ValueError("y_spec must be a complex tensor of shape (F, T).")
    if y_spec.ndim != 2:
        raise ValueError(f"Expected 2D (F, T) spectrogram, got shape {tuple(y_spec.shape)}.")

    device = y_spec.device
    orig_dtype = y_spec.dtype
    # Solve normal equations in complex128 for conditioning (especially at L=150..200)
    # ASSUMPTION: Double precision solve avoids ill-conditioning for long L=150/200 WPE filters.
    y = y_spec.to(torch.complex128)
    F_bins, T_frames = y.shape
    L = cfg.taps
    D = cfg.delay

    g_out = torch.zeros((F_bins, L), dtype=torch.complex128, device=device)
    x_out = y.clone()

    eye_L = torch.eye(L, dtype=torch.complex128, device=device).unsqueeze(0)

    for f0 in range(0, F_bins, freq_chunk):
        f1 = min(F_bins, f0 + freq_chunk)
        y_c = y[f0:f1]  # (Fc, T)
        Y_tilde = _build_delayed_matrix(y_c, taps=L, delay=D)  # (Fc, T, L)
        x_c = y_c.clone()
        g_c = torch.zeros((f1 - f0, L), dtype=torch.complex128, device=device)

        for _ in range(cfg.iterations):
            # Time-varying target variance lambda_k[t] = |x_k[t]|^2
            power = (x_c.real.square() + x_c.imag.square())  # (Fc, T)
            if cfg.psd_context > 0:
                k_size = 2 * cfg.psd_context + 1
                power = torch.nn.functional.avg_pool1d(
                    power.unsqueeze(1),
                    kernel_size=k_size,
                    stride=1,
                    padding=cfg.psd_context,
                ).squeeze(1)

            # Relative floor prevents silent bins from blowing up inverse variance
            bin_floor = torch.clamp(power.mean(dim=-1, keepdim=True) * 1e-6, min=cfg.psd_floor)
            inv_var = 1.0 / torch.clamp(power, min=bin_floor)  # (Fc, T)

            # Weighted autocorrelation: R_k = sum_t inv_var[k,t] * y_tilde[k,t] y_tilde[k,t]^H
            # Y_weighted: (Fc, T, L)
            Y_w = Y_tilde * inv_var.unsqueeze(-1)
            # R: (Fc, L, L) = Y_w^T @ Y_tilde^*  (Wait: y_tilde y_tilde^H has (i,j) = y_i * conj(y_j))
            R = torch.einsum("fti,ftj->fij", Y_w, Y_tilde.conj())
            # Cross-correlation: p_k = sum_t inv_var[k,t] * y_tilde[k,t] * conj(y[k,t])
            # Wait: check convention!
            # We minimize sum_t inv_var * |y[t] - g^H y_tilde[t]|^2
            # = sum_t inv_var * |y[t] - sum_l g_l^* y_tilde_l[t]|^2.
            # Taking d/d(g_l^*) = 0:
            # sum_t inv_var * (-y_tilde_l[t]) * (y[t]^* - sum_j g_j y_tilde_j[t]^*) = 0
            # => (sum_t inv_var * y_tilde_l[t] y_tilde_j[t]^*) g_j = sum_t inv_var * y_tilde_l[t] y[t]^*
            # => R g = p! Exact!
            p = torch.einsum("fti,ft->fi", Y_w, y_c.conj())  # (Fc, L)

            # Trace-scaled ridge regularization for stable solve
            trace_scale = torch.diagonal(R, dim1=-2, dim2=-1).real.mean(dim=-1, keepdim=True)
            ridge = (cfg.ridge_eps * torch.clamp(trace_scale, min=1e-12)).unsqueeze(-1)
            R_reg = R + ridge * eye_L

            try:
                g_c = torch.linalg.solve(R_reg, p.unsqueeze(-1)).squeeze(-1)  # (Fc, L)
            except RuntimeError:
                # Fallback to least-squares if a frequency bin is completely silent
                g_c = torch.linalg.lstsq(R_reg, p.unsqueeze(-1)).solution.squeeze(-1)

            # Update dereverbed estimate: x_k[t] = y_k[t] - sum_l g_{l,k}^* y_tilde[k,t,l]
            reverb_est = torch.einsum("fl,ftl->ft", g_c.conj(), Y_tilde)
            x_c = y_c - reverb_est

        g_out[f0:f1] = g_c
        x_out[f0:f1] = x_c

    return g_out.to(orig_dtype), x_out.to(orig_dtype)
