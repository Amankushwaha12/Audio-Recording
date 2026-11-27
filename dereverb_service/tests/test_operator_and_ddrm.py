"""
Unit & mathematical verification tests for the Saito et al. (ICASSP 2023) prototype.

Run with:
    pytest tests/test_operator_and_ddrm.py -v
"""

from __future__ import annotations

import numpy as np
import torch

from dereverb_lib.config import PostProcessConfig, RefinementConfig, STFTConfig, UNetConfig, WPEConfig
from dereverb_lib.operator import (
    apply_dereverb_operator,
    build_dereverb_matrix,
    compute_operator_svd,
    pack_wet_vector,
    refine_wpe_filter,
    unpack_dry_vector,
)
from dereverb_lib.stft import compute_istft, compute_stft, postprocess_dereverb_output
from dereverb_lib.unet import AudioDiffusionUNet
from dereverb_lib.wpe import estimate_wpe_filter


def test_toeplitz_matrix_matches_fast_operator_and_autograd():
    """Verify (I~ - G_k) y_vec matches fast Hankel operator in both values and Wirtinger gradients."""
    torch.manual_seed(0)
    F_bins, m, D, L = 8, 32, 4, 12
    n = m + D + L - 1

    y_vec = torch.complex(torch.randn(F_bins, n), torch.randn(F_bins, n))
    target = torch.complex(torch.randn(F_bins, m), torch.randn(F_bins, m))

    g1 = torch.complex(torch.randn(F_bins, L) * 0.05, torch.randn(F_bins, L) * 0.05).requires_grad_(True)
    g2 = g1.detach().clone().requires_grad_(True)

    out_explicit = apply_dereverb_operator(g1, y_vec, m=m, delay=D, use_explicit_matrix=True)
    out_fast = apply_dereverb_operator(g2, y_vec, m=m, delay=D, use_explicit_matrix=False)
    assert torch.allclose(out_explicit, out_fast, atol=1e-5, rtol=1e-5)

    loss1 = ((target - out_explicit).abs() ** 2).sum() + (g1.abs() ** 2).sum()
    loss2 = ((target - out_fast).abs() ** 2).sum() + (g2.abs() ** 2).sum()
    loss1.backward()
    loss2.backward()
    assert torch.allclose(g1.grad, g2.grad, atol=1e-4, rtol=1e-4)


def test_svd_pseudoinverse_identity():
    """Verify H_k = A_k^dagger satisfies A_k H_k = I_m and y_bar = V_H^H (A_k y_vec)."""
    torch.manual_seed(1)
    F_bins, m, D, L = 4, 24, 3, 8
    n = m + D + L - 1

    g_k = torch.complex(torch.randn(F_bins, L) * 0.02, torch.randn(F_bins, L) * 0.02)
    y_vec = torch.complex(torch.randn(F_bins, n), torch.randn(F_bins, n))

    A_k = build_dereverb_matrix(g_k, m=m, delay=D)  # (F, m, n)
    svd = compute_operator_svd(g_k, y_vec, m=m, delay=D)

    # Check y_bar == V_H^H (A_k y_vec)
    Ay = torch.bmm(A_k, y_vec.unsqueeze(-1)).squeeze(-1)  # (F, m)
    V_H_h = svd.V_H.conj().transpose(-1, -2)
    y_bar_direct = torch.bmm(V_H_h, Ay.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(svd.y_bar, y_bar_direct, atol=1e-4, rtol=1e-4)

    # Check V_H is unitary: V_H^H V_H = I_m
    eye = torch.eye(m, dtype=torch.complex64).unsqueeze(0).expand(F_bins, m, m)
    assert torch.allclose(torch.bmm(V_H_h, svd.V_H), eye, atol=1e-4)


def test_wpe_and_refinement_decrease_loss():
    """Verify WPE matches Toeplitz operator output and Step 3 refinement strictly reduces objective."""
    torch.manual_seed(2)
    F_bins, T, D, L = 8, 48, 4, 10
    y_spec = torch.complex(torch.randn(F_bins, T), torch.randn(F_bins, T))

    g_k, x_wpe = estimate_wpe_filter(y_spec, WPEConfig(taps=L, delay=D, iterations=1))
    y_vec = pack_wet_vector(y_spec, start_frame=0, m=T, delay=D, taps=L)
    x_op = unpack_dry_vector(apply_dereverb_operator(g_k, y_vec, m=T, delay=D))
    assert torch.allclose(x_wpe, x_op, atol=1e-4, rtol=1e-4)

    x_target = x_wpe * 0.8
    _, stats = refine_wpe_filter(
        g_init=g_k,
        x_bar_dry_chron=x_target,
        y_vec=y_vec,
        m=T,
        delay=D,
        cfg=RefinementConfig(alpha=1e-4, lam=1.0, n_refine=50),
    )
    assert stats["final_loss"] < stats["initial_loss"]


def test_unet_shape_and_parameter_count():
    """Verify modified U-Net accepts (1, 2, 512, 512) and reports ~31M parameters."""
    cfg = UNetConfig()
    model = AudioDiffusionUNet(cfg)
    params = model.count_parameters()
    assert 20_000_000 <= params <= 45_000_000, f"Unexpected parameter count: {params:,}"

    # Fast forward shape check on a smaller 64x64 config to run quickly on CPU
    small_cfg = UNetConfig(
        model_channels=32,
        channel_mult=(1, 2, 2),
        attention_resolutions=(16,),
        input_freq_bins=64,
        norm_groups=16,
    )
    small_model = AudioDiffusionUNet(small_cfg)
    x = torch.randn(1, 2, 64, 64)
    t = torch.tensor([100], dtype=torch.long)
    y = small_model(x, t)
    assert y.shape == x.shape


def test_stft_roundtrip_and_postprocess_safety():
    """Verify complex STFT round-trip and BS.1770 loudness match + -1.5 dBTP limiter."""
    sr = 44100
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    sig = 0.25 * np.sin(2.0 * np.pi * 440.0 * t) + 0.1 * np.sin(2.0 * np.pi * 880.0 * t)

    stft_cfg = STFTConfig(sample_rate=sr)
    spec = compute_stft(torch.from_numpy(sig), stft_cfg)
    assert spec.shape[0] == 512  # DC removed
    rec = compute_istft(spec, stft_cfg, length=len(sig)).numpy()
    assert float(np.mean(np.abs(rec - sig))) < 1e-4

    # Scale dry_est wildly louder and verify loudness match (+/- 0.5 LU) & -1.5 dBTP cap
    out, metrics = postprocess_dereverb_output(
        dry_est=sig * 2.5,
        wet_ref=sig,
        sample_rate=sr,
        cfg=PostProcessConfig(match_loudness=True, true_peak_ceiling_dbtp=-1.5, dry_mix=0.15),
    )
    assert len(out) == len(sig)
    assert abs(metrics["loudness_delta_lu"]) <= 0.55
    assert metrics["output_true_peak_dbtp"] <= -1.45
