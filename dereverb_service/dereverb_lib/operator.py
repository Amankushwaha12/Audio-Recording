"""
Differentiable WPE Toeplitz operator (Eq. 3-6), per-frequency SVD for DDRM, and
Algorithm 1 Step 3 gradient-descent filter refinement.

Paper equations (Saito et al., Sony, ICASSP 2023):
  - Eq. 3: x_hat_dry = (I~ - G_k) y_wet
  - Eq. 4: I~ = [ I_{m x m}, 0_{m x (D + L - 1)} ]
  - Eq. 5: G_k is the m x (m + D + L - 1) Toeplitz matrix built from the
           conjugated filter taps g_{1,k}^*, ..., g_{L,k}^*, shifted by D.
  - Eq. 6: y = (I~ - G_k)^dagger x_hat + z, where A^dagger = A^H (A A^H)^{-1}.
  - Step 3: Repeat N_refine times:
      g_k <- g_k - alpha * grad_{g_k} ( || x_bar_dry - (I~ - G_k) y_wet ||_2^2
                                        + lambda * || g_k ||_2^2 )

Indexing convention so Eq. 3-5 hold literally index-for-index:
  For a target dry patch of m frames covering [t0, t0 + m), stack y_wet in
  reverse chronological order (newest -> oldest) of length n = m + D + L - 1:
      y_vec[r] = y_wet[t0 + m - 1 - r],   for r = 0, ..., n - 1.
  Then:
      (I~ y_vec)[r] = y_vec[r] = y_wet[t]              (where t = t0 + m - 1 - r)
      (G_k y_vec)[r] = sum_{l=1}^{L} g_{l,k}^* y_vec[r + D + l - 1]
                     = sum_{l=1}^{L} g_{l,k}^* y_wet[t - D - l + 1]
  so ((I~ - G_k) y_vec)[r] = x_hat_dry[t], and reversing the m output elements
  restores chronological order [t0, ..., t0 + m - 1]!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

from .config import RefinementConfig


@dataclass
class PerFreqOperatorSVD:
    """
    Precomputed per-frequency SVD of the inverse problem operator
        H_k = (I~ - G_k)^dagger = V_A Sigma_A^{-1} U_A^H = U_H Sigma_H V_H^H
    where A_k = (I~ - G_k) = U_A Sigma_A V_A^H.

    To save memory (512 bins x 512 x 669 would otherwise take ~1.4 GB), we store:
      - V_H (= U_A): shape (F, m, m) complex64
      - s_H (= 1 / Sigma_A): shape (F, m) float32
      - y_bar (= S_H^{-1} U_H^H y_vec = Sigma_A (V_A^H y_vec)): shape (F, m) complex64
    Notice that DDRM Eq. 8 ONLY needs V_H, V_H^H, s_H, and y_bar!
    """

    V_H: torch.Tensor  # (F, m, m) unitary matrix (= U_A)
    s_H: torch.Tensor  # (F, m) positive singular values of H_k = A_k^dagger
    y_bar: torch.Tensor  # (F, m) spectral observation S_H^{-1} U_H^H y_vec
    m: int
    n: int


def pack_wet_vector(
    y_full: torch.Tensor,
    start_frame: int,
    m: int,
    delay: int,
    taps: int,
) -> torch.Tensor:
    """
    Extract wet observation vector y_vec of shape (F, m + D + L - 1) in newest-to-oldest
    order matching Eq. 3-5, using available left-context frames from y_full (zero-padded
    only before frame 0).

    Args:
        y_full: Full wet STFT (F, T_total) complex.
        start_frame: Chronological start frame t0 of the m-frame dry target window.
        m: Target dry patch frames (512).
        delay: Prediction delay D.
        taps: Filter length L.

    Returns:
        y_vec: Complex tensor of shape (F, m + D + L - 1) in newest-to-oldest order.
    """
    F_bins, T_total = y_full.shape
    ctx = delay + taps - 1
    n = m + ctx

    # Chronological indices needed: [start_frame - ctx, start_frame + m)
    chron_start = start_frame - ctx
    chron_end = start_frame + m

    buf = torch.zeros((F_bins, n), dtype=y_full.dtype, device=y_full.device)
    src_start = max(0, chron_start)
    src_end = min(T_total, chron_end)
    if src_end > src_start:
        dst_start = src_start - chron_start
        dst_end = dst_start + (src_end - src_start)
        buf[:, dst_start:dst_end] = y_full[:, src_start:src_end]

    # Reverse to newest-to-oldest so I~ = [I_m, 0] and G_k is upper-shifted Toeplitz
    return torch.flip(buf, dims=[-1]).contiguous()


def unpack_dry_vector(x_vec: torch.Tensor) -> torch.Tensor:
    """Convert newest-to-oldest dry vector (F, m) back to chronological STFT (F, m)."""
    return torch.flip(x_vec, dims=[-1]).contiguous()


def pack_dry_vector(x_chron: torch.Tensor) -> torch.Tensor:
    """Convert chronological dry STFT patch (F, m) to newest-to-oldest order (F, m)."""
    return torch.flip(x_chron, dims=[-1]).contiguous()


def build_dereverb_matrix(
    g_k: torch.Tensor,
    m: int,
    delay: int,
) -> torch.Tensor:
    """
    Differentiably construct A_k = I~ - G_k of shape (F, m, m + D + L - 1) from
    complex filter taps g_k of shape (F, L), matching Eq. 4 & Eq. 5 exactly.

    Gradients flow cleanly through g_k via PyTorch autograd.
    """
    F_bins, L = g_k.shape
    n = m + delay + L - 1
    device = g_k.device
    dtype = g_k.dtype

    # I~ = [ I_{m x m}, 0_{m x (D + L - 1)} ] (Eq. 4)
    I_tilde = torch.zeros((1, m, n), dtype=dtype, device=device)
    diag_idx = torch.arange(m, device=device)
    I_tilde[0, diag_idx, diag_idx] = 1.0

    # G_k (Eq. 5): row r has g_{l,k}^* at column r + delay + (l - 1), for l=1..L
    rows = torch.arange(m, device=device).unsqueeze(1).expand(m, L)  # (m, L)
    taps_idx = torch.arange(L, device=device).unsqueeze(0).expand(m, L)  # (m, L)
    cols = rows + delay + taps_idx  # (m, L), max col = (m-1) + D + (L-1) = n - 1

    G = torch.zeros((F_bins, m, n), dtype=dtype, device=device)
    # Differentiable indexed assignment (out-of-place via scatter on flattened last dim)
    flat_idx = (rows * n + cols).reshape(1, m * L).expand(F_bins, m * L)
    src_vals = g_k.conj().unsqueeze(1).expand(F_bins, m, L).reshape(F_bins, m * L)
    G = G.reshape(F_bins, m * n).scatter(dim=1, index=flat_idx, src=src_vals).reshape(F_bins, m, n)

    return I_tilde - G


def apply_dereverb_operator(
    g_k: torch.Tensor,
    y_vec: torch.Tensor,
    m: int,
    delay: int,
    use_explicit_matrix: bool = False,
) -> torch.Tensor:
    """
    Compute x_hat_vec = (I~ - G_k) y_vec differentiably in g_k.

    Args:
        g_k: Complex filter taps (F, L).
        y_vec: Newest-to-oldest wet frames (F, m + D + L - 1).
        m: Target dry frames (512).
        delay: Prediction delay D.
        use_explicit_matrix: If True, materializes full (F, m, n) matrix A_k and multiplies.
            If False (default), uses mathematically identical O(F * m * L) Hankel unfold
            so 10,000 refinement iterations fit easily in VRAM/RAM.

    Returns:
        x_hat_vec: Complex tensor (F, m) in newest-to-oldest order.
    """
    if use_explicit_matrix:
        A_k = build_dereverb_matrix(g_k, m=m, delay=delay)
        return torch.bmm(A_k, y_vec.unsqueeze(-1)).squeeze(-1)

    L = g_k.shape[-1]
    # y_curr = I~ y_vec -> first m entries
    y_curr = y_vec[:, :m]  # (F, m)
    # For row r in 0..m-1, delayed taps are y_vec[:, r + delay : r + delay + L]
    y_delayed = y_vec[:, delay : delay + m + L - 1].unfold(dimension=1, size=L, step=1)  # (F, m, L)
    reverb = torch.einsum("fl,fml->fm", g_k.conj(), y_delayed)
    return y_curr - reverb


@torch.no_grad()
def compute_operator_svd(
    g_k: torch.Tensor,
    y_vec: torch.Tensor,
    m: int,
    delay: int,
    svd_eps: float = 1e-5,
    freq_chunk: int = 32,
) -> PerFreqOperatorSVD:
    """
    Compute per-frequency SVD of A_k = (I~ - G_k) = U_A diag(s_A) V_A^H
    and derive the exact DDRM degradation operator H_k = A_k^dagger = V_A diag(1/s_A) U_A^H.

    Therefore for H_k = U_H diag(s_H) V_H^H:
      - U_H = V_A   (F, n, m)
      - s_H = 1 / s_A   (F, m)
      - V_H = U_A   (F, m, m)
      - y_bar = S_H^{-1} U_H^H y_vec = s_A * (V_A^H y_vec)   (F, m)
    """
    F_bins, L = g_k.shape
    n = m + delay + L - 1
    device = g_k.device

    V_H_all = torch.empty((F_bins, m, m), dtype=torch.complex64, device=device)
    s_H_all = torch.empty((F_bins, m), dtype=torch.float32, device=device)
    y_bar_all = torch.empty((F_bins, m), dtype=torch.complex64, device=device)

    for f0 in range(0, F_bins, freq_chunk):
        f1 = min(F_bins, f0 + freq_chunk)
        g_c = g_k[f0:f1].to(torch.complex64)
        y_c = y_vec[f0:f1].to(torch.complex64)

        A_c = build_dereverb_matrix(g_c, m=m, delay=delay)  # (Fc, m, n)
        # Thin SVD: U_A (Fc, m, m), s_A (Fc, m), Vh_A (Fc, m, n)
        U_A, s_A, Vh_A = torch.linalg.svd(A_c, full_matrices=False)

        s_A_safe = torch.clamp(s_A.real, min=svd_eps)
        s_H_c = 1.0 / s_A_safe

        # U_H^H y = V_A^H y = Vh_A @ y_c
        uh_y = torch.bmm(Vh_A, y_c.unsqueeze(-1)).squeeze(-1)  # (Fc, m)
        # y_bar = S_H^{-1} U_H^H y = s_A * (Vh_A @ y_c)
        y_bar_c = uh_y * s_A_safe.to(uh_y.dtype)

        V_H_all[f0:f1] = U_A
        s_H_all[f0:f1] = s_H_c
        y_bar_all[f0:f1] = y_bar_c

    return PerFreqOperatorSVD(
        V_H=V_H_all,
        s_H=s_H_all,
        y_bar=y_bar_all,
        m=m,
        n=n,
    )


def refine_wpe_filter(
    g_init: torch.Tensor,
    x_bar_dry_chron: torch.Tensor,
    y_vec: torch.Tensor,
    m: int,
    delay: int,
    cfg: RefinementConfig,
    n_iters: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Algorithm 1 Step 3: Refine per-frequency WPE filter g_k using PyTorch autograd:

        g_k <- g_k - alpha * grad_{g_k} ( || x_bar_dry - (I~ - G_k) y_wet ||_2^2
                                          + lambda * || g_k ||_2^2 )

    Note on PyTorch complex autograd (Wirtinger calculus):
      For real scalar loss L(g) = sum_k ( ||r_k||^2 + lam ||g_k||^2 ), PyTorch's
      `g.grad` returns `2 * dL / d(g^*)` which is the exact steepest ascent direction
      in R^{2L}, so `g <- g - alpha * g.grad` is exact gradient descent!
    """
    iters = cfg.n_refine if n_iters is None else int(n_iters)
    if iters <= 0:
        return g_init.detach().clone(), {"iters": 0, "initial_loss": 0.0, "final_loss": 0.0}

    device = g_init.device
    g_param = g_init.detach().to(torch.complex64).clone().requires_grad_(True)
    x_target_vec = pack_dry_vector(x_bar_dry_chron.detach().to(torch.complex64))
    y_vec_c = y_vec.detach().to(torch.complex64)

    # Pre-unfold delayed slices once (read-only) so each iteration is a single einsum
    L = g_param.shape[-1]
    y_curr = y_vec_c[:, :m]  # (F, m)
    y_delayed = y_vec_c[:, delay : delay + m + L - 1].unfold(dimension=1, size=L, step=1)  # (F, m, L)

    alpha = float(cfg.alpha)
    lam = float(cfg.lam)

    initial_loss = 0.0
    final_loss = 0.0

    for step in range(iters):
        if g_param.grad is not None:
            g_param.grad.zero_()

        reverb = torch.einsum("fl,fml->fm", g_param.conj(), y_delayed)
        x_hat_vec = y_curr - reverb
        residual = x_target_vec - x_hat_vec

        data_term = (residual.real.square() + residual.imag.square()).sum()
        reg_term = lam * (g_param.real.square() + g_param.imag.square()).sum()
        loss = data_term + reg_term
        loss.backward()

        loss_val = float(loss.detach().item())
        if step == 0:
            initial_loss = loss_val
        final_loss = loss_val

        with torch.no_grad():
            g_param.sub_(alpha * g_param.grad)

        if progress_cb is not None and (step == 0 or (step + 1) % max(1, iters // 10) == 0 or step == iters - 1):
            progress_cb(step + 1, iters, loss_val)

    return g_param.detach().to(g_init.dtype), {
        "iters": iters,
        "initial_loss": round(initial_loss, 6),
        "final_loss": round(final_loss, 6),
    }
