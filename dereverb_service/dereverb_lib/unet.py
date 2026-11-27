"""
Modified OpenAI guided-diffusion / DDRM U-Net for Complex STFT Diffusion Prior
(Saito et al., Sony, ICASSP 2023, Table 2).

Architecture (following openai/guided-diffusion & bahjat-kawar/ddrm, MIT License):
  - Input / Output: 2 channels (real, imaginary) of shape (B, 2, 512, 512)
  - Base channels: 64
  - Depth: 2 residual blocks per resolution
  - Channel multipliers: (1, 1, 2, 2, 4, 4) -> resolutions 512, 256, 128, 64, 32, 16
  - Attention resolutions: (32, 16) with 2 attention heads
  - ResBlock up/downsampling (BigGAN style)
  - Dropout: 0.0
  - Modification 1: Adaptive Group Normalization (AdaGN / scale-shift norm) in every ResBlock
  - Modification 2: Time-distributed fully connected (FC) layer at the end of every ResBlock
    (operates along the frequency axis for each time frame independently, capturing global
    harmonic correlations across frequency bins; ~31.3M total parameters).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import UNetConfig


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it (guided-diffusion convention)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embeddings matching openai/guided-diffusion."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class GroupNorm32(nn.GroupNorm):
    """Float32-safe GroupNorm for 16-bit AMP training stability."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class TimestepBlock(nn.Module):
    """Base module for blocks that accept (x, emb)."""

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential wrapper passing timestep embeddings to TimestepBlock children."""

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample2x(nn.Module):
    def __init__(self, channels: int, use_conv: bool = False):
        super().__init__()
        self.use_conv = use_conv
        self.conv = nn.Conv2d(channels, channels, 3, padding=1) if use_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.conv is not None:
            x = self.conv(x)
        return x


class Downsample2x(nn.Module):
    def __init__(self, channels: int, use_conv: bool = False):
        super().__init__()
        self.op = (
            nn.Conv2d(channels, channels, 3, stride=2, padding=1)
            if use_conv
            else nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class TimeDistributedFreqFC(nn.Module):
    """
    Time-distributed fully connected layer operating along the frequency axis
    at the end of every residual block.

    Input shape: (B, C, F_l, T_l)
    Operation: Applies a shared linear projection (F_l -> F_l) along frequency F_l
    independently for every time frame t and channel c, followed by SiLU and a
    residual connection initialized near zero so training starts cleanly.

    ASSUMPTION: The paper specifies "a time-distributed fully connected layer at the
    end of every residual block" without printing its exact channel-sharing vs
    bottleneck ratio. Sharing a (F_l -> F_l) linear map across (B, C, T_l) with
    GroupNorm + SiLU keeps memory tractable at 512x512 and brings total parameter
    count into the ~31M range reported in Table 2.
    """

    def __init__(self, channels: int, freq_dim: int, norm_groups: int = 32):
        super().__init__()
        groups = min(norm_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = GroupNorm32(groups, channels)
        self.act = nn.SiLU()
        self.fc = zero_module(nn.Linear(freq_dim, freq_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T) -> transpose to (B, C, T, F) -> Linear(F, F) -> back to (B, C, F, T)
        h = self.act(self.norm(x))
        h = h.transpose(-1, -2)  # (B, C, T, F)
        h = self.fc(h)
        h = h.transpose(-1, -2)  # (B, C, F, T)
        return x + h


class ResBlock(TimestepBlock):
    """
    Residual block with:
      - Adaptive Group Normalization (AdaGN: scale & shift from timestep embedding)
      - Optional BigGAN-style up/downsampling inside the residual block
      - Optional TimeDistributedFreqFC at the end of the block (Saito et al. mod)
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: int | None = None,
        use_scale_shift_norm: bool = True,
        up: bool = False,
        down: bool = False,
        use_time_distributed_fc: bool = True,
        out_freq_dim: int = 512,
        norm_groups: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down

        in_groups = min(norm_groups, channels)
        out_groups = min(norm_groups, self.out_channels)

        self.in_layers = nn.Sequential(
            GroupNorm32(in_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        if up:
            self.h_upd: nn.Module = Upsample2x(channels, use_conv=False)
            self.x_upd: nn.Module = Upsample2x(channels, use_conv=False)
        elif down:
            self.h_upd = Downsample2x(channels, use_conv=False)
            self.x_upd = Downsample2x(channels, use_conv=False)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            GroupNorm32(out_groups, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection: nn.Module = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

        self.td_fc: nn.Module | None = (
            TimeDistributedFreqFC(self.out_channels, out_freq_dim, norm_groups=norm_groups)
            if use_time_distributed_fc
            else None
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out.unsqueeze(-1)

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1.0 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        out = self.skip_connection(x) + h
        if self.td_fc is not None:
            out = self.td_fc(out)
        return out


class AttentionBlock(nn.Module):
    """Multi-head 2D spatial self-attention block (guided-diffusion style)."""

    def __init__(self, channels: int, num_heads: int = 2, norm_groups: int = 32):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        groups = min(norm_groups, channels)
        self.norm = GroupNorm32(groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, h * w)
        qkv = self.qkv(self.norm(x_flat))  # (B, 3C, N)
        ch_per_head = c // self.num_heads
        q, k, v = qkv.reshape(b * self.num_heads, 3 * ch_per_head, h * w).chunk(3, dim=1)

        # Use PyTorch scaled_dot_product_attention (FlashAttention / memory-efficient on GPU)
        q_t = q.transpose(1, 2)  # (BH, N, D)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q_t, k_t, v_t)  # (BH, N, D)
        attn = attn.transpose(1, 2).reshape(b, c, h * w)
        out = self.proj_out(attn)
        return (x_flat + out).reshape(b, c, h, w)


class AudioDiffusionUNet(nn.Module):
    """
    Modified OpenAI guided-diffusion U-Net for 2-channel 512x512 complex STFT
    x0-prediction diffusion prior (Saito et al., ICASSP 2023, Table 2).
    """

    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        model_channels = cfg.model_channels
        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = int(cfg.channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(cfg.in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]

        ds = 1  # Downsample factor relative to input_freq_bins (512)
        attn_ds = {cfg.input_freq_bins // r for r in cfg.attention_resolutions}

        for level, mult in enumerate(cfg.channel_mult):
            cur_freq = cfg.input_freq_bins // ds
            out_ch = int(mult * model_channels)
            for _ in range(cfg.num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=cfg.dropout,
                        out_channels=out_ch,
                        use_scale_shift_norm=cfg.use_scale_shift_norm,
                        use_time_distributed_fc=cfg.use_time_distributed_fc,
                        out_freq_dim=cur_freq,
                        norm_groups=cfg.norm_groups,
                    )
                ]
                ch = out_ch
                if ds in attn_ds:
                    layers.append(
                        AttentionBlock(ch, num_heads=cfg.num_heads, norm_groups=cfg.norm_groups)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(cfg.channel_mult) - 1:
                next_freq = cur_freq // 2
                if cfg.resblock_updown:
                    down_block: nn.Module = ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=cfg.dropout,
                        out_channels=ch,
                        use_scale_shift_norm=cfg.use_scale_shift_norm,
                        down=True,
                        use_time_distributed_fc=cfg.use_time_distributed_fc,
                        out_freq_dim=next_freq,
                        norm_groups=cfg.norm_groups,
                    )
                else:
                    down_block = Downsample2x(ch, use_conv=True)
                self.input_blocks.append(TimestepEmbedSequential(down_block))
                input_block_chans.append(ch)
                ds *= 2

        mid_freq = cfg.input_freq_bins // ds
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=cfg.dropout,
                use_scale_shift_norm=cfg.use_scale_shift_norm,
                use_time_distributed_fc=cfg.use_time_distributed_fc,
                out_freq_dim=mid_freq,
                norm_groups=cfg.norm_groups,
            ),
            AttentionBlock(ch, num_heads=cfg.num_heads, norm_groups=cfg.norm_groups),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=cfg.dropout,
                use_scale_shift_norm=cfg.use_scale_shift_norm,
                use_time_distributed_fc=cfg.use_time_distributed_fc,
                out_freq_dim=mid_freq,
                norm_groups=cfg.norm_groups,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(cfg.channel_mult))[::-1]:
            cur_freq = cfg.input_freq_bins // ds
            out_ch = int(mult * model_channels)
            for i in range(cfg.num_res_blocks + 1):
                skip_ch = input_block_chans.pop()
                layers = [
                    ResBlock(
                        channels=ch + skip_ch,
                        emb_channels=time_embed_dim,
                        dropout=cfg.dropout,
                        out_channels=out_ch,
                        use_scale_shift_norm=cfg.use_scale_shift_norm,
                        use_time_distributed_fc=cfg.use_time_distributed_fc,
                        out_freq_dim=cur_freq,
                        norm_groups=cfg.norm_groups,
                    )
                ]
                ch = out_ch
                if ds in attn_ds:
                    layers.append(
                        AttentionBlock(ch, num_heads=cfg.num_heads, norm_groups=cfg.norm_groups)
                    )
                if level > 0 and i == cfg.num_res_blocks:
                    next_freq = cur_freq * 2
                    if cfg.resblock_updown:
                        layers.append(
                            ResBlock(
                                channels=ch,
                                emb_channels=time_embed_dim,
                                dropout=cfg.dropout,
                                out_channels=ch,
                                use_scale_shift_norm=cfg.use_scale_shift_norm,
                                up=True,
                                use_time_distributed_fc=cfg.use_time_distributed_fc,
                                out_freq_dim=next_freq,
                                norm_groups=cfg.norm_groups,
                            )
                        )
                    else:
                        layers.append(Upsample2x(ch, use_conv=True))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        out_groups = min(cfg.norm_groups, ch)
        self.out = nn.Sequential(
            GroupNorm32(out_groups, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, cfg.out_channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Predict clean dry STFT x_0 from noisy STFT x_t (Eq. 9 x0-prediction).

        Args:
            x: (B, 2, 512, 512) real tensor [Re, Im]
            timesteps: (B,) integer or float diffusion timesteps in [0, T-1]

        Returns:
            x0_hat: (B, 2, 512, 512) predicted dry complex STFT
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.cfg.model_channels))
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
