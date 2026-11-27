#!/usr/bin/env python3
"""
CLI Entry Point: Unsupervised Diffusion Dereverberation (Saito et al., ICASSP 2023)

Usage:
  python dereverb.py in.wav out.wav --checkpoint checkpoints/latest.pt [--fast]
  python dereverb.py in.wav out.wav --baseline-wpe

Safety rules enforced:
  - Never overwrites the input file (exits with error if input and output resolve identically)
  - Uses CUDA GPU automatically if available; prints explicit warning if running on CPU
  - Matches output loudness to input within +/- 0.5 LU (ITU-R BS.1770)
  - Applies -1.5 dBTP true-peak limiter and configurable 10-20% dry/wet blend (default 15%)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import soundfile as sf
import torch

from dereverb_lib.config import (
    DDRMConfig,
    DereverbPipelineConfig,
    PostProcessConfig,
    RefinementConfig,
    STFTConfig,
    UNetConfig,
    WPEConfig,
)
from dereverb_lib.ddrm import dereverb_waveform
from dereverb_lib.diffusion import GaussianDiffusionPrior
from dereverb_lib.unet import AudioDiffusionUNet


def load_prior_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[AudioDiffusionUNet, GaussianDiffusionPrior, STFTConfig]:
    """Load trained EMA diffusion prior weights and STFT config from checkpoint."""
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    unet_dict = ckpt.get("unet_cfg", {})
    stft_dict = ckpt.get("stft_cfg", {})
    train_dict = ckpt.get("train_cfg", {})

    # Convert list fields back to tuples for dataclasses
    if "channel_mult" in unet_dict:
        unet_dict["channel_mult"] = tuple(unet_dict["channel_mult"])
    if "attention_resolutions" in unet_dict:
        unet_dict["attention_resolutions"] = tuple(unet_dict["attention_resolutions"])

    unet_cfg = UNetConfig(**unet_dict) if unet_dict else UNetConfig()
    stft_cfg = STFTConfig(**stft_dict) if stft_dict else STFTConfig()

    model = AudioDiffusionUNet(unet_cfg).to(device)
    weights = ckpt.get("ema") or ckpt.get("model")
    if weights is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'ema' or 'model' state_dict.")
    model.load_state_dict(weights)
    model.eval()

    diffusion = GaussianDiffusionPrior(
        num_timesteps=int(train_dict.get("diffusion_steps", 4000)),
        schedule=str(train_dict.get("noise_schedule", "cosine")),
        gamma_mode=str(ckpt.get("gamma_mode", "snr_clamped")),
        device=device,
    )
    return model, diffusion, stft_cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unsupervised Vocal Dereverberation via Diffusion Prior + DDRM + WPE Refinement (Saito et al., ICASSP 2023)"
    )
    p.add_argument("input_wav", type=str, help="Path to input reverberant audio file")
    p.add_argument("output_wav", type=str, help="Path to write dereverbed WAV file (must differ from input)")
    p.add_argument("--checkpoint", type=str, default="", help="Path to trained diffusion prior .pt checkpoint")
    p.add_argument("--fast", action="store_true", help="Fast mode: fewer refinement iterations & DDRM steps (trades quality for speed)")
    p.add_argument("--no-refine", action="store_true", help="Ablation: skip Step 3 filter refinement (DDRM pass 1 only)")
    p.add_argument("--baseline-wpe", action="store_true", help="Run baseline WPE alone (L=200, D=8, 3 iterations; no checkpoint needed)")
    p.add_argument("--dry-mix", type=float, default=0.15, help="Original dry/wet blend ratio 0.0-0.5 (recommended 0.10-0.20, default 0.15)")
    p.add_argument("--no-loudness-match", action="store_true", help="Disable ITU-R BS.1770 loudness matching")
    p.add_argument("--ceiling-dbtp", type=float, default=-1.5, help="True-peak limiter ceiling in dBTP (default -1.5)")
    # Paper inference overrides (all configurable)
    p.add_argument("--wpe-taps", type=int, default=150, help="WPE filter length L (default 150)")
    p.add_argument("--wpe-delay", type=int, default=8, help="WPE prediction delay D (default 8)")
    p.add_argument("--wpe-iters", type=int, default=1, help="WPE init iterations (default 1)")
    p.add_argument("--ddrm-steps", type=int, default=20, help="DDRM sampling steps (default 20)")
    p.add_argument("--eta", type=float, default=0.7, help="DDRM eta (default 0.7)")
    p.add_argument("--eta-b", type=float, default=0.2, help="DDRM eta_b (default 0.2)")
    p.add_argument("--sigma-y", type=float, default=1.0e-6, help="DDRM observation noise sigma_y (default 1e-6)")
    p.add_argument("--alpha", type=float, default=1.0e-6, help="Refinement step size alpha (default 1e-6)")
    p.add_argument("--lam", type=float, default=1.0, help="Refinement L2 weight lambda (default 1.0)")
    p.add_argument("--n-refine", type=int, default=10_000, help="Refinement iterations N_refine (default 10000)")
    p.add_argument("--fast-n-refine", type=int, default=500, help="Refinement iterations in --fast mode (default 500)")
    p.add_argument("--sample-rate", type=int, default=44100, help="Model STFT sample rate when not overridden by checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-report", type=str, default="", help="Optional path to save JSON metrics report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_wav).expanduser().resolve()
    out_path = Path(args.output_wav).expanduser().resolve()

    if not in_path.is_file():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    if in_path == out_path:
        print("ERROR: Refusing to overwrite input file. Choose a different output path.", file=sys.stderr)
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: Optional[AudioDiffusionUNet] = None
    diffusion: Optional[GaussianDiffusionPrior] = None
    stft_cfg = STFTConfig(sample_rate=args.sample_rate)

    if not args.baseline_wpe:
        if not args.checkpoint:
            print(
                "ERROR: --checkpoint <path/to/prior.pt> is required for diffusion dereverberation.\n"
                "Train a dry-only prior with train.py, or pass --baseline-wpe to run the WPE baseline.",
                file=sys.stderr,
            )
            sys.exit(2)
        ckpt_path = Path(args.checkpoint).expanduser().resolve()
        if not ckpt_path.is_file():
            print(f"ERROR: Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        model, diffusion, stft_cfg = load_prior_checkpoint(ckpt_path, device=device)

    cfg = DereverbPipelineConfig(
        stft=stft_cfg,
        wpe=WPEConfig(taps=args.wpe_taps, delay=args.wpe_delay, iterations=args.wpe_iters),
        ddrm=DDRMConfig(
            eta=args.eta,
            eta_b=args.eta_b,
            sigma_y=args.sigma_y,
            sampling_steps=args.ddrm_steps,
        ),
        refinement=RefinementConfig(
            alpha=args.alpha,
            lam=args.lam,
            n_refine=args.n_refine,
            fast_n_refine=args.fast_n_refine,
        ),
        post=PostProcessConfig(
            match_loudness=not args.no_loudness_match,
            true_peak_ceiling_dbtp=args.ceiling_dbtp,
            dry_mix=args.dry_mix,
        ),
        fast=args.fast,
        skip_refinement=args.no_refine,
        baseline_wpe_only=args.baseline_wpe,
        seed=args.seed,
    )

    audio, sr = sf.read(str(in_path), dtype="float32", always_2d=False)

    def _progress(frac: float, msg: str) -> None:
        print(f"[{int(frac * 100):3d}%] {msg}")

    out_wav, report = dereverb_waveform(
        wet_mono=audio,
        sample_rate=sr,
        cfg=cfg,
        model=model,
        diffusion=diffusion,
        device=device,
        progress_cb=_progress,
    )

    sf.write(str(out_path), out_wav, sr, subtype="PCM_16")
    print(f"\nSaved dereverbed audio to: {out_path}")
    print(
        f"Mode: {report['mode']} | Device: {report['device']} | "
        f"Elapsed: {report['elapsed_sec']} s ({report['sec_per_minute_audio']} s per min audio)"
    )
    post = report["postprocess"]
    print(
        f"Loudness: {post['input_lufs']} -> {post['output_lufs']} LUFS "
        f"(delta {post['loudness_delta_lu']:+.2f} LU) | "
        f"True Peak: {post['output_true_peak_dbtp']} dBTP | Dry mix: {int(post['dry_mix'] * 100)}%"
    )

    if args.json_report:
        Path(args.json_report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
