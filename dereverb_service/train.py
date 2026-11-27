#!/usr/bin/env python3
"""
Unsupervised Diffusion Prior Training Script (Dry Audio Only — No Pairs)
Saito et al., Sony, ICASSP 2023 (arXiv:2211.04124), Table 2.

Hyperparameters (defaults match Paper Table 2):
  - Complex STFT: n_fft=1024, hop=256, Hann, drop DC -> (2, 512, 512)
  - U-Net: base_ch=64, depth=2, mults=(1,1,2,2,4,4), heads=2, attn=(32,16),
           BigGAN up/down, dropout=0.0, AdaGN + time-distributed FC (~31.3M params)
  - Diffusion: 4000 steps, cosine schedule, Eq. 9 weighted x0 MSE loss
  - Optimizer: AdamW (lr=1e-4, beta1=0.9, beta2=0.999), batch_size=6, 370K iters
  - Precision: 16-bit AMP (CUDA) + EMA 0.9999

Usage:
  python train.py --data-dir /path/to/dry_audio --out-dir ./checkpoints
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dereverb_lib.config import STFTConfig, TrainingConfig, UNetConfig
from dereverb_lib.dataset import DryAudioSTFTDataset
from dereverb_lib.diffusion import EMAHelper, GaussianDiffusionPrior
from dereverb_lib.unet import AudioDiffusionUNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train dry-only complex STFT diffusion prior (Saito et al., ICASSP 2023)")
    p.add_argument("--data-dir", type=str, required=True, help="Directory containing dry mono/stereo WAV/FLAC audio")
    p.add_argument("--out-dir", type=str, default="checkpoints", help="Directory to save checkpoints & logs")
    p.add_argument("--sample-rate", type=int, default=44100, help="Target sample rate (default: 44100)")
    p.add_argument("--batch-size", type=int, default=6, help="Batch size (paper Table 2: 6)")
    p.add_argument("--max-iters", type=int, default=370_000, help="Total training iterations (paper Table 2: 370000)")
    p.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (paper Table 2: 1e-4)")
    p.add_argument("--ema-rate", type=float, default=0.9999, help="EMA decay rate (paper Table 2: 0.9999)")
    p.add_argument("--gamma-mode", type=str, default="snr_clamped", choices=["snr_clamped", "snr", "uniform"],
                   help="Eq. 9 weighting gamma_t mode (ASSUMPTION: default snr_clamped)")
    p.add_argument("--stft-scale", type=float, default=1.0, help="STFT scaling constant stored in checkpoint")
    p.add_argument("--no-amp", action="store_true", help="Disable 16-bit mixed precision")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt file to resume training")
    p.add_argument("--save-every", type=int, default=10_000, help="Save checkpoint every N iterations")
    p.add_argument("--log-every", type=int, default=100, help="Log metrics every N iterations")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        warnings.warn(
            "WARNING: No CUDA GPU detected. Training a 31M-parameter 512x512 diffusion U-Net "
            "for 370K iterations on CPU is impractical; use a GPU (e.g., A100 / RTX 4090).",
            RuntimeWarning,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stft_cfg = STFTConfig(sample_rate=args.sample_rate, stft_scale=args.stft_scale)
    unet_cfg = UNetConfig()
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.lr,
        ema_rate=args.ema_rate,
        use_amp=(not args.no_amp) and device.type == "cuda",
        save_every=args.save_every,
        log_every=args.log_every,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = AudioDiffusionUNet(unet_cfg).to(device)
    param_count = model.count_parameters()
    print(f"[Model] AudioDiffusionUNet initialized: {param_count:,} parameters ({param_count / 1e6:.2f}M)")
    print(f"[Device] {device} | AMP fp16: {train_cfg.use_amp} | Sample rate: {stft_cfg.sample_rate} Hz")

    ema = EMAHelper(model, rate=train_cfg.ema_rate)
    diffusion = GaussianDiffusionPrior(
        num_timesteps=train_cfg.diffusion_steps,
        schedule=train_cfg.noise_schedule,
        gamma_mode=args.gamma_mode,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        weight_decay=train_cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.use_amp)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and train_cfg.use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))
        print(f"[Resume] Loaded checkpoint {args.resume} at step {start_step}")

    dataset = DryAudioSTFTDataset(
        root_dir=args.data_dir,
        stft_cfg=stft_cfg,
        samples_per_epoch=max(args.batch_size * 2000, 10_000),
    )
    print(f"[Dataset] Found {len(dataset.files)} dry audio files under {args.data_dir}")

    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    def save_checkpoint(step: int, tag: str = "") -> Path:
        name = f"prior_step_{step:07d}{tag}.pt"
        path = out_dir / name
        payload = {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "stft_cfg": dataclasses.asdict(stft_cfg),
            "unet_cfg": dataclasses.asdict(unet_cfg),
            "train_cfg": dataclasses.asdict(train_cfg),
            "gamma_mode": args.gamma_mode,
            "param_count": param_count,
        }
        torch.save(payload, path)
        latest = out_dir / "latest.pt"
        torch.save(payload, latest)
        return path

    model.train()
    step = start_step
    t_log = time.perf_counter()
    running_loss = 0.0
    running_mse = 0.0
    count_log = 0

    while step < train_cfg.max_iters:
        for batch_x0 in loader:
            if step >= train_cfg.max_iters:
                break

            x0 = batch_x0.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=train_cfg.use_amp):
                loss, metrics = diffusion.training_loss(model, x0)

            scaler.scale(loss).backward()
            if train_cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            step += 1

            running_loss += metrics["loss"]
            running_mse += metrics["unweighted_mse"]
            count_log += 1

            if step % train_cfg.log_every == 0 or step == 1:
                dt = time.perf_counter() - t_log
                avg_loss = running_loss / max(1, count_log)
                avg_mse = running_mse / max(1, count_log)
                it_per_sec = count_log / max(dt, 1e-6)
                print(
                    f"[Step {step:>7d}/{train_cfg.max_iters}] "
                    f"loss={avg_loss:.6f} | x0_mse={avg_mse:.6f} | {it_per_sec:.2f} it/s"
                )
                running_loss = 0.0
                running_mse = 0.0
                count_log = 0
                t_log = time.perf_counter()

            if step % train_cfg.save_every == 0:
                saved = save_checkpoint(step)
                print(f"[Checkpoint] Saved {saved}")

    final_path = save_checkpoint(step, tag="_final")
    meta_path = out_dir / "training_summary.json"
    meta_path.write_text(
        json.dumps(
            {
                "final_step": step,
                "param_count": param_count,
                "sample_rate": stft_cfg.sample_rate,
                "gamma_mode": args.gamma_mode,
                "checkpoint": str(final_path),
            },
            indent=2,
        )
    )
    print(f"[Done] Training finished at step {step}. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
