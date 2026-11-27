#!/usr/bin/env python3
"""
Evaluation Suite for Unsupervised Diffusion Dereverberation (Saito et al., ICASSP 2023).

Compares on held-out dry audio convolved with synthetic RIRs, licensed real RIRs,
or optional plugin-style artificial reverb:
  1. Wet (unprocessed reverberant input)
  2. Baseline: WPE alone (L=200, D=8, 3 iterations)
  3. Proposed (no refinement): Algorithm 1 Steps 1-2 (requires trained checkpoint)
  4. Proposed + refinement: Full Algorithm 1 Steps 1-4 (requires trained checkpoint)
  5. Oracle: DDRM using least-squares oracle filter g_k* (requires trained checkpoint)

Metrics:
  - Paper metrics: Magnitude spectrogram L1 loss vs dry reference + 16 kHz VGGish FAD
  - Speech metrics: Blind RT60, DRR proxy, optional DNSMOS P.835, optional Whisper WER
  - Wall-clock runtime (seconds per minute of audio)
  - Side-by-side comparison with Saito et al. Table 1 reference benchmarks
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

from dereverb_lib.config import (
    DDRMConfig,
    DereverbPipelineConfig,
    PostProcessConfig,
    RefinementConfig,
    STFTConfig,
    WPEConfig,
)
from dereverb_lib.dataset import AUDIO_EXTENSIONS, load_mono_audio, simulate_reverb
from dereverb_lib.ddrm import dereverb_waveform, run_algorithm1_patch
from dereverb_lib.metrics import (
    PAPER_TABLE1_REFERENCE,
    compute_dnsmos_p835,
    compute_mag_stft_l1,
    compute_vggish_fad_16k,
    compute_whisper_wer,
    estimate_blind_rt60_and_drr,
    estimate_oracle_filter,
)
from dereverb_lib.stft import compute_istft, compute_stft, postprocess_dereverb_output
from dereverb import load_prior_checkpoint


def _resample_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return wav.astype(np.float32)
    from scipy.signal import resample_poly

    gcd = math.gcd(sr, 16000)
    return resample_poly(wav, 16000 // gcd, sr // gcd).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate WPE baseline and Diffusion Dereverb (Saito et al., ICASSP 2023)")
    p.add_argument("--dry-dir", type=str, required=True, help="Directory of held-out dry audio files")
    p.add_argument("--checkpoint", type=str, default="", help="Trained diffusion prior checkpoint (.pt). If omitted, evaluates Wet + WPE baseline only.")
    p.add_argument("--reverb-mode", type=str, default="synthetic_rir", choices=["synthetic_rir", "real_rir", "plugin"])
    p.add_argument("--rir-dir", type=str, default="", help="Directory of real RIR WAVs (required when --reverb-mode=real_rir)")
    p.add_argument("--rt60", type=float, default=0.8, help="Synthetic RIR RT60 in seconds (default 0.8)")
    p.add_argument("--drr-db", type=float, default=3.0, help="Synthetic RIR DRR in dB (default 3.0)")
    p.add_argument("--max-files", type=int, default=10, help="Maximum test files to evaluate")
    p.add_argument("--max-sec", type=float, default=6.0, help="Crop each test file to at most N seconds")
    p.add_argument("--fast", action="store_true", help="Use --fast mode for proposed refinement")
    p.add_argument("--dry-mix", type=float, default=0.0, help="Dry/wet blend during evaluation (0.0 matches paper Table 1; 0.15 for listening)")
    p.add_argument("--dnsmos-onnx", type=str, default="", help="Optional path to DNSMOS P.835 ONNX model")
    p.add_argument("--transcripts-json", type=str, default="", help="Optional JSON mapping filename -> reference transcript for Whisper WER")
    p.add_argument("--out-json", type=str, default="eval_results.json", help="Output JSON results path")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dry_dir = Path(args.dry_dir).expanduser().resolve()
    dry_files = sorted(
        p for p in dry_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )[: args.max_files]
    if not dry_files:
        raise FileNotFoundError(f"No audio files found in {dry_dir}")

    rir_files: List[Path] = []
    if args.reverb_mode == "real_rir":
        rir_dir = Path(args.rir_dir).expanduser().resolve()
        rir_files = sorted(p for p in rir_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
        if not rir_files:
            raise FileNotFoundError(f"No RIR files found in {rir_dir}")

    transcripts: Dict[str, str] = {}
    if args.transcripts_json:
        transcripts = json.loads(Path(args.transcripts_json).read_text())

    model = None
    diffusion = None
    stft_cfg = STFTConfig(sample_rate=args.sample_rate)
    if args.checkpoint:
        model, diffusion, stft_cfg = load_prior_checkpoint(Path(args.checkpoint), device=device)

    methods = ["wet", "wpe_baseline"]
    if model is not None:
        methods.extend(["proposed_no_refine", "proposed_with_refine", "oracle"])

    stats: Dict[str, Dict[str, List[float]]] = {
        m: {"l1": [], "rt60": [], "drr": [], "sec_per_min": [], "dnsmos_ovrl": [], "wer": []}
        for m in methods
    }

    with tempfile.TemporaryDirectory(prefix="dereverb_eval_") as tmp_root:
        tmp_path = Path(tmp_root)
        ref_16k_dir = tmp_path / "ref_16k"
        ref_16k_dir.mkdir()
        method_dirs = {}
        for m in methods:
            d = tmp_path / f"{m}_16k"
            d.mkdir()
            method_dirs[m] = d

        for idx, dry_path in enumerate(dry_files):
            dry_wav = load_mono_audio(dry_path, stft_cfg.sample_rate)
            max_samples = int(args.max_sec * stft_cfg.sample_rate)
            if len(dry_wav) > max_samples:
                dry_wav = dry_wav[:max_samples]

            rir_path = rir_files[idx % len(rir_files)] if rir_files else None
            wet_wav, _ = simulate_reverb(
                dry_wav,
                sample_rate=stft_cfg.sample_rate,
                mode=args.reverb_mode,
                rt60_sec=args.rt60,
                drr_db=args.drr_db,
                rir_path=rir_path,
                seed=args.seed + idx,
            )

            sf.write(str(ref_16k_dir / f"{idx:04d}.wav"), _resample_16k(dry_wav, stft_cfg.sample_rate), 16000)
            outputs: Dict[str, tuple[np.ndarray, float]] = {"wet": (wet_wav, 0.0)}

            # 1. Baseline WPE alone (L=200, D=8, 3 iters)
            cfg_wpe = DereverbPipelineConfig(
                stft=stft_cfg,
                post=PostProcessConfig(dry_mix=args.dry_mix),
                baseline_wpe_only=True,
            )
            wpe_wav, wpe_rep = dereverb_waveform(wet_wav, stft_cfg.sample_rate, cfg_wpe, device=device)
            outputs["wpe_baseline"] = (wpe_wav, float(wpe_rep["sec_per_minute_audio"]))

            if model is not None and diffusion is not None:
                # 2. Proposed (no refinement)
                cfg_noref = DereverbPipelineConfig(
                    stft=stft_cfg,
                    post=PostProcessConfig(dry_mix=args.dry_mix),
                    fast=args.fast,
                    skip_refinement=True,
                )
                noref_wav, noref_rep = dereverb_waveform(
                    wet_wav, stft_cfg.sample_rate, cfg_noref, model=model, diffusion=diffusion, device=device
                )
                outputs["proposed_no_refine"] = (noref_wav, float(noref_rep["sec_per_minute_audio"]))

                # 3. Proposed + refinement
                cfg_ref = DereverbPipelineConfig(
                    stft=stft_cfg,
                    post=PostProcessConfig(dry_mix=args.dry_mix),
                    fast=args.fast,
                    skip_refinement=False,
                )
                ref_wav_out, ref_rep = dereverb_waveform(
                    wet_wav, stft_cfg.sample_rate, cfg_ref, model=model, diffusion=diffusion, device=device
                )
                outputs["proposed_with_refine"] = (ref_wav_out, float(ref_rep["sec_per_minute_audio"]))

                # 4. Oracle operator + DDRM (single patch up to 512 frames for paper sanity-check)
                t_o = time.perf_counter()
                y_spec = compute_stft(torch.from_numpy(wet_wav).to(device), stft_cfg)
                x_spec = compute_stft(torch.from_numpy(dry_wav).to(device), stft_cfg)
                g_oracle = estimate_oracle_filter(y_spec, x_spec, taps=150, delay=8)
                cfg_oracle = DereverbPipelineConfig(
                    stft=stft_cfg,
                    post=PostProcessConfig(dry_mix=args.dry_mix),
                    fast=args.fast,
                    skip_refinement=True,
                )
                x_o_spec, _ = run_algorithm1_patch(
                    y_full_spec=y_spec,
                    start_frame=0,
                    model=model,
                    diffusion=diffusion,
                    cfg=cfg_oracle,
                    g_init_override=g_oracle,
                )
                valid_f = min(y_spec.shape[1], x_o_spec.shape[1])
                oracle_raw = compute_istft(x_o_spec[:, :valid_f], stft_cfg, length=len(wet_wav)).detach().cpu().numpy()
                oracle_wav, _ = postprocess_dereverb_output(oracle_raw, wet_wav, stft_cfg.sample_rate, cfg_oracle.post)
                o_sec_per_min = (time.perf_counter() - t_o) / max((len(wet_wav) / stft_cfg.sample_rate) / 60.0, 1e-6)
                outputs["oracle"] = (oracle_wav, float(o_sec_per_min))

            # Score all methods on this clip
            ref_txt = transcripts.get(dry_path.name, "")
            dnsmos_path = Path(args.dnsmos_onnx) if args.dnsmos_onnx else None

            for m_name, (m_wav, sec_per_min) in outputs.items():
                wav_16k = _resample_16k(m_wav, stft_cfg.sample_rate)
                wav_16k_path = method_dirs[m_name] / f"{idx:04d}.wav"
                sf.write(str(wav_16k_path), wav_16k, 16000)

                l1 = compute_mag_stft_l1(m_wav, dry_wav, stft_cfg)
                rt60_est, drr_est = estimate_blind_rt60_and_drr(m_wav, stft_cfg.sample_rate)
                stats[m_name]["l1"].append(l1)
                stats[m_name]["sec_per_min"].append(sec_per_min)
                if rt60_est is not None:
                    stats[m_name]["rt60"].append(rt60_est)
                if drr_est is not None:
                    stats[m_name]["drr"].append(drr_est)

                dns = compute_dnsmos_p835(wav_16k, onnx_model_path=dnsmos_path)
                if dns is not None:
                    stats[m_name]["dnsmos_ovrl"].append(dns["ovrl"])
                if ref_txt:
                    w_val = compute_whisper_wer(wav_16k_path, ref_txt)
                    if w_val is not None:
                        stats[m_name]["wer"].append(w_val)

        # Aggregate & compute 16 kHz VGGish FAD across directory
        summary: Dict[str, Dict] = {}
        for m_name in methods:
            fad_val = compute_vggish_fad_16k(ref_16k_dir, method_dirs[m_name])
            s = stats[m_name]
            summary[m_name] = {
                "mag_stft_l1": round(float(np.mean(s["l1"])), 4) if s["l1"] else None,
                "vggish_fad_16k": round(fad_val, 3) if fad_val is not None else None,
                "rt60_sec": round(float(np.mean(s["rt60"])), 3) if s["rt60"] else None,
                "drr_db": round(float(np.mean(s["drr"])), 2) if s["drr"] else None,
                "dnsmos_ovrl": round(float(np.mean(s["dnsmos_ovrl"])), 3) if s["dnsmos_ovrl"] else None,
                "whisper_wer": round(float(np.mean(s["wer"])), 4) if s["wer"] else None,
                "sec_per_min_audio": round(float(np.mean(s["sec_per_min"])), 2) if s["sec_per_min"] else 0.0,
                "paper_ref": PAPER_TABLE1_REFERENCE.get(m_name),
            }

    Path(args.out_json).write_text(
        json.dumps(
            {
                "device": str(device),
                "reverb_mode": args.reverb_mode,
                "num_files": len(dry_files),
                "checkpoint": args.checkpoint or None,
                "results": summary,
            },
            indent=2,
        )
    )

    print("\n| Method | Mag STFT L1 (Ours / Paper) | 16k VGGish FAD (Ours / Paper) | Est RT60 (s) | Est DRR (dB) | DNSMOS OVRL | Whisper WER | Sec / Min Audio |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m_name, row in summary.items():
        ref = row.get("paper_ref") or {}
        l1_str = f"{row['mag_stft_l1']} / {ref.get('mag_stft_l1', '—')}"
        fad_str = f"{row['vggish_fad_16k'] if row['vggish_fad_16k'] is not None else 'N/A'} / {ref.get('vggish_fad_16k', '—')}"
        print(
            f"| {m_name} | {l1_str} | {fad_str} | "
            f"{row['rt60_sec'] if row['rt60_sec'] is not None else 'N/A'} | "
            f"{row['drr_db'] if row['drr_db'] is not None else 'N/A'} | "
            f"{row['dnsmos_ovrl'] if row['dnsmos_ovrl'] is not None else 'N/A'} | "
            f"{row['whisper_wer'] if row['whisper_wer'] is not None else 'N/A'} | "
            f"{row['sec_per_min_audio']} |"
        )


if __name__ == "__main__":
    main()
