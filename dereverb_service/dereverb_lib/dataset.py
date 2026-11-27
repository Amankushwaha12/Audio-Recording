"""
Dry-only unsupervised training dataset and reverb simulation utilities.

Training uses ONLY dry audio (no paired wet/dry data):
  - Configurable sample rate (default 44.1 kHz mono)
  - Random crop yielding a 512-frame complex STFT (DC removed -> 2 x 512 x 512)
  - Supports NHSS / NUS-48E (singing vocals, non-commercial research licenses)
    or speech corpora such as LibriTTS / VCTK (CC BY 4.0) for domain adaptation.

Evaluation reverb simulation supports:
  1. Real Room Impulse Responses (RIRs) from a user-supplied folder (e.g. OpenSLR SLR28)
  2. Synthetic Schroeder / exponential-decay diffuse RIRs with configurable RT60 & DRR
  3. Optional plugin-style artificial reverb via `pedalboard` (Spotify, GPLv3 — optional extra)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .config import STFTConfig
from .stft import complex_to_2ch, compute_stft

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def load_mono_audio(path: Path, target_sr: int) -> np.ndarray:
    """Load audio file as float32 mono resampled to target_sr."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=-1).astype(np.float32)
    if sr != target_sr:
        from scipy.signal import resample_poly

        gcd = math.gcd(sr, target_sr)
        data = resample_poly(data, target_sr // gcd, sr // gcd).astype(np.float32)
    return data


class DryAudioSTFTDataset(Dataset):
    """
    Unsupervised dry-audio-only dataset yielding 2-channel 512 x 512 complex STFT crops.
    """

    def __init__(
        self,
        root_dir: str | Path,
        stft_cfg: STFTConfig,
        samples_per_epoch: int = 10_000,
        rms_normalize_db: Optional[float] = -20.0,
        min_active_rms_db: float = -45.0,
    ):
        self.root_dir = Path(root_dir)
        self.stft_cfg = stft_cfg
        self.samples_per_epoch = samples_per_epoch
        # ASSUMPTION: Normalizing dry training crops to -20 dBFS RMS (or None to disable)
        # stabilizes diffusion training variance across diverse microphones/corpora.
        self.rms_normalize_db = rms_normalize_db
        self.min_active_rms_db = min_active_rms_db

        self.files: List[Path] = sorted(
            p for p in self.root_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not self.files:
            raise FileNotFoundError(
                f"No audio files ({', '.join(sorted(AUDIO_EXTENSIONS))}) found under {self.root_dir}"
            )

        # Samples required so torch.stft(..., center=True) yields >= patch_frames frames
        self.crop_samples = (stft_cfg.patch_frames - 1) * stft_cfg.hop_length

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _random_active_crop(self, wav: np.ndarray) -> np.ndarray:
        n = len(wav)
        if n <= self.crop_samples:
            out = np.zeros(self.crop_samples, dtype=np.float32)
            out[:n] = wav
            return out

        best_crop = None
        best_rms = -1.0
        min_lin = 10.0 ** (self.min_active_rms_db / 20.0)
        for _ in range(6):
            start = random.randint(0, n - self.crop_samples)
            seg = wav[start : start + self.crop_samples]
            rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
            if rms >= min_lin:
                return seg
            if rms > best_rms:
                best_rms = rms
                best_crop = seg
        return best_crop if best_crop is not None else wav[: self.crop_samples]

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index % len(self.files)]
        wav = load_mono_audio(path, self.stft_cfg.sample_rate)
        crop = self._random_active_crop(wav)

        if self.rms_normalize_db is not None:
            rms = float(np.sqrt(np.mean(crop * crop) + 1e-12))
            if rms > 1e-6:
                target_rms = 10.0 ** (self.rms_normalize_db / 20.0)
                crop = np.clip(crop * (target_rms / rms), -1.0, 1.0).astype(np.float32)

        spec = compute_stft(torch.from_numpy(crop), self.stft_cfg)  # (512, T)
        if spec.shape[1] < self.stft_cfg.patch_frames:
            spec = torch.nn.functional.pad(spec, (0, self.stft_cfg.patch_frames - spec.shape[1]))
        else:
            spec = spec[:, : self.stft_cfg.patch_frames]

        return complex_to_2ch(spec)  # (2, 512, 512) float32


# ---------------------------------------------------------------------------
# Reverb Simulation for Controlled Evaluation (Synthetic RIR, Real RIR, Plugin)
# ---------------------------------------------------------------------------

def synthesize_rir(
    sample_rate: int,
    rt60_sec: float = 0.8,
    drr_db: float = 3.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a realistic diffuse exponential-decay Room Impulse Response (RIR) with
    direct impulse, sparse early reflections, and frequency-damped late tail.
    """
    rng = np.random.default_rng(seed)
    rir_len = max(int(sample_rate * max(rt60_sec * 1.2, 0.25)), 1024)
    t = np.arange(rir_len, dtype=np.float64) / sample_rate

    # Exponential decay constant so envelope drops 60 dB at t = rt60_sec
    tau = max(rt60_sec, 0.05) / (3.0 * math.log(10.0))
    env = np.exp(-t / tau)

    # Late diffuse tail (Gaussian noise shaped by gentle air-absorption low-pass)
    noise = rng.standard_normal(rir_len)
    try:
        from scipy.signal import butter, filtfilt

        b, a = butter(2, min(0.85, 6500.0 / (0.5 * sample_rate)), btype="low")
        noise = filtfilt(b, a, noise)
    except Exception:
        pass

    late_start = int(0.025 * sample_rate)
    tail = np.zeros(rir_len, dtype=np.float64)
    tail[late_start:] = noise[late_start:] * env[late_start:]

    # Add 8 sparse early reflections between 5 ms and 35 ms
    for _ in range(8):
        pos = int(rng.uniform(0.005, 0.035) * sample_rate)
        if pos < rir_len:
            amp = rng.uniform(0.15, 0.45) * rng.choice([-1.0, 1.0]) * math.exp(-pos / (sample_rate * tau))
            tail[pos] += amp

    # Scale tail relative to direct impulse to match target Direct-to-Reverberant Ratio (DRR)
    tail_energy = float(np.sum(tail * tail)) + 1e-12
    target_tail_energy = 10.0 ** (-drr_db / 10.0)
    tail *= math.sqrt(target_tail_energy / tail_energy)

    rir = tail.copy()
    rir[0] += 1.0  # Direct path at t=0
    rir /= max(float(np.sqrt(np.sum(rir * rir))), 1e-8)
    return rir.astype(np.float32)


def simulate_reverb(
    dry_wav: np.ndarray,
    sample_rate: int,
    mode: str = "synthetic_rir",
    rt60_sec: float = 0.8,
    drr_db: float = 3.0,
    rir_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Create reverberant wet test audio from held-out dry audio.

    Modes:
      - "synthetic_rir": convolve with synthesized diffuse RIR (controlled RT60 & DRR)
      - "real_rir": convolve with a real licensed RIR WAV file (rir_path required)
      - "plugin": artificial algorithmic reverb via Spotify `pedalboard` (if installed)
    """
    from scipy.signal import fftconvolve

    dry = np.asarray(dry_wav, dtype=np.float32)
    dry_rms = float(np.sqrt(np.mean(dry * dry) + 1e-12))

    if mode == "plugin":
        try:
            from pedalboard import Pedalboard, Reverb  # type: ignore

            room_size = float(np.clip(rt60_sec / 2.5, 0.1, 0.95))
            wet_level = float(np.clip(1.0 / (1.0 + 10.0 ** (drr_db / 20.0)), 0.1, 0.75))
            board = Pedalboard([Reverb(room_size=room_size, damping=0.45, wet_level=wet_level, dry_level=1.0 - wet_level)])
            wet = board(dry, sample_rate).astype(np.float32)
            meta = {"reverb_mode": "plugin_pedalboard", "room_size": round(room_size, 3), "wet_level": round(wet_level, 3)}
        except ImportError:
            mode = "synthetic_rir"

    if mode == "real_rir":
        if rir_path is None or not Path(rir_path).is_file():
            raise FileNotFoundError(f"Real RIR file not found: {rir_path}")
        rir = load_mono_audio(Path(rir_path), target_sr=sample_rate)
        # Align direct peak to index 0 so wet & dry remain sample-aligned for L1 evaluation
        peak_idx = int(np.argmax(np.abs(rir)))
        rir = rir[peak_idx:]
        wet = fftconvolve(dry, rir, mode="full")[: len(dry)].astype(np.float32)
        meta = {"reverb_mode": "real_rir", "rir_file": Path(rir_path).name}

    if mode == "synthetic_rir":
        rir = synthesize_rir(sample_rate=sample_rate, rt60_sec=rt60_sec, drr_db=drr_db, seed=seed)
        wet = fftconvolve(dry, rir, mode="full")[: len(dry)].astype(np.float32)
        meta = {"reverb_mode": "synthetic_rir", "rt60_sec": rt60_sec, "drr_db": drr_db}

    # Match wet RMS to dry RMS (no clipping) so comparisons are loudness-fair
    wet_rms = float(np.sqrt(np.mean(wet * wet) + 1e-12))
    if wet_rms > 1e-8 and dry_rms > 1e-8:
        wet = wet * (dry_rms / wet_rms)
    peak = float(np.max(np.abs(wet)))
    if peak > 0.98:
        wet = wet * (0.98 / peak)
    return wet.astype(np.float32), meta
