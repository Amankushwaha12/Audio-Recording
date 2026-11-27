"""
Evaluation metrics for Saito et al. (ICASSP 2023) & Speech Dereverberation:

Paper metrics:
  - Magnitude spectrogram L1 loss vs dry reference
  - 16 kHz VGGish Fréchet Audio Distance (FAD) (requires optional `frechet_audio_distance`)

Speech metrics:
  - Blind RT60 & Direct-to-Reverberant Ratio (DRR) estimates
  - Optional no-reference DNSMOS P.835 (when ONNX models are configured)
  - Optional Whisper WER (when `faster-whisper` or `openai-whisper` + `jiwer` are installed)

Oracle operator helper:
  - Solves exact least-squares per-frequency filter g_k* minimizing
    || x_dry - (I~ - G_k) y_wet ||_2^2 + eps ||g_k||_2^2 for Table 1 Oracle sanity-check.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import STFTConfig
from .stft import compute_stft
from .wpe import _build_delayed_matrix

# Paper Table 1 reference numbers (NUS-48E singing vocals + FabFilter Pro-R artificial reverb)
PAPER_TABLE1_REFERENCE = {
    "wet": {"mag_stft_l1": 0.114, "vggish_fad_16k": 13.7},
    "wpe_baseline": {"mag_stft_l1": 0.103, "vggish_fad_16k": 10.1},
    "proposed_no_refine": {"mag_stft_l1": 0.102, "vggish_fad_16k": 9.85},
    "proposed_with_refine": {"mag_stft_l1": 0.100, "vggish_fad_16k": 9.69},
    "oracle": {"mag_stft_l1": 0.079, "vggish_fad_16k": 4.44},
}


def compute_mag_stft_l1(
    est_wav: np.ndarray,
    ref_wav: np.ndarray,
    stft_cfg: STFTConfig,
) -> float:
    """
    Magnitude spectrogram L1 loss between estimated dry waveform and dry reference:
        mean(| |STFT(est)| - |STFT(ref)| |)
    """
    min_len = min(len(est_wav), len(ref_wav))
    est_t = torch.from_numpy(np.asarray(est_wav[:min_len], dtype=np.float32))
    ref_t = torch.from_numpy(np.asarray(ref_wav[:min_len], dtype=np.float32))
    est_spec = compute_stft(est_t, stft_cfg)
    ref_spec = compute_stft(ref_t, stft_cfg)
    l1 = torch.abs(torch.abs(est_spec) - torch.abs(ref_spec)).mean()
    return float(l1.item())


def estimate_blind_rt60_and_drr(
    audio: np.ndarray,
    sample_rate: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Blind estimation of late-decay RT60 (seconds) and approximate DRR proxy (dB)
    from voiced-to-pause energy decays in the broadband speech band (300-4000 Hz).

    Note: Blind single-channel RT60/DRR is an approximation; for exact ground truth,
    measure directly on known RIRs during synthetic/real-RIR evaluation.
    """
    x = np.asarray(audio, dtype=np.float64)
    if len(x) < int(0.5 * sample_rate):
        return None, None

    try:
        from scipy.signal import butter, filtfilt

        nyq = 0.5 * sample_rate
        lo = max(0.01, min(0.45, 300.0 / nyq))
        hi = max(lo + 0.05, min(0.95, 4000.0 / nyq))
        b, a = butter(3, [lo, hi], btype="band")
        x = filtfilt(b, a, x)
    except Exception:
        pass

    frame_len = max(1, int(round(0.02 * sample_rate)))  # 20 ms
    hop = max(1, frame_len // 2)  # 10 ms
    n_frames = max(1, (len(x) - frame_len) // hop + 1)
    env_db = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop : i * hop + frame_len]
        env_db[i] = 10.0 * math.log10(float(np.mean(seg * seg)) + 1e-12)

    p90 = float(np.percentile(env_db, 90))
    p15 = float(np.percentile(env_db, 15))
    if p90 - p15 < 12.0:
        return None, None

    # Find monotonic decay segments starting near local peaks (>= p90 - 6 dB)
    slopes_db_per_sec: List[float] = []
    drr_estimates: List[float] = []
    dt = hop / sample_rate

    i = 2
    while i < n_frames - 12:
        if env_db[i] >= p90 - 8.0 and env_db[i] >= env_db[i - 1] and env_db[i] > env_db[i + 1]:
            # Track downward decay up to 350 ms
            j = i + 1
            max_j = min(n_frames - 1, i + int(0.35 / dt))
            while j < max_j and env_db[j] <= env_db[j - 1] + 1.5 and env_db[j] > p15 + 3.0:
                j += 1
            seg_len = j - i
            if seg_len >= 6 and (env_db[i] - env_db[j - 1]) >= 8.0:
                t_axis = np.arange(seg_len, dtype=np.float64) * dt
                y_seg = env_db[i:j]
                slope, _ = np.polyfit(t_axis, y_seg, 1)
                if slope < -12.0:  # Valid decay (RT60 <= 5.0 s)
                    slopes_db_per_sec.append(float(slope))
                    # Approximate DRR proxy: direct 0-20ms energy vs tail 30-150ms energy
                    early_db = float(np.mean(env_db[i : min(j, i + 2)]))
                    late_slice = env_db[min(j - 1, i + 3) : min(j, i + 15)]
                    if len(late_slice) > 0:
                        drr_estimates.append(early_db - float(np.mean(late_slice)))
            i = j
        else:
            i += 1

    if not slopes_db_per_sec:
        return None, None

    med_slope = float(np.median(slopes_db_per_sec))
    rt60 = float(np.clip(-60.0 / med_slope, 0.05, 4.0))
    drr = float(np.median(drr_estimates)) if drr_estimates else None
    return round(rt60, 3), (round(drr, 2) if drr is not None else None)


def compute_vggish_fad_16k(ref_dir: Path, eval_dir: Path) -> Optional[float]:
    """
    Compute 16 kHz VGGish Fréchet Audio Distance (FAD) between two directories of WAV files.
    Requires optional `frechet_audio_distance` package (Apache-2.0). Returns None if not installed.
    """
    try:
        from frechet_audio_distance import FrechetAudioDistance  # type: ignore

        fad = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,
            use_pca=False,
            use_activation=False,
            verbose=False,
        )
        score = fad.score(str(ref_dir), str(eval_dir), dtype="float32")
        return float(score)
    except Exception:
        return None


def compute_dnsmos_p835(
    wav_16k: np.ndarray,
    onnx_model_path: Optional[Path] = None,
) -> Optional[Dict[str, float]]:
    """
    Optional no-reference DNSMOS P.835 scoring (SIG, BAK, OVRL).
    Requires `onnxruntime` and Microsoft's public DNSMOS P.835 ONNX model file
    (CC BY 4.0 / MIT via Microsoft DNS-Challenge repo) supplied via `onnx_model_path`.
    Returns None cleanly if unavailable.
    """
    if onnx_model_path is None or not Path(onnx_model_path).is_file():
        return None
    try:
        import onnxruntime as ort  # type: ignore

        sess = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
        seg_len = 144160  # 9.01 s @ 16 kHz (standard DNSMOS P.835 input length)
        x = np.asarray(wav_16k, dtype=np.float32)
        if len(x) < seg_len:
            reps = int(math.ceil(seg_len / max(1, len(x))))
            x = np.tile(x, reps)[:seg_len]
        else:
            x = x[:seg_len]
        inp_name = sess.get_inputs()[0].name
        outs = sess.run(None, {inp_name: x[None, :]})[0][0]
        # Standard DNSMOS P.835 polynomial calibration
        sig = float(outs[0])
        bak = float(outs[1])
        ovrl = float(outs[2])
        return {"sig": round(sig, 3), "bak": round(bak, 3), "ovrl": round(ovrl, 3)}
    except Exception:
        return None


def compute_whisper_wer(
    audio_path: Path,
    reference_transcript: str,
    model_size: str = "base",
) -> Optional[float]:
    """
    Optional Whisper Word Error Rate (WER) for speech dereverberation evaluation.
    Requires optional `faster-whisper` (or `openai-whisper`) and `jiwer`.
    """
    if not reference_transcript.strip():
        return None
    try:
        from jiwer import wer  # type: ignore
    except ImportError:
        return None

    hypothesis = None
    try:
        from faster_whisper import WhisperModel  # type: ignore

        wm = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = wm.transcribe(str(audio_path))
        hypothesis = " ".join(s.text.strip() for s in segments).strip()
    except ImportError:
        try:
            import whisper  # type: ignore

            wm = whisper.load_model(model_size)
            res = wm.transcribe(str(audio_path))
            hypothesis = str(res.get("text", "")).strip()
        except ImportError:
            return None

    if hypothesis is None:
        return None
    return float(wer(reference_transcript.lower(), hypothesis.lower()))


@torch.no_grad()
def estimate_oracle_filter(
    y_spec: torch.Tensor,
    x_dry_spec: torch.Tensor,
    taps: int = 150,
    delay: int = 8,
    ridge_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Solve the oracle per-frequency dereverberation filter g_k* directly from paired
    (y_wet, x_dry) minimizing || x_dry - (y_wet - sum_l g_{l,k}^* y_wet[t - D - l + 1]) ||_2^2
    + ridge * ||g_k||_2^2.

    Used strictly for evaluation sanity-checks against Paper Table 1 ("Oracle" row).
    """
    y = y_spec.to(torch.complex128)
    x = x_dry_spec.to(torch.complex128)
    F_bins, T = y.shape
    Y_tilde = _build_delayed_matrix(y, taps=taps, delay=delay)  # (F, T, L)
    reverb_target = y - x  # (F, T)

    eye_L = torch.eye(taps, dtype=torch.complex128, device=y.device).unsqueeze(0)
    R = torch.einsum("fti,ftj->fij", Y_tilde, Y_tilde.conj())
    p = torch.einsum("fti,ft->fi", Y_tilde, reverb_target.conj())
    trace_scale = torch.diagonal(R, dim1=-2, dim2=-1).real.mean(dim=-1, keepdim=True)
    ridge = (ridge_eps * torch.clamp(trace_scale, min=1e-12)).unsqueeze(-1)
    g_oracle = torch.linalg.solve(R + ridge * eye_L, p.unsqueeze(-1)).squeeze(-1)
    return g_oracle.to(y_spec.dtype)
