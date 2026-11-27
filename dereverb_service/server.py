#!/usr/bin/env python3
"""
FastAPI Service exposing Unsupervised Diffusion Dereverberation (Saito et al., ICASSP 2023)
directly to the Mini Audio Mastering Lab web app (bypassing Vercel's ~4.5 MB serverless limit).

Endpoints:
  - GET  /health   : Reports GPU/CPU device, whether a trained checkpoint is loaded, and limits.
  - POST /dereverb : Accepts direct multipart audio upload + validated options, returns
                     16-bit PCM WAV bytes with `X-Dereverb-Metrics` JSON header.

Security safeguards:
  - Model checkpoint is loaded strictly from server-side `DEREVERB_CHECKPOINT_PATH` env var
    (never from user-supplied request paths).
  - Strict upload size cap (`MAX_UPLOAD_MB`, default 100 MB) enforced while streaming chunks.
  - Uploads written strictly inside an isolated TemporaryDirectory with UUID filenames.
  - Explicit CORS allowlist via `ALLOWED_ORIGINS` (no wildcard `*` with credentials).
  - Optional Bearer token auth via `DEREVERB_API_TOKEN` using constant-time comparison.
"""

from __future__ import annotations

import hmac
import io
import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from dereverb import load_prior_checkpoint
from dereverb_lib.config import (
    DDRMConfig,
    DereverbPipelineConfig,
    PostProcessConfig,
    RefinementConfig,
    STFTConfig,
    WPEConfig,
)
from dereverb_lib.ddrm import dereverb_waveform
from dereverb_lib.diffusion import GaussianDiffusionPrior
from dereverb_lib.unet import AudioDiffusionUNet

ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
MAX_UPLOAD_MB = max(1, min(500, int(os.getenv("MAX_UPLOAD_MB", "100"))))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
API_TOKEN = os.getenv("DEREVERB_API_TOKEN", "").strip()
CHECKPOINT_ENV_PATH = os.getenv("DEREVERB_CHECKPOINT_PATH", "").strip()


class ServiceState:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[AudioDiffusionUNet] = None
        self.diffusion: Optional[GaussianDiffusionPrior] = None
        self.stft_cfg: STFTConfig = STFTConfig()
        self.checkpoint_loaded: bool = False
        self.checkpoint_name: Optional[str] = None
        self.param_count: Optional[int] = None


state = ServiceState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if CHECKPOINT_ENV_PATH:
        ckpt_path = Path(CHECKPOINT_ENV_PATH).expanduser().resolve()
        if ckpt_path.is_file():
            model, diffusion, stft_cfg = load_prior_checkpoint(ckpt_path, device=state.device)
            state.model = model
            state.diffusion = diffusion
            state.stft_cfg = stft_cfg
            state.checkpoint_loaded = True
            state.checkpoint_name = ckpt_path.name
            state.param_count = model.count_parameters()
    yield


app = FastAPI(
    title="Diffusion Vocal Dereverberation API (Saito et al., ICASSP 2023)",
    version="1.0.0",
    lifespan=lifespan,
)

# Explicit CORS allowlist (never '*' with credentials)
raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173",
)
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["X-Dereverb-Metrics", "Content-Disposition"],
)


def verify_optional_bearer(authorization: Optional[str] = Header(default=None)) -> None:
    """Enforce Bearer auth when DEREVERB_API_TOKEN is configured on the server."""
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    provided = authorization[len("Bearer ") :].strip()
    if not hmac.compare_digest(provided.encode("utf-8"), API_TOKEN.encode("utf-8")):
        raise HTTPException(status_code=403, detail="Invalid Bearer token.")


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "device": str(state.device),
        "cuda_available": torch.cuda.is_available(),
        "checkpoint_loaded": state.checkpoint_loaded,
        "checkpoint_name": state.checkpoint_name,
        "param_count": state.param_count,
        "max_upload_mb": MAX_UPLOAD_MB,
        "auth_required": bool(API_TOKEN),
        "available_modes": (
            ["proposed", "fast", "no_refine", "wpe_baseline"]
            if state.checkpoint_loaded
            else ["wpe_baseline"]
        ),
    }


@app.post("/dereverb")
async def dereverb_endpoint(
    file: UploadFile = File(...),
    mode: Literal["proposed", "fast", "no_refine", "wpe_baseline"] = Form("fast"),
    dry_mix: float = Form(0.15, ge=0.0, le=0.5),
    match_loudness: bool = Form(True),
    ceiling_dbtp: float = Form(-1.5, ge=-6.0, le=-0.1),
    wpe_taps: int = Form(150, ge=20, le=300),
    wpe_delay: int = Form(8, ge=1, le=32),
    ddrm_steps: int = Form(20, ge=4, le=100),
    n_refine: int = Form(10000, ge=0, le=20000),
    fast_n_refine: int = Form(500, ge=0, le=5000),
    _auth: None = Depends(verify_optional_bearer),
) -> Response:
    baseline_only = mode == "wpe_baseline"
    if not baseline_only and not state.checkpoint_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained diffusion prior checkpoint is loaded on this server "
                "(set DEREVERB_CHECKPOINT_PATH). Use mode='wpe_baseline' for WPE-only dereverberation."
            ),
        )

    orig_name = Path(file.filename or "audio.wav").name
    ext = Path(orig_name).suffix.lower() or ".wav"
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Stream upload safely to an isolated temp directory while enforcing size cap
    with tempfile.TemporaryDirectory(prefix="dereverb_req_") as tmp_dir:
        in_path = Path(tmp_dir) / f"{uuid.uuid4().hex}{ext}"
        bytes_written = 0
        try:
            with open(in_path, "wb") as out_f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload exceeds server limit of {MAX_UPLOAD_MB} MB.",
                        )
                    out_f.write(chunk)
        finally:
            await file.close()

        if bytes_written == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        try:
            audio, sr = sf.read(str(in_path), dtype="float32", always_2d=False)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Could not decode uploaded audio file. Please upload standard PCM WAV, FLAC, or OGG.",
            )

        if audio.size == 0 or sr < 8000 or sr > 192000:
            raise HTTPException(status_code=400, detail="Audio file has invalid sample rate or zero length.")

        cfg = DereverbPipelineConfig(
            stft=state.stft_cfg,
            wpe=WPEConfig(taps=wpe_taps, delay=wpe_delay, iterations=1),
            ddrm=DDRMConfig(sampling_steps=ddrm_steps),
            refinement=RefinementConfig(n_refine=n_refine, fast_n_refine=fast_n_refine),
            post=PostProcessConfig(
                match_loudness=match_loudness,
                true_peak_ceiling_dbtp=ceiling_dbtp,
                dry_mix=dry_mix,
            ),
            fast=(mode == "fast"),
            skip_refinement=(mode == "no_refine"),
            baseline_wpe_only=baseline_only,
        )

        try:
            out_wav, report = dereverb_waveform(
                wet_mono=audio,
                sample_rate=sr,
                cfg=cfg,
                model=state.model,
                diffusion=state.diffusion,
                device=state.device,
            )
        except Exception:
            raise HTTPException(status_code=500, detail="Dereverberation processing failed on server.")

        buf = io.BytesIO()
        sf.write(buf, out_wav, sr, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

    header_metrics = {
        "mode": report["mode"],
        "device": report["device"],
        "cpu_warning": report["cpu_warning"],
        "audio_duration_sec": report["audio_duration_sec"],
        "num_windows": report["num_windows"],
        "elapsed_sec": report["elapsed_sec"],
        "sec_per_minute_audio": report["sec_per_minute_audio"],
        "postprocess": report["postprocess"],
    }

    safe_stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in Path(orig_name).stem)[:64] or "audio"
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_stem}_dereverb.wav"',
            "X-Dereverb-Metrics": json.dumps(header_metrics),
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("server:app", host=host, port=port, reload=False)
