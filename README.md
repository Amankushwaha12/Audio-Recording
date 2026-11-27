# Mini Audio Mastering Lab

Local, browser-only audio analysis, **Voice Clean**, and mastering workbench built with Vite, React, and the Web Audio API. Zero backend required for core processing, zero login, and all PCM processing runs locally on your machine.

## Voice Clean Pipeline (Stage 1 — Browser Web Worker)

Voice Clean runs off the main thread in `src/audio/voiceClean.worker.js` (with chunked 16 s / 80 ms crossfaded processing and instant **Cancel** support) **before** the existing compressor, auto-leveler, and loudness stages:

1. **Pre-Clean High-Pass (75 Hz, 12 dB/oct Butterworth):** Gently removes sub-bass rumble and handling bumps without thinning vocal warmth.
2. **Spectral Noise Reduction (2048-pt STFT, 75% overlap, Hann window):**
   - Automatically estimates the noise profile from the quietest ~10% of frames, or from a region you drag-select on the Original waveform (**Use selected region as noise sample** / `Shift + Drag`).
   - Uses stereo-linked decision-directed Wiener gain with temporal and frequency smoothing (no watery "musical noise") and a strict gain floor (6–18 dB max attenuation mapped from 0–100%).
3. **Late-Reverberation Suppression:** Estimates approximate RT60 decay rate in the STFT domain and suppresses predicted late-reverb tails with a safe gain floor. Automatically skips and notifies you when the recording is already dry (`RT60 < 0.22 s`).
4. **Natural-Voice Blend ("Keep natural voice", default 15%):** Mixes time-aligned pre-cleaned dry speech back in to preserve natural breath and prevent robotic artifacts.
5. **Gentle Dynamic De-Esser (5–9 kHz):** Attenuates sharp sibilance spikes only when they exceed threshold.
6. **Gentle Tone Shaping:** `-2.0 dB` at `250 Hz` (mud) and `+1.5 dB` at `3.5 kHz` (presence) — strictly within `±3 dB`.
7. **Loudness Match (Default ON) + True-Peak Limiter (`-1.5 dBTP`):** Matches output integrated LUFS within `±0.5 LU` of the input so cleaned audio never sounds artificially "better" just by being louder. Selecting any LUFS target preset (e.g., Podcast `-16 LUFS`) overrides loudness matching and targets that loudness instead.

### One-Click Voice Clean Presets
- **Natural:** Noise `40%`, Reverb `20%`, Keep natural voice `25%`
- **Clear (Default):** Noise `60%`, Reverb `40%`, Keep natural voice `15%`
- **Strong:** Noise `80%`, Reverb `65%`, Keep natural voice `10%`

## Stage 2: DeepFilterNet AI Denoise (Browser WASM + Optional FastAPI Service)

### Built-In Browser WASM (Enabled Behind Feature Flag)
This app ships with `@lofcz/deepfilternet-web` (MIT/Apache-2.0, 48 kHz DeepFilterNet3 compiled to WASM SIMD). When **AI denoise (DeepFilterNet)** is checked in Voice Clean:
- The WASM binary (`/df_bg.wasm.bin`, gzipped ~12 MB) is **lazy-loaded inside the Web Worker** only when you process with AI denoise enabled, keeping initial page load fast.
- It replaces only Step 2 (spectral noise reduction) while keeping reverb suppression, natural-voice blend, de-esser, gentle tone shaping, and loudness match intact.

### Optional External Python DeepFilterNet Service (`POST /clean`)
If you prefer offloading heavy multi-minute recordings to a dedicated Python GPU/CPU server, `src/audio/aiDenoiser.js` exports `createExternalFastApiDenoiser(endpointUrl)` implementing the `AiDenoiser` interface.

**Important for Vercel deployments:** Upload directly from the browser to your external FastAPI service rather than proxying through a Vercel Serverless Function, because Vercel functions enforce a ~4.5 MB request body limit.

Example minimal FastAPI server (`server_df.py`):

```python
import io
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from df.enhance import enhance, init_df

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-vercel-app.vercel.app", "http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

model, df_state, _ = init_df()  # Loads DeepFilterNet3 once at startup

@app.post("/clean")
async def clean_audio(file: UploadFile = File(...)):
    raw = await file.read()
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    enhanced = enhance(model, df_state, audio)
    out = io.BytesIO()
    sf.write(out, enhanced, sr, format="WAV", subtype="PCM_16")
    return Response(content=out.getvalue(), media_type="audio/wav")
```

Run with `uvicorn server_df:app --host 0.0.0.0 --port 8000`, and wire the endpoint URL into `createExternalFastApiDenoiser("https://your-df-service.example.com/clean")`.

## Quick Tuning Guide
- **Sounds slightly robotic or watery:** Raise **Keep natural voice** (`20–30%`) or lower **Noise reduction** by `10–20%`.
- **Background hiss or hum still audible:** Raise **Noise reduction**, or click **Use selected region as noise sample** (`Shift + Drag` over a quiet pause on the Original waveform), or enable **AI denoise (DeepFilterNet)**.
- **Still sounds echoey / roomy:** Raise **Reverb reduction** (`50–75%`).
- **Too sharp / sibilant:** Enable **Smooth Tone** in the mastering rack below Voice Clean.

## Verification & Tests

Run the automated DSP test suite in Node:

```bash
node tests/voiceClean.test.mjs
```

The test suite verifies:
1. Output LUFS stays within `±0.5 LU` of input with Loudness Match ON.
2. Zero clipping (`0` samples `>= 0.999` and true peak `<= -1.4 dBTP`).
3. Exact duration and sample rate preserved (mono and stereo) without mutating the original buffer.
4. Silent input never crashes and emits a clear warning.
5. Clean dry recordings auto-skip reverb suppression and are not degraded.
