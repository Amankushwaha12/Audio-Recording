# Unsupervised Vocal Dereverberation with Diffusion-based Generative Models

Research prototype implementing **Saito et al. (Sony, ICASSP 2023, arXiv:2211.04124)** in PyTorch 2.x (Python 3.10+), exposed via a FastAPI service (`POST /dereverb`) that **Mini Audio Mastering Lab** calls directly from the browser (bypassing Vercel's ~4.5 MB serverless body limit).

---

## Method Summary

Given a reverberant STFT observation $y_{\text{wet}} = x_{\text{dry}} + x_{\text{reverb}} + z$, we train an **unsupervised diffusion prior** $p_\theta(x_{\text{dry}})$ exclusively on dry mono audio (no wet/dry pairs) in the 2-channel (`Re`, `Im`) DC-removed $512 \times 512$ complex STFT domain (`n_fft=1024`, `hop=256`, Hann window). At inference (**Algorithm 1**):

1. **Per-frequency WPE init**: Estimate complex taps $g_k$ ($L=150, D=8, 1$ iteration) and form the differentiable Toeplitz operator $A_k = \tilde{I} - G_k$ (Eq. 3–5).
2. **First DDRM pass**: Compute the per-frequency SVD of $A_k = U_A \Sigma_A V_A^H$, invert to $H_k = A_k^\dagger = V_A \Sigma_A^{-1} U_A^H$ (Eq. 6), and sample a dry estimate $\bar{x}_{\text{dry}}$ via DDRM (Kawar et al. 2022, Eq. 8; $\eta=0.7, \eta_b=0.2, \sigma_y=10^{-6}, 20$ steps).
3. **Autograd filter refinement**: Refine $g_k$ for $N_{\text{refine}} = 10{,}000$ gradient descent steps ($\alpha=10^{-6}, \lambda=1.0$) minimizing $\|\bar{x}_{\text{dry}} - (\tilde{I} - G_k) y_{\text{wet}}\|_2^2 + \lambda \|g_k\|_2^2$.
4. **Second DDRM pass**: Recompute the per-frequency SVD for $g_{k,\text{refined}}$ and run DDRM again to obtain $\tilde{x}_{\text{dry}}$, followed by 10–20% dry/wet blend, ITU-R BS.1770 loudness matching ($\pm 0.5\text{ LU}$), and a $-1.5\text{ dBTP}$ true-peak limiter.

---

## Project Structure

```text
dereverb_service/
├── dereverb_lib/
│   ├── config.py       # Table 2 & Section 4 hyperparameters + ASSUMPTION flags
│   ├── stft.py         # 1024/256 Hann complex STFT (DC dropped), crossfading, BS.1770 & limiter
│   ├── unet.py         # Modified OpenAI guided-diffusion U-Net (AdaGN + time-distributed FC)
│   ├── diffusion.py    # 4000-step cosine schedule, Eq. 9 weighted x0 MSE loss, EMA 0.9999
│   ├── wpe.py          # Pure-PyTorch per-frequency WPE solver (init & L=200/3-iter baseline)
│   ├── operator.py     # Differentiable Toeplitz (I~ - G_k), per-freq SVD, Step 3 refinement
│   ├── ddrm.py         # Exact Kawar et al. Eq. 8 DDRM sampler & Algorithm 1 long-file runner
│   ├── dataset.py      # Dry-only STFT training loader + synthetic/real RIR & plugin reverb sim
│   └── metrics.py      # Mag STFT L1, 16 kHz VGGish FAD, blind RT60/DRR, DNSMOS, Whisper WER
├── train.py            # Unsupervised dry-only prior training (batch 6, 370K iters, AMP fp16)
├── dereverb.py         # CLI: python dereverb.py in.wav out.wav [--fast | --baseline-wpe]
├── evaluate.py         # Benchmark harness vs WPE baseline & Paper Table 1 reference values
├── server.py           # FastAPI service (GET /health, POST /dereverb)
├── tests/test_operator_and_ddrm.py
├── requirements.txt
└── Dockerfile
```

---

## Commands

### 1. Install & Run Mathematical Unit Tests
```bash
cd dereverb_service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/test_operator_and_ddrm.py -v
```

### 2. Train Unsupervised Dry-Only Prior (GPU required for 370K iters)
```bash
# Singing vocals (e.g., NHSS / NUS-48E for non-commercial research) or dry speech (LibriTTS / VCTK)
python train.py \
  --data-dir /path/to/dry_audio \
  --out-dir ./checkpoints \
  --sample-rate 44100 \
  --batch-size 6 \
  --max-iters 370000 \
  --lr 1e-4 \
  --ema-rate 0.9999
```

### 3. CLI Inference
```bash
# Full Algorithm 1 (10,000 refinement steps, 20 DDRM steps)
python dereverb.py input_wet.wav output_dry.wav --checkpoint ./checkpoints/latest.pt

# Fast mode (500 refinement steps, 10 DDRM steps — faster, slightly less refined)
python dereverb.py input_wet.wav output_dry_fast.wav --checkpoint ./checkpoints/latest.pt --fast

# Baseline WPE alone (L=200, D=8, 3 iterations — works immediately with no checkpoint)
python dereverb.py input_wet.wav output_wpe.wav --baseline-wpe --dry-mix 0.15
```

### 4. Evaluation Against Baseline & Paper Reference Benchmarks
```bash
# Synthetic RIR evaluation (or pass --reverb-mode real_rir --rir-dir /path/to/rirs)
python evaluate.py \
  --dry-dir /path/to/held_out_dry \
  --checkpoint ./checkpoints/latest.pt \
  --reverb-mode synthetic_rir \
  --rt60 0.8 \
  --drr-db 3.0 \
  --out-json eval_results.json
```

### 5. Run FastAPI Server for Mini Audio Mastering Lab
```bash
# Local or GPU host (set DEREVERB_CHECKPOINT_PATH once trained; omit to serve WPE baseline)
export DEREVERB_CHECKPOINT_PATH="./checkpoints/latest.pt"
export ALLOWED_ORIGINS="http://localhost:3000,https://your-vercel-app.vercel.app"
export MAX_UPLOAD_MB=100
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then set `VITE_DEREVERB_API_URL="http://localhost:8000"` (or your HTTPS GPU host URL) in the web app `.env` so the browser uploads directly to `POST /dereverb` without passing through Vercel serverless functions.

---

## Reference Benchmarks (Saito et al., ICASSP 2023, Table 1)

*Paper results below were measured on held-out NUS-48E **singing vocals** with **FabFilter Pro-R** artificial reverb. Speech recordings with real room impulse responses are a different domain and require training on dry speech (e.g., VCTK / LibriTTS) and evaluating with DNSMOS / Whisper WER.*

| Method | Paper Mag STFT L1 (↓) | Paper 16k VGGish FAD (↓) | Local Prototype Status |
|---|---:|---:|---|
| Wet (unprocessed) | 0.114 | 13.7 | Ready in `evaluate.py` |
| WPE baseline (`L=200, D=8, 3 iters`) | 0.103 | 10.1 | Ready immediately (no checkpoint needed) |
| Proposed (no refinement) | 0.102 | 9.85 | Requires training dry-only checkpoint via `train.py` |
| Proposed + refinement (`N_refine=10000`) | 0.100 | 9.69 | Requires training dry-only checkpoint via `train.py` |
| Oracle operator + DDRM | 0.079 | 4.44 | Supported in `evaluate.py` once checkpoint is trained |

---

## Licenses Audit (Dependencies & Datasets)

| Component / Dataset | License | Commercial Use Notes |
|---|---|---|
| `bahjat-kawar/ddrm` (reference algorithm) | MIT | Permissive |
| `openai/guided-diffusion` (U-Net reference) | MIT | Permissive |
| PyTorch, SciPy, NumPy, soundfile | BSD-3-Clause | Permissive |
| FastAPI, Uvicorn, pyloudnorm, pytest | MIT / BSD | Permissive |
| `frechet_audio_distance` (optional eval) | Apache-2.0 | Permissive |
| `faster-whisper` / `jiwer` (optional eval) | MIT / Apache-2.0 | Permissive |
| `pedalboard` (optional plugin reverb eval) | **GPLv3** | Keep optional / eval-only if avoiding GPL |
| **NHSS** (paper singing training set) | **Non-commercial research only** | Request access; not for commercial deployment |
| **NUS-48E** (paper singing eval set) | **Non-commercial research only** | Research benchmark only |
| **LibriTTS / LibriSpeech / VCTK** (speech alt) | **CC BY 4.0** | Commercial use permitted with attribution |
| **OpenSLR SLR28 RIRs** (real RIR eval alt) | **Apache-2.0** | Permissive real room impulse responses |

---

## Explicit List of `ASSUMPTION` Items

1. **STFT padding & scaling (`stft.py`)**: Uses PyTorch `center=True`, `normalized=True` ($1/\sqrt{1024}$), and a checkpoint-stored `stft_scale` (default `1.0`) plus optional `-20 dBFS` RMS crop normalization during training (`dataset.py`) because the paper does not report exact STFT normalization constants.
2. **Time-distributed FC layer structure (`unet.py`)**: Implemented as `GroupNorm32 -> SiLU -> zero-init Linear(F_l, F_l)` shared across channels and applied independently per time frame at the end of every ResBlock (residual connection), as the paper specifies its placement without printing an explicit layer diagram.
3. **Eq. 9 loss weight $\gamma_t$ (`diffusion.py`)**: Defaults to clamped SNR weighting $\min(\text{SNR}(t), 5.0)$ (configurable via `--gamma-mode uniform|snr|snr_clamped`), since Eq. 9 writes $\gamma_t$ generally without a numeric table.
4. **Complex-to-2ch noise variance matching (`ddrm.py`)**: Complex Gaussian samples inside DDRM use variance `1.0` per real/imaginary component (matching i.i.d. $\mathcal{N}(0,1)$ noise on each real channel during DDPM training; unitary $V_H$ preserves this distribution).
5. **WPE precision & regularization (`wpe.py`)**: Solves the $L \times L$ normal equations in `complex128` with trace-scaled ridge `1e-6`, relative variance floor `1e-6 * mean_power`, and `psd_context=0`.
6. **SVD singular value floor & chunking (`operator.py`)**: Clamps $\sigma_{A,i} \ge 10^{-5}$ before inverting $\Sigma_H = \Sigma_A^{-1}$ and chunks `torch.linalg.svd` across 32 frequency bins at a time to bound peak VRAM.
7. **Long-audio windowing & filter handling (`stft.py`, `ddrm.py`)**: Audio longer than 512 STFT frames (~2.97 s @ 44.1 kHz) is tiled with 50% overlap (`hop=256` frames). Each window pulls causal left-context frames from the full wet STFT, estimates & refines its own per-window $g_k$, and stitches reconstructed time-domain chunks with raised-cosine crossfades.
8. **`--fast` mode trade-off (`config.py`, `ddrm.py`)**: Defaults to `fast_n_refine=500` (vs 10,000) and `fast_ddrm_steps=10` (vs 20). Cuts inference time substantially at the expense of less converged filter refinement and coarser reverse diffusion.
