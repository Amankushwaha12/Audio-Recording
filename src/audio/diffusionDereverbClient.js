/**
 * diffusionDereverbClient.js
 *
 * Direct browser-to-FastAPI client for the Saito et al. (ICASSP 2023)
 * Unsupervised Diffusion Dereverberation service (`POST /dereverb`).
 *
 * Uploads WAV directly from the browser to your GPU/Docker FastAPI host
 * (configured via VITE_DEREVERB_API_URL or overridden in the Advanced Lab panel)
 * so large audio files bypass Vercel's ~4.5 MB serverless body limit.
 */

import { encodeWav16Bit } from './processor.js';

export const DEFAULT_DIFFUSION_DEREVERB_OPTIONS = {
  apiUrl: (typeof import.meta !== 'undefined' && import.meta.env?.VITE_DEREVERB_API_URL) || '',
  mode: 'fast', // 'fast' | 'proposed' | 'no_refine' | 'wpe_baseline'
  dryMix: 0.15, // 10-20% recommended original blend
  matchLoudness: true, // +/- 0.5 LU ITU-R BS.1770 match
  ceilingDbtp: -1.5, // -1.5 dBTP true-peak ceiling
  wpeTaps: 150,
  wpeDelay: 8,
  ddrmSteps: 20,
  nRefine: 10000,
  fastNRefine: 500,
  chainMastering: false, // Audition pure dereverb by default
};

/**
 * Validate and normalize an external FastAPI base URL.
 * Rejects non-http(s) schemes (e.g. javascript:, data:, file:) and strips trailing slash.
 */
export function normalizeDereverbApiUrl(rawUrl) {
  const trimmed = String(rawUrl || '').trim();
  if (!trimmed) return '';
  let parsed;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error(
      'Invalid Dereverb API URL. Enter a full http:// or https:// server address (e.g. http://localhost:8000).'
    );
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error('Dereverb API URL must use http:// or https://.');
  }
  return `${parsed.origin}${parsed.pathname}`.replace(/\/+$/, '');
}

/**
 * Query GET /health on the FastAPI dereverberation service to inspect GPU/CPU
 * availability and whether a trained diffusion prior checkpoint is loaded.
 */
export async function checkDereverbHealth(rawBaseUrl, signal) {
  const baseUrl = normalizeDereverbApiUrl(rawBaseUrl);
  if (!baseUrl) {
    throw new Error(
      'No Dereverb API URL configured. Set VITE_DEREVERB_API_URL or enter your FastAPI server URL.'
    );
  }

  let res;
  try {
    res = await fetch(`${baseUrl}/health`, {
      method: 'GET',
      signal,
    });
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    throw new Error(
      `Could not reach ${baseUrl}/health. Ensure the FastAPI server is running and ALLOWED_ORIGINS includes ${window.location.origin}.`
    );
  }

  if (!res.ok) {
    throw new Error(`Health check returned HTTP ${res.status}.`);
  }

  const data = await res.json();
  return data;
}

/**
 * Encode an AudioBuffer to 16-bit WAV and upload directly to POST /dereverb.
 * Returns { audioBuffer, report }.
 */
export async function runDiffusionDereverb(
  audioBuffer,
  options,
  { audioCtx, signal, onProgress } = {}
) {
  const opts = { ...DEFAULT_DIFFUSION_DEREVERB_OPTIONS, ...options };
  const baseUrl = normalizeDereverbApiUrl(opts.apiUrl);
  if (!baseUrl) {
    throw new Error(
      'Diffusion Dereverb API URL is not configured. Start dereverb_service/server.py (or Docker container) and enter its URL (e.g. http://localhost:8000).'
    );
  }

  onProgress?.({ percent: 8, stage: 'Encoding 16-bit WAV for direct GPU upload...' });
  const wavBlob = encodeWav16Bit(audioBuffer);

  if (signal?.aborted) {
    const err = new Error('Processing cancelled.');
    err.name = 'AbortError';
    throw err;
  }

  const form = new FormData();
  form.append('file', wavBlob, 'input_wet.wav');
  form.append('mode', String(opts.mode || 'fast'));
  form.append('dry_mix', String(Math.max(0, Math.min(0.5, Number(opts.dryMix ?? 0.15)))));
  form.append('match_loudness', String(Boolean(opts.matchLoudness)));
  form.append(
    'ceiling_dbtp',
    String(Math.max(-6, Math.min(-0.1, Number(opts.ceilingDbtp ?? -1.5))))
  );
  form.append('wpe_taps', String(Math.round(Number(opts.wpeTaps ?? 150))));
  form.append('wpe_delay', String(Math.round(Number(opts.wpeDelay ?? 8))));
  form.append('ddrm_steps', String(Math.round(Number(opts.ddrmSteps ?? 20))));
  form.append('n_refine', String(Math.round(Number(opts.nRefine ?? 10000))));
  form.append('fast_n_refine', String(Math.round(Number(opts.fastNRefine ?? 500))));

  const modeLabel =
    opts.mode === 'wpe_baseline'
      ? 'WPE baseline (L=200, 3 iters)'
      : opts.mode === 'proposed'
        ? 'Algorithm 1 Full (DDRM + 10K refine)'
        : opts.mode === 'no_refine'
          ? 'DDRM Pass 1 (no refinement)'
          : `Algorithm 1 Fast (DDRM + ${opts.fastNRefine} refine)`;

  onProgress?.({
    percent: 24,
    stage: `Uploading (${(wavBlob.size / (1024 * 1024)).toFixed(1)} MB) & running ${modeLabel}...`,
  });

  const tStart = performance.now();
  let res;
  try {
    res = await fetch(`${baseUrl}/dereverb`, {
      method: 'POST',
      body: form,
      signal,
    });
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    throw new Error(
      `Direct upload to ${baseUrl}/dereverb failed. Verify the FastAPI service is running and ALLOWED_ORIGINS includes ${window.location.origin}.`
    );
  }

  if (!res.ok) {
    let detail = `Server returned HTTP ${res.status}.`;
    try {
      const errJson = await res.json();
      if (errJson?.detail) detail = String(errJson.detail);
    } catch {
      // Keep status message
    }
    throw new Error(detail);
  }

  onProgress?.({ percent: 84, stage: 'Decoding dereverbed WAV from server...' });
  const metricsHeader = res.headers.get('X-Dereverb-Metrics');
  let serverReport = null;
  if (metricsHeader) {
    try {
      serverReport = JSON.parse(metricsHeader);
    } catch {
      serverReport = null;
    }
  }

  const outArrayBuffer = await res.arrayBuffer();
  if (signal?.aborted) {
    const err = new Error('Processing cancelled.');
    err.name = 'AbortError';
    throw err;
  }

  if (!audioCtx) {
    throw new Error('Web Audio context is required to decode the returned WAV.');
  }
  const decoded = await audioCtx.decodeAudioData(outArrayBuffer.slice(0));
  const roundTripSec = (performance.now() - tStart) / 1000;

  onProgress?.({ percent: 96, stage: 'Finalizing dereverb analysis...' });

  return {
    audioBuffer: decoded,
    report: {
      ...(serverReport || { mode: opts.mode, device: 'remote' }),
      roundTripSec: Number(roundTripSec.toFixed(2)),
      apiUrl: baseUrl,
      uploadBytes: wavBlob.size,
    },
  };
}
