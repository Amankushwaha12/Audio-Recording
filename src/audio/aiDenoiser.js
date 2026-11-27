/**
 * aiDenoiser.js — Stage 2 Optional AI Denoiser (DeepFilterNet3) Adapter
 *
 * Feature flag:
 *   ENABLE_AI_DENOISER = true because `@lofcz/deepfilternet-web` is a maintained
 *   48 kHz WASM SIMD build of DeepFilterNet3 (permissive MIT/Apache-2.0 license)
 *   that runs ~2–3x faster than real-time on CPU in modern browsers, and is loaded
 *   lazily from `/df_bg.wasm.bin` (~12.1 MB gzipped) only when the user enables it.
 *
 * Clean Adapter Interface:
 *   interface AiDenoiser {
 *     readonly id: string;
 *     readonly label: string;
 *     processChannels(channels: Float32Array[], sampleRate: number, opts?: object): Promise<Float32Array[]>;
 *     process?(buffer: AudioBuffer, opts?: object): Promise<AudioBuffer>;
 *   }
 *
 * Includes BOTH:
 *   1. `DeepFilterWasmDenoiser` — runs locally inside the Voice Clean Web Worker
 *      (lazy WASM compilation, 48 kHz linear resampling if input != 48 kHz, exact
 *      3-frame lookahead latency compensation, zero-phase vocal integrity guard).
 *   2. `ExternalFastApiDenoiser` — optional adapter stub for calling an external
 *      Python FastAPI `POST /clean` DeepFilterNet service directly from the client
 *      (bypassing Vercel's ~4.5 MB serverless function body limit).
 */

import { DeepFilter, SAMPLE_RATE as DF_SAMPLE_RATE } from '@lofcz/deepfilternet-web';
import { dbToLinear, linearToDb } from './analyzer.js';

export const ENABLE_AI_DENOISER = true;

let cachedModulePromise = null;

/**
 * Lazily fetches `/df_bg.wasm.bin`, decompresses gzip, and compiles a reusable
 * WebAssembly.Module (works identically inside Web Workers and main thread).
 */
async function loadDeepFilterModule(wasmBinUrl, onProgress) {
  if (!cachedModulePromise) {
    cachedModulePromise = (async () => {
      onProgress?.({ percent: 5, stage: 'AI Denoise: Downloading DeepFilterNet3 weights (lazy)...' });
      const res = await fetch(wasmBinUrl, { cache: 'force-cache' });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status} loading DeepFilterNet3 binary`);
      }
      const ct = res.headers.get('Content-Type') || '';
      if (ct.includes('text/html')) {
        throw new Error('Unexpected HTML payload instead of DeepFilterNet3 WASM binary.');
      }

      const rawBytes = new Uint8Array(await res.arrayBuffer());
      let wasmBytes = rawBytes;

      // Decompress gzipped payload natively in browser/worker
      if (rawBytes.byteLength >= 2 && rawBytes[0] === 0x1f && rawBytes[1] === 0x8b) {
        if (typeof DecompressionStream === 'undefined') {
          throw new Error('Browser lacks DecompressionStream support for DeepFilterNet3.');
        }
        onProgress?.({ percent: 15, stage: 'AI Denoise: Decompressing & compiling WASM SIMD...' });
        const ds = new DecompressionStream('gzip');
        const stream = new Blob([rawBytes]).stream().pipeThrough(ds);
        wasmBytes = new Uint8Array(await new Response(stream).arrayBuffer());
      }

      if (
        wasmBytes.byteLength < 8 ||
        wasmBytes[0] !== 0x00 ||
        wasmBytes[1] !== 0x61 ||
        wasmBytes[2] !== 0x73 ||
        wasmBytes[3] !== 0x6d
      ) {
        throw new Error('Invalid DeepFilterNet3 WebAssembly header.');
      }

      return await WebAssembly.compile(wasmBytes);
    })().catch((err) => {
      cachedModulePromise = null;
      throw err;
    });
  }
  return await cachedModulePromise;
}

/**
 * Pure-JS Worker-safe cubic Hermite resampler (needed when input sampleRate !== 48 kHz
 * because Web Workers do not have OfflineAudioContext).
 */
function resampleFloat32(input, fromRate, toRate, targetLength) {
  if (fromRate === toRate && (!targetLength || targetLength === input.length)) {
    return Float32Array.from(input);
  }
  const outLen = targetLength || Math.max(1, Math.round((input.length * toRate) / fromRate));
  const out = new Float32Array(outLen);
  const ratio = fromRate / toRate;
  const maxIdx = input.length - 1;

  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const t = pos - idx;
    const y0 = input[Math.max(0, idx - 1)];
    const y1 = input[Math.min(maxIdx, idx)];
    const y2 = input[Math.min(maxIdx, idx + 1)];
    const y3 = input[Math.min(maxIdx, idx + 2)];
    const t2 = t * t;
    const t3 = t2 * t;
    out[i] =
      0.5 *
      (2 * y1 +
        (-y0 + y2) * t +
        (2 * y0 - 5 * y1 + 4 * y2 - y3) * t2 +
        (-y0 + 3 * y1 - 3 * y2 + y3) * t3);
  }
  return out;
}

function isDualMono(channels) {
  if (channels.length !== 2) return false;
  const l = channels[0];
  const r = channels[1];
  const stride = Math.max(1, Math.floor(l.length / 4000));
  for (let i = 0; i < l.length; i += stride) {
    if (Math.abs(l[i] - r[i]) > 1e-4) return false;
  }
  return true;
}

/**
 * Stage 2 In-Browser DeepFilterNet3 WASM Denoiser (Worker-safe).
 * Replaces only the spectral noise step inside Voice Clean while preserving
 * exact sample rate, duration, stereo coherence, and active-speech integrity.
 */
export const DeepFilterWasmDenoiser = {
  id: 'deepfilternet-wasm',
  label: 'DeepFilterNet3 (Browser WASM SIMD)',

  async processChannels(channels, sampleRate, opts = {}) {
    const {
      noiseReduction = 60,
      wasmBinUrl = '/df_bg.wasm.bin',
      onProgress,
      shouldAbort,
    } = opts;

    const checkAbort = () => {
      if (shouldAbort?.()) {
        const err = new Error('AI denoise cancelled.');
        err.name = 'AbortError';
        throw err;
      }
    };

    const wasmModule = await loadDeepFilterModule(wasmBinUrl, onProgress);
    checkAbort();

    const numChannels = channels.length;
    const origLength = channels[0].length;
    // Map 0–100% slider to a speech-safe DeepFilterNet3 attenuation cap (12–48 dB)
    const attenuationLimit = Math.round(12 + (Math.max(0, Math.min(100, noiseReduction)) / 100) * 36);

    const dualMono = isDualMono(channels);
    const toProcess = dualMono ? 1 : numChannels;
    const outChannels = [];

    for (let ch = 0; ch < toProcess; ch++) {
      checkAbort();
      const ch48 =
        sampleRate === DF_SAMPLE_RATE
          ? channels[ch]
          : resampleFloat32(channels[ch], sampleRate, DF_SAMPLE_RATE);

      const df = await DeepFilter.create({
        wasmUrl: wasmModule,
        attenuationLimit,
      });

      try {
        df.setAttenuationLimit(attenuationLimit);
        df.setPostFilterBeta(0.0); // Speech-safe: avoids underwater consonant artifacts

        const frameLen = df.frameLength; // 480 samples (10 ms @ 48 kHz)
        const lookaheadSamples = 3 * frameLen; // Exact 0-sample latency alignment
        const len48 = ch48.length;
        const paddedLen = len48 + lookaheadSamples;
        const paddedOut = new Float32Array(paddedLen);
        const frameBuf = new Float32Array(frameLen);
        const totalFrames = Math.ceil(paddedLen / frameLen);

        for (let f = 0; f < totalFrames; f++) {
          if (f % 60 === 0) {
            checkAbort();
            const frac = (ch + f / Math.max(1, totalFrames)) / toProcess;
            onProgress?.({
              percent: Math.round(20 + frac * 75),
              stage: `AI Denoise (DeepFilterNet3): ${Math.round(frac * 100)}%...`,
            });
          }

          const offset = f * frameLen;
          frameBuf.fill(0);
          if (offset < len48) {
            const copyLen = Math.min(frameLen, len48 - offset);
            frameBuf.set(ch48.subarray(offset, offset + copyLen), 0);
          }
          const cleanFrame = df.process(frameBuf);
          paddedOut.set(
            cleanFrame.subarray(0, Math.min(frameLen, paddedLen - offset)),
            offset
          );
        }

        const aligned48 = paddedOut.subarray(lookaheadSamples, lookaheadSamples + len48);

        // Quick 0-sample phase-locked speech-integrity guard prevents sudden mid-word drops
        const numBlocks = Math.ceil(len48 / frameLen);
        const inDb = new Float32Array(numBlocks);
        const dfDb = new Float32Array(numBlocks);
        const active = [];
        const room = [];

        for (let b = 0; b < numBlocks; b++) {
          const s = b * frameLen;
          const e = Math.min(len48, s + frameLen);
          let iSq = 0;
          let dSq = 0;
          for (let i = s; i < e; i++) {
            iSq += ch48[i] * ch48[i];
            dSq += aligned48[i] * aligned48[i];
          }
          const cnt = Math.max(1, e - s);
          const id = linearToDb(Math.sqrt(iSq / cnt));
          const dd = linearToDb(Math.sqrt(dSq / cnt));
          inDb[b] = id;
          dfDb[b] = dd;
          if (id > -55) active.push(id);
          if (id > -72) room.push(id);
        }

        const guarded48 = Float32Array.from(aligned48);
        if (active.length > 5 && room.length > 5) {
          active.sort((a, b) => a - b);
          room.sort((a, b) => a - b);
          const speechRef = active[Math.floor(active.length * 0.85)];
          const floorRef = room[Math.floor(room.length * 0.15)];

          if (speechRef - floorRef >= 8) {
            const coreThresh = Math.max(floorRef + 11, speechRef - 22);
            const floorGuard = Math.max(floorRef + 4, speechRef - 32);
            const core = new Uint8Array(numBlocks);
            for (let b = 0; b < numBlocks; b++) {
              if (inDb[b] >= coreThresh) core[b] = 1;
            }
            // Close mid-word gaps up to 360 ms
            let lastIdx = -1;
            for (let b = 0; b < numBlocks; b++) {
              if (core[b]) {
                if (lastIdx >= 0 && b - lastIdx <= 36) {
                  for (let k = lastIdx + 1; k < b; k++) core[k] = 1;
                }
                lastIdx = b;
              }
            }
            const targetMix = new Float32Array(numBlocks);
            for (let b = 0; b < numBlocks; b++) {
              if (!core[b]) continue;
              const drop = inDb[b] - dfDb[b];
              if (inDb[b] >= floorGuard && drop > 2.0) {
                const desired = dbToLinear(-1.6);
                const cur = dbToLinear(-drop);
                targetMix[b] = Math.max(
                  0,
                  Math.min(0.92, (desired - cur) / Math.max(0.04, 1 - cur))
                );
              }
            }
            const atk = Math.exp(-1 / (0.008 * DF_SAMPLE_RATE));
            const rel = Math.exp(-1 / (0.07 * DF_SAMPLE_RATE));
            let mix = 0;
            for (let i = 0; i < len48; i++) {
              const tgt = targetMix[Math.min(numBlocks - 1, Math.floor(i / frameLen))];
              mix = tgt > mix ? atk * mix + (1 - atk) * tgt : rel * mix + (1 - rel) * tgt;
              guarded48[i] = guarded48[i] * (1 - mix) + ch48[i] * mix;
            }
          }
        }

        // Resample back to exact original sampleRate & sample length
        const restored =
          sampleRate === DF_SAMPLE_RATE
            ? guarded48
            : resampleFloat32(guarded48, DF_SAMPLE_RATE, sampleRate, origLength);

        outChannels.push(restored);
      } finally {
        df.destroy();
      }
    }

    if (dualMono) {
      outChannels.push(Float32Array.from(outChannels[0]));
    }

    return outChannels;
  },

  /**
   * Standard `AiDenoiser.process(buffer, opts)` method for main-thread callers.
   * Copies channels without mutating the input `AudioBuffer`.
   */
  async process(audioBuffer, opts = {}) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const copies = [];
    for (let ch = 0; ch < numChannels; ch++) {
      copies.push(Float32Array.from(audioBuffer.getChannelData(ch)));
    }
    const cleaned = await this.processChannels(copies, sampleRate, opts);
    const outLen = Math.max(1, cleaned[0]?.length || audioBuffer.length);
    const offlineCtx = new OfflineAudioContext(numChannels, outLen, sampleRate);
    const outBuf = offlineCtx.createBuffer(numChannels, outLen, sampleRate);
    for (let ch = 0; ch < numChannels; ch++) {
      outBuf.copyToChannel(cleaned[ch], ch);
    }
    return outBuf;
  },
};

/**
 * Optional External Python FastAPI Adapter Stub.
 * Demonstrates how to connect an external DeepFilterNet service (`POST /clean`)
 * directly from the browser (avoiding Vercel's ~4.5 MB serverless body limit).
 * See README.md for the FastAPI server snippet.
 */
export function createExternalFastApiDenoiser(endpointUrl) {
  return {
    id: 'external-deepfilter-fastapi',
    label: 'External DeepFilterNet API',
    async process(audioBuffer, opts = {}) {
      if (!endpointUrl) {
        throw new Error(
          'External DeepFilterNet endpoint URL is not configured. See README.md for FastAPI setup.'
        );
      }
      // Browser uploads WAV directly to the external FastAPI server (not via Vercel API routes)
      const { encodeWav16Bit } = await import('./processor.js');
      const wavBlob = encodeWav16Bit(audioBuffer);
      const form = new FormData();
      form.append('file', wavBlob, 'input.wav');
      if (opts.attenuationLimit != null) {
        form.append('atten_lim_db', String(opts.attenuationLimit));
      }

      const res = await fetch(`${endpointUrl.replace(/\/$/, '')}/clean`, {
        method: 'POST',
        body: form,
        signal: opts.signal,
      });
      if (!res.ok) {
        throw new Error(`External DeepFilterNet API failed (HTTP ${res.status})`);
      }
      const cleanedArrayBuffer = await res.arrayBuffer();
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = opts.audioContext || new AudioCtx();
      return await ctx.decodeAudioData(cleanedArrayBuffer);
    },
  };
}
