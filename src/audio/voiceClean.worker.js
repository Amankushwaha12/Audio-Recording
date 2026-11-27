/**
 * voiceClean.worker.js — Dedicated Module Web Worker for Voice Clean DSP
 *
 * Runs the entire Stage 1 (and optional Stage 2 DeepFilterNet3 WASM) Voice Clean
 * pipeline completely off the main thread so playback, waveform scrubbing, and
 * UI controls stay 60 fps responsive. Transfers output PCM ArrayBuffers back
 * with zero-copy Transferables to keep memory overhead minimal.
 */

import { runVoiceCleanPipeline } from './voiceCleanCore.js';
import { runVoiceIsolationPipeline } from './voiceIsolation.js';
import { DeepFilterWasmDenoiser, ENABLE_AI_DENOISER } from './aiDenoiser.js';

let cancelled = false;

self.onmessage = async (event) => {
  const msg = event.data;
  if (!msg) return;

  if (msg.type === 'cancel') {
    cancelled = true;
    return;
  }

  if (msg.type !== 'run') return;
  cancelled = false;

  const { channels, sampleRate, options, wasmBinUrl } = msg;

  try {
    const wantAi = Boolean(options?.useAiDenoise || options?.useAiBoost);
    const aiDenoiseFn =
      ENABLE_AI_DENOISER && wantAi
        ? (chList, sr, opts) =>
            DeepFilterWasmDenoiser.processChannels(chList, sr, {
              ...opts,
              wasmBinUrl: wasmBinUrl || '/df_bg.wasm.bin',
            })
        : undefined;

    const pipelineFn =
      options?.mode === 'voiceIsolation'
        ? runVoiceIsolationPipeline
        : runVoiceCleanPipeline;

    const result = await pipelineFn(channels, sampleRate, options, {
      aiDenoiseFn,
      shouldAbort: () => cancelled,
      onProgress: (prog) => {
        self.postMessage({
          type: 'progress',
          percent: prog.percent,
          stage: prog.stage,
        });
      },
    });

    if (cancelled) {
      self.postMessage({ type: 'cancelled' });
      return;
    }

    const outChannels = result.channels.map((ch) =>
      ch.byteOffset === 0 && ch.byteLength === ch.buffer.byteLength
        ? ch
        : Float32Array.from(ch)
    );
    const transferList = outChannels.map((ch) => ch.buffer);

    self.postMessage(
      {
        type: 'done',
        channels: outChannels,
        sampleRate,
        report: result.report,
      },
      transferList
    );
  } catch (err) {
    if (cancelled || err?.name === 'AbortError') {
      self.postMessage({ type: 'cancelled' });
      return;
    }
    self.postMessage({
      type: 'error',
      message: err?.message || 'Unexpected Voice Clean DSP worker error.',
    });
  }
};
