/**
 * voiceCleanClient.js — Main-Thread Orchestrator for Voice Clean Web Worker
 *
 * Copies PCM channel arrays out of the input `AudioBuffer` (never modifying the
 * original buffer), transfers them to `voiceClean.worker.js`, streams progress
 * updates back to React, and supports instant cancellation via `AbortSignal`.
 */

import { runVoiceCleanPipeline } from './voiceCleanCore.js';
import { runVoiceIsolationPipeline } from './voiceIsolation.js';

/**
 * Runs Voice Clean off the main thread in a dedicated Web Worker.
 *
 * @param {AudioBuffer} inputBuffer - Original AudioBuffer (never mutated)
 * @param {object} options - Voice Clean options
 * @param {object} [params]
 * @param {(prog: { percent: number, stage: string }) => void} [params.onProgress]
 * @param {AbortSignal} [params.signal]
 * @returns {Promise<{ audioBuffer: AudioBuffer, report: object }>}
 */
export async function runVoiceCleanInWorker(
  inputBuffer,
  options,
  { onProgress, signal } = {}
) {
  if (!inputBuffer || inputBuffer.length === 0) {
    throw new Error('Cannot clean an empty audio buffer.');
  }

  if (signal?.aborted) {
    const err = new Error('Voice Clean cancelled.');
    err.name = 'AbortError';
    throw err;
  }

  const numChannels = inputBuffer.numberOfChannels;
  const sampleRate = inputBuffer.sampleRate;
  const length = inputBuffer.length;

  // Always copy input channels so transferring to the Worker never detaches or alters originalBuffer
  const copies = [];
  for (let ch = 0; ch < numChannels; ch++) {
    copies.push(Float32Array.from(inputBuffer.getChannelData(ch)));
  }

  const rebuildAudioBuffer = (outChannels) => {
    const outLen = Math.max(1, outChannels[0]?.length || length);
    const offlineCtx = new OfflineAudioContext(numChannels, outLen, sampleRate);
    const outBuf = offlineCtx.createBuffer(numChannels, outLen, sampleRate);
    for (let ch = 0; ch < numChannels; ch++) {
      outBuf.copyToChannel(outChannels[ch], ch);
    }
    return outBuf;
  };

  const runLocalFallback = () => {
    const fn =
      options?.mode === 'voiceIsolation'
        ? runVoiceIsolationPipeline
        : runVoiceCleanPipeline;
    return fn(copies, sampleRate, options, {
      onProgress,
      shouldAbort: () => Boolean(signal?.aborted),
    });
  };

  // Fallback for environments without Web Worker support (e.g., SSR / unit test runners)
  if (typeof Worker === 'undefined') {
    const res = await runLocalFallback();
    return {
      audioBuffer: rebuildAudioBuffer(res.channels),
      report: res.report,
    };
  }

  return await new Promise((resolve, reject) => {
    let worker;
    try {
      worker = new Worker(new URL('./voiceClean.worker.js', import.meta.url), {
        type: 'module',
      });
    } catch (spawnErr) {
      // Graceful fallback if worker construction is blocked by CSP
      runLocalFallback()
        .then((res) =>
          resolve({
            audioBuffer: rebuildAudioBuffer(res.channels),
            report: res.report,
          })
        )
        .catch(reject);
      return;
    }

    let settled = false;
    const cleanup = () => {
      if (signal) {
        signal.removeEventListener('abort', onAbort);
      }
      try {
        worker.terminate();
      } catch {
        // Ignore terminate errors
      }
    };

    const onAbort = () => {
      if (settled) return;
      settled = true;
      try {
        worker.postMessage({ type: 'cancel' });
      } catch {
        // Ignore
      }
      cleanup();
      const err = new Error('Voice Clean cancelled by user.');
      err.name = 'AbortError';
      reject(err);
    };

    if (signal) {
      signal.addEventListener('abort', onAbort, { once: true });
    }

    worker.onmessage = (event) => {
      const msg = event.data;
      if (!msg || settled) return;

      if (msg.type === 'progress') {
        onProgress?.({ percent: msg.percent, stage: msg.stage });
      } else if (msg.type === 'done') {
        settled = true;
        cleanup();
        try {
          const audioBuffer = rebuildAudioBuffer(msg.channels);
          resolve({ audioBuffer, report: msg.report });
        } catch (buildErr) {
          reject(buildErr);
        }
      } else if (msg.type === 'cancelled') {
        settled = true;
        cleanup();
        const err = new Error('Voice Clean cancelled by user.');
        err.name = 'AbortError';
        reject(err);
      } else if (msg.type === 'error') {
        settled = true;
        cleanup();
        reject(new Error(msg.message || 'Voice Clean worker failed.'));
      }
    };

    worker.onerror = (errEvent) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error(errEvent?.message || 'Voice Clean worker crashed unexpectedly.'));
    };

    const wasmBinUrl =
      typeof window !== 'undefined' && window.location?.origin
        ? `${window.location.origin}/df_bg.wasm.bin`
        : '/df_bg.wasm.bin';

    const transferList = copies.map((ch) => ch.buffer);
    worker.postMessage(
      {
        type: 'run',
        channels: copies,
        sampleRate,
        options,
        wasmBinUrl,
      },
      transferList
    );
  });
}

/**
 * Runs the Voice Isolation pipeline off the main thread in the dedicated Web Worker.
 */
export function runVoiceIsolationInWorker(inputBuffer, options = {}, hooks = {}) {
  return runVoiceCleanInWorker(
    inputBuffer,
    { ...options, mode: 'voiceIsolation' },
    hooks
  );
}
