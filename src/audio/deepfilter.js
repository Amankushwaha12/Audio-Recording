/**
 * deepfilter.js — Client-Side DeepFilterNet3 Neural Speech Enhancement (WASM SIMD)
 *
 * Plain-language overview:
 * Unlike a simple noise gate (which only silences pauses between words),
 * DeepFilterNet3 is a deep neural network trained specifically on 48 kHz speech.
 * It analyzes short-time frequency spectra in two stages (ERB-band gain prediction
 * + complex deep filtering) so it can remove steady fan hum, traffic, keyboard
 * clicks, and room hiss *even while someone is actively talking*.
 *
 * Asset transport note:
 * The full DeepFilterNet3 WASM binary with baked neural weights is ~34.1 MB raw,
 * which exceeds standard 32 MB HTTP proxy caps and is slow to download raw.
 * We serve it pre-compressed as `/df_bg.wasm.bin` (~12.6 MB gzip), decompress it
 * natively in the browser via `DecompressionStream('gzip')`, and cache the
 * compiled WASM module in memory for instant subsequent runs.
 */

import { DeepFilter, SAMPLE_RATE as DF_SAMPLE_RATE } from '@lofcz/deepfilternet-web';

let wasmReady = false;
let wasmBlobUrlPromise = null;

function yieldToUI() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Fetches the ~12.6 MB gzipped DeepFilterNet3 WASM binary, decompresses it
 * natively in the browser, and returns a local `blob:` URL with `application/wasm`
 * MIME type. Cached after the first load so subsequent runs are instantaneous.
 */
async function ensureDeepFilterWasmUrl(onProgress) {
  if (wasmReady) {
    return undefined; // Already initialized inside @lofcz/deepfilternet-web module scope
  }

  if (!wasmBlobUrlPromise) {
    wasmBlobUrlPromise = (async () => {
      const response = await fetch('/df_bg.wasm.bin', { cache: 'force-cache' });
      if (!response.ok) {
        throw new Error(
          `Failed to load DeepFilterNet3 model (HTTP ${response.status} ${response.statusText})`
        );
      }

      const contentLength = Number(response.headers.get('Content-Length') || 0);
      let compressedBytes;

      if (response.body && typeof response.body.getReader === 'function') {
        const reader = response.body.getReader();
        const chunks = [];
        let loaded = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          loaded += value.byteLength;
          if (contentLength > 0) {
            const dlRatio = Math.min(1, loaded / contentLength);
            onProgress?.({
              percent: Math.round(4 + dlRatio * 10),
              stage: `DeepFilterNet3: Downloading model (${(loaded / (1024 * 1024)).toFixed(1)} / ${(
                contentLength /
                (1024 * 1024)
              ).toFixed(1)} MB)...`,
            });
          }
        }

        compressedBytes = new Uint8Array(loaded);
        let offset = 0;
        for (let i = 0; i < chunks.length; i++) {
          compressedBytes.set(chunks[i], offset);
          offset += chunks[i].byteLength;
        }
      } else {
        compressedBytes = new Uint8Array(await response.arrayBuffer());
      }

      onProgress?.({
        percent: 15,
        stage: 'DeepFilterNet3: Decompressing & compiling WASM neural engine...',
      });
      await yieldToUI();

      let wasmBytes = compressedBytes;
      // Check for gzip magic header (0x1f 0x8b) and decompress natively
      if (
        compressedBytes.byteLength >= 2 &&
        compressedBytes[0] === 0x1f &&
        compressedBytes[1] === 0x8b
      ) {
        if (typeof DecompressionStream === 'undefined') {
          throw new Error(
            'Your browser does not support DecompressionStream needed to unpack DeepFilterNet3.'
          );
        }
        const ds = new DecompressionStream('gzip');
        const decompressedStream = new Blob([compressedBytes])
          .stream()
          .pipeThrough(ds);
        wasmBytes = new Uint8Array(await new Response(decompressedStream).arrayBuffer());
      }

      const wasmBlob = new Blob([wasmBytes], { type: 'application/wasm' });
      return URL.createObjectURL(wasmBlob);
    })().catch((err) => {
      wasmBlobUrlPromise = null;
      throw err;
    });
  }

  return await wasmBlobUrlPromise;
}

/**
 * Resamples an AudioBuffer to `targetSampleRate` using Web Audio's built-in
 * high-quality OfflineAudioContext resampler.
 */
async function resampleAudioBuffer(inputBuffer, targetSampleRate) {
  if (inputBuffer.sampleRate === targetSampleRate) {
    return inputBuffer;
  }
  const numChannels = inputBuffer.numberOfChannels;
  const targetLength = Math.max(
    1,
    Math.round(inputBuffer.duration * targetSampleRate)
  );
  const offlineCtx = new OfflineAudioContext(
    numChannels,
    targetLength,
    targetSampleRate
  );
  const src = offlineCtx.createBufferSource();
  src.buffer = inputBuffer;
  src.connect(offlineCtx.destination);
  src.start(0);
  return await offlineCtx.startRendering();
}

/**
 * Runs DeepFilterNet3 neural denoising over an entire AudioBuffer.
 * Handles arbitrary input sample rates (automatically resamples to 48 kHz
 * for DeepFilterNet3 and resamples back to the original rate afterward) and
 * mono or stereo channels while reporting smooth frame-by-frame progress.
 *
 * @param {AudioBuffer} inputBuffer
 * @param {object} [options]
 * @param {number} [options.attenuationLimit=80] - Max noise attenuation in dB (0-100)
 * @param {number} [options.postFilterBeta=0.02] - Residual post-filter strength (0-0.05)
 * @param {(progress: { percent: number, stage: string }) => void} [onProgress]
 * @returns {Promise<{ audioBuffer: AudioBuffer, elapsedMs: number }>}
 */
export async function denoiseWithDeepFilterNet(
  inputBuffer,
  options = {},
  onProgress
) {
  const startMs = performance.now();
  const attenuationLimit = Math.max(
    0,
    Math.min(100, options.attenuationLimit ?? 80)
  );
  const postFilterBeta = Math.max(
    0,
    Math.min(0.05, options.postFilterBeta ?? 0.02)
  );

  const origSampleRate = inputBuffer.sampleRate;
  const numChannels = inputBuffer.numberOfChannels;

  // Step 0: Load & decompress the cached WASM model once per browser session
  const wasmUrl = await ensureDeepFilterWasmUrl(onProgress);

  // Step 1: Resample to DeepFilterNet3's native 48 kHz rate if needed
  let workingBuffer = inputBuffer;
  if (origSampleRate !== DF_SAMPLE_RATE) {
    onProgress?.({
      percent: 18,
      stage: `DeepFilterNet3: Resampling ${(origSampleRate / 1000).toFixed(1)} kHz → 48.0 kHz...`,
    });
    await yieldToUI();
    workingBuffer = await resampleAudioBuffer(inputBuffer, DF_SAMPLE_RATE);
  }

  const length48k = workingBuffer.length;
  const outChannels48k = [];

  // Step 2: Process each channel frame-by-frame through DeepFilterNet3 WASM
  for (let ch = 0; ch < numChannels; ch++) {
    const df = await DeepFilter.create({
      wasmUrl,
      attenuationLimit,
    });
    wasmReady = true;

    try {
      df.setAttenuationLimit(attenuationLimit);
      df.setPostFilterBeta(postFilterBeta);

      const frameLen = df.frameLength; // 480 samples (10 ms at 48 kHz)
      const srcData = workingBuffer.getChannelData(ch);
      const outData = new Float32Array(length48k);
      const frameBuf = new Float32Array(frameLen);

      const totalFrames = Math.ceil(length48k / frameLen);
      const yieldEvery = 45;

      for (let f = 0; f < totalFrames; f++) {
        const offset = f * frameLen;
        const copyLen = Math.min(frameLen, length48k - offset);

        frameBuf.fill(0);
        frameBuf.set(srcData.subarray(offset, offset + copyLen), 0);

        const cleanFrame = df.process(frameBuf);
        outData.set(cleanFrame.subarray(0, copyLen), offset);

        if (f % yieldEvery === 0) {
          const chProgress = (ch + f / Math.max(1, totalFrames)) / numChannels;
          const pct = Math.round(20 + chProgress * 66);
          onProgress?.({
            percent: pct,
            stage: `DeepFilterNet3 neural denoising (ch ${ch + 1}/${numChannels}): ${Math.round(
              chProgress * 100
            )}%...`,
          });
          await yieldToUI();
        }
      }

      outChannels48k.push(outData);
    } finally {
      df.destroy();
    }
  }

  // Step 3: Build 48 kHz AudioBuffer and resample back to original rate if needed
  const offline48k = new OfflineAudioContext(
    numChannels,
    Math.max(1, length48k),
    DF_SAMPLE_RATE
  );
  let resultBuffer = offline48k.createBuffer(
    numChannels,
    Math.max(1, length48k),
    DF_SAMPLE_RATE
  );
  for (let ch = 0; ch < numChannels; ch++) {
    resultBuffer.copyToChannel(outChannels48k[ch], ch);
  }

  if (origSampleRate !== DF_SAMPLE_RATE) {
    onProgress?.({
      percent: 90,
      stage: `DeepFilterNet3: Resampling back to ${(origSampleRate / 1000).toFixed(1)} kHz...`,
    });
    await yieldToUI();
    resultBuffer = await resampleAudioBuffer(resultBuffer, origSampleRate);
  }

  return {
    audioBuffer: resultBuffer,
    elapsedMs: Math.round(performance.now() - startMs),
  };
}
