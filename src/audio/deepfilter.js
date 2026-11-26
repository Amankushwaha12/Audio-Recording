/**
 * deepfilter.js — Studio-Tuned Client-Side DeepFilterNet3 Neural Engine (WASM SIMD)
 *
 * Key studio reliability & quality upgrades:
 *   1. Direct WebAssembly.Module compilation: fetches `/df_bg.wasm.bin` (~12.1 MB
 *      gzipped), decompresses via `DecompressionStream('gzip')`, compiles a
 *      `WebAssembly.Module` once in memory, and passes it directly into
 *      `DeepFilter.create({ wasmUrl: wasmModule })` — completely eliminating
 *      `blob:` URL streaming errors across Chrome, Safari, Firefox, and iframes.
 *   2. Pre-conditioning (75 Hz HPF + Smart Pre-Gain): laptop/phone microphones
 *      often record very quietly (-35 dBFS) with DC/rumble offset. Pre-conditioning
 *      brings speech into DeepFilterNet3's optimal training range (-18 dBFS RMS)
 *      so quiet words are never mistaken for noise and speech stays crisp & full.
 *   3. Latency compensation: pads trailing frames and trims DeepFilterNet3's
 *      2-frame (20 ms) STFT overlap-add delay so A/B playback stays aligned.
 */

import { DeepFilter, SAMPLE_RATE as DF_SAMPLE_RATE } from '@lofcz/deepfilternet-web';
import { applyBiquad, dbToLinear, linearToDb } from './analyzer.js';

let cachedWasmModulePromise = null;

function yieldToUI() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Fetches `/df_bg.wasm.bin`, decompresses gzip natively, verifies the WASM
 * binary magic header (`\0asm`), and compiles a reusable `WebAssembly.Module`.
 */
async function getCompiledDeepFilterModule(onProgress) {
  if (!cachedWasmModulePromise) {
    cachedWasmModulePromise = (async () => {
      const response = await fetch('/df_bg.wasm.bin', { cache: 'force-cache' });
      if (!response.ok) {
        throw new Error(
          `Could not fetch DeepFilterNet3 model (HTTP ${response.status} ${response.statusText})`
        );
      }

      const contentType = response.headers.get('Content-Type') || '';
      if (contentType.includes('text/html')) {
        throw new Error(
          'Received HTML instead of DeepFilterNet3 binary (check static asset routing).'
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
              stage: `DeepFilterNet3: Loading neural weights (${(
                loaded /
                (1024 * 1024)
              ).toFixed(1)} / ${(contentLength / (1024 * 1024)).toFixed(1)} MB)...`,
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
        stage: 'DeepFilterNet3: Unpacking & compiling WebAssembly SIMD engine...',
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
            'Your browser does not support DecompressionStream required for DeepFilterNet3.'
          );
        }
        const ds = new DecompressionStream('gzip');
        const decompressedStream = new Blob([compressedBytes])
          .stream()
          .pipeThrough(ds);
        wasmBytes = new Uint8Array(await new Response(decompressedStream).arrayBuffer());
      }

      // Verify WebAssembly magic header: 0x00 0x61 0x73 0x6D ("\0asm")
      if (
        wasmBytes.byteLength < 8 ||
        wasmBytes[0] !== 0x00 ||
        wasmBytes[1] !== 0x61 ||
        wasmBytes[2] !== 0x73 ||
        wasmBytes[3] !== 0x6d
      ) {
        throw new Error('DeepFilterNet3 model payload is corrupted or not valid WebAssembly.');
      }

      return await WebAssembly.compile(wasmBytes);
    })().catch((err) => {
      cachedWasmModulePromise = null;
      throw err;
    });
  }

  return await cachedWasmModulePromise;
}

/**
 * Resamples an AudioBuffer to `targetSampleRate` using Web Audio's built-in
 * bandlimited OfflineAudioContext resampler.
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
 * Gentle 75 Hz Butterworth High-Pass pre-filter to remove DC offset and sub-bass
 * mic rumble before neural analysis so DeepFilterNet3 focuses purely on speech.
 */
function preFilterRumble(samples, sampleRate) {
  const f0 = 75;
  const w0 = (2 * Math.PI * f0) / sampleRate;
  const cosW0 = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
  const a0 = 1 + alpha;
  return applyBiquad(samples, {
    b0: ((1 + cosW0) / 2) / a0,
    b1: -(1 + cosW0) / a0,
    b2: ((1 + cosW0) / 2) / a0,
    a1: (-2 * cosW0) / a0,
    a2: (1 - alpha) / a0,
  });
}

/**
 * Computes optimal pre-gain so quiet mic recordings enter DeepFilterNet3 near
 * -18 dBFS active speech RMS without clipping peaks above -2 dBFS.
 */
function computeOptimalPreGain(channels) {
  let peak = 0;
  let activeSumSq = 0;
  let activeCount = 0;
  const length = channels[0].length;
  const blockSize = 2400; // 50 ms at 48 kHz
  const blockRms = [];

  for (let start = 0; start < length; start += blockSize) {
    const end = Math.min(length, start + blockSize);
    let sq = 0;
    let cnt = 0;
    for (let ch = 0; ch < channels.length; ch++) {
      const d = channels[ch];
      for (let i = start; i < end; i++) {
        const v = d[i];
        const a = Math.abs(v);
        if (a > peak) peak = a;
        sq += v * v;
        cnt++;
      }
    }
    blockRms.push(Math.sqrt(sq / Math.max(1, cnt)));
  }

  blockRms.sort((a, b) => a - b);
  // Top 30% loudest frames represent active speech
  const speechStartIdx = Math.floor(blockRms.length * 0.7);
  for (let i = speechStartIdx; i < blockRms.length; i++) {
    activeSumSq += blockRms[i] * blockRms[i];
    activeCount++;
  }

  const speechRmsDb = linearToDb(Math.sqrt(activeSumSq / Math.max(1, activeCount)));
  const peakDb = linearToDb(peak);

  if (speechRmsDb <= -62 || peak <= 1e-5) {
    return 1.0;
  }

  // Target ~-18 dBFS speech RMS while preserving at least 2 dB peak headroom
  const desiredGainDb = -18 - speechRmsDb;
  const maxHeadroomGainDb = -2 - peakDb;
  const clampedDb = Math.max(-6, Math.min(20, Math.min(desiredGainDb, maxHeadroomGainDb)));
  return dbToLinear(clampedDb);
}

/**
 * Runs DeepFilterNet3 neural denoising over an entire AudioBuffer with
 * pre-conditioning, direct compiled WebAssembly.Module instantiation, and
 * STFT lookahead delay compensation.
 */
export async function denoiseWithDeepFilterNet(
  inputBuffer,
  options = {},
  onProgress
) {
  const startMs = performance.now();
  const attenuationLimit = Math.max(
    0,
    Math.min(100, options.attenuationLimit ?? 85)
  );
  const postFilterBeta = Math.max(
    0,
    Math.min(0.05, options.postFilterBeta ?? 0.025)
  );

  const origSampleRate = inputBuffer.sampleRate;
  const numChannels = inputBuffer.numberOfChannels;

  // Step 0: Load & compile reusable WebAssembly.Module
  const wasmModule = await getCompiledDeepFilterModule(onProgress);

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

  // Step 2: Pre-filter sub-bass rumble & optimize gain for quiet mic recordings
  const preChannels = [];
  for (let ch = 0; ch < numChannels; ch++) {
    preChannels.push(preFilterRumble(workingBuffer.getChannelData(ch), DF_SAMPLE_RATE));
  }
  const preGain = computeOptimalPreGain(preChannels);
  const invPreGain = preGain > 0 ? 1 / preGain : 1;
  const keepPreGain = Boolean(options.chainMastering); // If chaining mastering, keep healthy level; if pure DF3, restore original level

  const outChannels48k = [];

  // Step 3: Process each channel frame-by-frame through DeepFilterNet3 WASM
  for (let ch = 0; ch < numChannels; ch++) {
    const df = await DeepFilter.create({
      wasmUrl: wasmModule,
      attenuationLimit,
    });

    try {
      df.setAttenuationLimit(attenuationLimit);
      df.setPostFilterBeta(postFilterBeta);

      const frameLen = df.frameLength; // 480 samples (10 ms at 48 kHz)
      const lookaheadFrames = 2; // Compensate libDF's 2-frame STFT lookahead delay
      const lookaheadSamples = lookaheadFrames * frameLen;

      const srcData = preChannels[ch];
      const totalPaddedLen = length48k + lookaheadSamples;
      const paddedOut = new Float32Array(totalPaddedLen);
      const frameBuf = new Float32Array(frameLen);

      const totalFrames = Math.ceil(totalPaddedLen / frameLen);
      const yieldEvery = 45;

      for (let f = 0; f < totalFrames; f++) {
        const offset = f * frameLen;
        frameBuf.fill(0);

        if (offset < length48k) {
          const copyLen = Math.min(frameLen, length48k - offset);
          for (let i = 0; i < copyLen; i++) {
            frameBuf[i] = srcData[offset + i] * preGain;
          }
        }

        const cleanFrame = df.process(frameBuf);
        const writeLen = Math.min(frameLen, totalPaddedLen - offset);
        paddedOut.set(cleanFrame.subarray(0, writeLen), offset);

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

      // Trim startup lookahead offset so denoised speech stays time-aligned with Original
      const aligned = new Float32Array(length48k);
      const scaleBack = keepPreGain ? 1.0 : invPreGain;
      for (let i = 0; i < length48k; i++) {
        const srcIdx = i + lookaheadSamples;
        const val = srcIdx < totalPaddedLen ? paddedOut[srcIdx] * scaleBack : 0;
        aligned[i] = Math.max(-1, Math.min(1, val));
      }

      outChannels48k.push(aligned);
    } finally {
      df.destroy();
    }
  }

  // Step 4: Build 48 kHz AudioBuffer and resample back to original rate if needed
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
