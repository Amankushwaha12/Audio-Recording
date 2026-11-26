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
 * Everything executes locally in your browser via WebAssembly (@lofcz/deepfilternet-web)
 * with zero backend server calls.
 */

import { DeepFilter, SAMPLE_RATE as DF_SAMPLE_RATE } from '@lofcz/deepfilternet-web';
import dfWasmUrl from '@lofcz/deepfilternet-web/df_bg.wasm?url';

function yieldToUI() {
  return new Promise((resolve) => setTimeout(resolve, 0));
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

  onProgress?.({
    percent: 6,
    stage: 'DeepFilterNet3: Loading WASM neural model (cached after first run)...',
  });
  await yieldToUI();

  // Step 1: Resample to DeepFilterNet3's native 48 kHz rate if needed
  let workingBuffer = inputBuffer;
  if (origSampleRate !== DF_SAMPLE_RATE) {
    onProgress?.({
      percent: 14,
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
      wasmUrl: dfWasmUrl,
      attenuationLimit,
    });

    try {
      df.setAttenuationLimit(attenuationLimit);
      df.setPostFilterBeta(postFilterBeta);

      const frameLen = df.frameLength; // 480 samples (10 ms at 48 kHz)
      const srcData = workingBuffer.getChannelData(ch);
      const outData = new Float32Array(length48k);
      const frameBuf = new Float32Array(frameLen);

      const totalFrames = Math.ceil(length48k / frameLen);
      const yieldEvery = 45; // Yield roughly every ~450 ms of audio so UI stays responsive

      for (let f = 0; f < totalFrames; f++) {
        const offset = f * frameLen;
        const copyLen = Math.min(frameLen, length48k - offset);

        // Copy samples into fixed-length frame buffer, zero-padding the final tail frame
        frameBuf.fill(0);
        frameBuf.set(srcData.subarray(offset, offset + copyLen), 0);

        const cleanFrame = df.process(frameBuf);
        outData.set(cleanFrame.subarray(0, copyLen), offset);

        if (f % yieldEvery === 0) {
          const chProgress = (ch + f / Math.max(1, totalFrames)) / numChannels;
          const pct = Math.round(18 + chProgress * 66);
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
      percent: 88,
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
