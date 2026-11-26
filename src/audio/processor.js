/**
 * processor.js — Deterministic 6-Stage Automatic Audio Mastering Pipeline
 *
 * Plain-language overview of the processing chain (applied in strict order):
 *   a. High-Pass Filter — Removes sub-bass rumble, wind pops, and AC hum
 *      below the cutoff frequency (default 80 Hz) without thinning the voice.
 *   b. Simple Noise Gate / Noise Reduction — Measures the quietest 10% of
 *      50 ms frames to estimate the background noise floor, then smoothly
 *      attenuates quiet frames below the gate threshold using attack/hold/release.
 *   c. Dynamics Compressor — Soft-knee feed-forward compressor that gently
 *      turns down loud peaks above the threshold so quiet and loud words sit
 *      closer together.
 *   d. Auto-Leveler — Splits audio into ~0.5 s windows, measures loudness, and
 *      rides gain smoothly toward a consistent speech level while leaving
 *      silence and room noise untouched.
 *   e. Loudness Normalization + True-Peak Limiter — Scales the whole track to
 *      the target loudness (LUFS for Podcast/Broadcast, or RMS for Audiobook)
 *      and catches any fast inter-sample peaks with a 4x oversampled lookahead
 *      limiter at the chosen ceiling (default -1.0 dBTP).
 *   f. Optional Silence Trimming — Finds pauses longer than N seconds and
 *      shortens them to a natural brief pause using smooth cosine crossfades.
 */

import {
  applyBiquad,
  dbToLinear,
  linearToDb,
  measureBS1770Loudness,
} from './analyzer.js';
import { fftRadix2, ifftRadix2 } from './fft.js';

/**
 * Curated mastering presets covering both Current DSP and DeepFilterNet3 engines.
 */
export const PRESETS = {
  studioAi: {
    id: 'studioAi',
    label: 'Studio AI Master',
    summary: 'Best quality: DeepFilterNet3 + Voice Clarity EQ + -15 LUFS broadcast loudness',
    recommendedEngine: 'deepfilternet',
    dfOptions: {
      attenuationLimit: 88,
      postFilterBeta: 0.025,
      chainMastering: true,
    },
    options: {
      highPassEnabled: true,
      highPassFreq: 85,
      noiseGateEnabled: true,
      noiseMode: 'spectral',
      noiseGateReductionDb: 18,
      noiseGateMarginDb: 6,
      voiceClarityEnabled: true,
      voiceClarityAmount: 0.75,
      compressorEnabled: true,
      compThresholdDb: -19,
      compRatio: 2.8,
      compAttackMs: 8,
      compReleaseMs: 120,
      levelerEnabled: true,
      levelerStrength: 0.85,
      normalizeEnabled: true,
      normalizeMode: 'lufs',
      targetLoudness: -15,
      limiterCeilingDbtp: -1.0,
      silenceTrimEnabled: false,
      silenceMaxPauseSec: 1.1,
    },
  },
  podcast: {
    id: 'podcast',
    label: 'Podcast',
    summary: '-16 LUFS, spectral noise cleanup, studio voice clarity, balanced dynamics',
    dfOptions: {
      attenuationLimit: 70,
      postFilterBeta: 0.02,
      chainMastering: true,
    },
    options: {
      highPassEnabled: true,
      highPassFreq: 80,
      noiseGateEnabled: true,
      noiseMode: 'spectral',
      noiseGateReductionDb: 16,
      noiseGateMarginDb: 6,
      voiceClarityEnabled: true,
      voiceClarityAmount: 0.65,
      compressorEnabled: true,
      compThresholdDb: -20,
      compRatio: 2.6,
      compAttackMs: 10,
      compReleaseMs: 135,
      levelerEnabled: true,
      levelerStrength: 0.8,
      normalizeEnabled: true,
      normalizeMode: 'lufs',
      targetLoudness: -16,
      limiterCeilingDbtp: -1.0,
      silenceTrimEnabled: false,
      silenceMaxPauseSec: 1.2,
    },
  },
  audiobook: {
    id: 'audiobook',
    label: 'Audiobook',
    summary: 'ACX style: -20 dBFS RMS, natural voice warmth, strict -3.0 dBTP ceiling',
    dfOptions: {
      attenuationLimit: 45,
      postFilterBeta: 0.01,
      chainMastering: true,
    },
    options: {
      highPassEnabled: true,
      highPassFreq: 75,
      noiseGateEnabled: true,
      noiseMode: 'spectral',
      noiseGateReductionDb: 10,
      noiseGateMarginDb: 5,
      voiceClarityEnabled: true,
      voiceClarityAmount: 0.4,
      compressorEnabled: true,
      compThresholdDb: -22,
      compRatio: 2.0,
      compAttackMs: 18,
      compReleaseMs: 180,
      levelerEnabled: true,
      levelerStrength: 0.55,
      normalizeEnabled: true,
      normalizeMode: 'rms',
      targetLoudness: -20,
      limiterCeilingDbtp: -3.0,
      silenceTrimEnabled: false,
      silenceMaxPauseSec: 1.5,
    },
  },
  youtube: {
    id: 'youtube',
    label: 'YouTube / Reels',
    summary: 'Loud & crisp -14 LUFS voiceover, strong clarity EQ, trims pauses >0.8 s',
    dfOptions: {
      attenuationLimit: 85,
      postFilterBeta: 0.025,
      chainMastering: true,
    },
    options: {
      highPassEnabled: true,
      highPassFreq: 85,
      noiseGateEnabled: true,
      noiseMode: 'spectral',
      noiseGateReductionDb: 20,
      noiseGateMarginDb: 6,
      voiceClarityEnabled: true,
      voiceClarityAmount: 0.85,
      compressorEnabled: true,
      compThresholdDb: -18,
      compRatio: 3.2,
      compAttackMs: 6,
      compReleaseMs: 110,
      levelerEnabled: true,
      levelerStrength: 0.85,
      normalizeEnabled: true,
      normalizeMode: 'lufs',
      targetLoudness: -14,
      limiterCeilingDbtp: -1.0,
      silenceTrimEnabled: true,
      silenceMaxPauseSec: 0.8,
    },
  },
  broadcast: {
    id: 'broadcast',
    label: 'Broadcast EBU',
    summary: 'EBU R128 / TV standard: -23 LUFS, natural clarity, -1.0 dBTP ceiling',
    dfOptions: {
      attenuationLimit: 55,
      postFilterBeta: 0.015,
      chainMastering: true,
    },
    options: {
      highPassEnabled: true,
      highPassFreq: 80,
      noiseGateEnabled: true,
      noiseMode: 'spectral',
      noiseGateReductionDb: 12,
      noiseGateMarginDb: 5,
      voiceClarityEnabled: true,
      voiceClarityAmount: 0.5,
      compressorEnabled: true,
      compThresholdDb: -24,
      compRatio: 2.2,
      compAttackMs: 15,
      compReleaseMs: 160,
      levelerEnabled: true,
      levelerStrength: 0.65,
      normalizeEnabled: true,
      normalizeMode: 'lufs',
      targetLoudness: -23,
      limiterCeilingDbtp: -1.0,
      silenceTrimEnabled: false,
      silenceMaxPauseSec: 1.5,
    },
  },
  noisy: {
    id: 'noisy',
    label: 'Noisy Recording',
    summary: 'Heavy rescue: 100 Hz HPF, max noise suppression, voice clarity, trims >1.0 s',
    dfOptions: {
      attenuationLimit: 100,
      postFilterBeta: 0.035,
      chainMastering: true,
    },
    options: {
      highPassEnabled: true,
      highPassFreq: 100,
      noiseGateEnabled: true,
      noiseMode: 'spectral',
      noiseGateReductionDb: 26,
      noiseGateMarginDb: 7,
      voiceClarityEnabled: true,
      voiceClarityAmount: 0.75,
      compressorEnabled: true,
      compThresholdDb: -21,
      compRatio: 3.4,
      compAttackMs: 8,
      compReleaseMs: 120,
      levelerEnabled: true,
      levelerStrength: 0.9,
      normalizeEnabled: true,
      normalizeMode: 'lufs',
      targetLoudness: -15,
      limiterCeilingDbtp: -1.0,
      silenceTrimEnabled: true,
      silenceMaxPauseSec: 1.0,
    },
  },
};

export const LOUDNESS_PRESETS = [
  { id: 'youtube-14', label: 'YouTube (-14 LUFS)', mode: 'lufs', target: -14 },
  { id: 'podcast-16', label: 'Podcast (-16 LUFS)', mode: 'lufs', target: -16 },
  { id: 'broadcast-23', label: 'Broadcast (-23 LUFS)', mode: 'lufs', target: -23 },
  { id: 'audiobook-20', label: 'Audiobook (-20 RMS)', mode: 'rms', target: -20 },
];

/**
 * Helper to yield to the browser's main thread between DSP stages
 * so the progress bar and stage label paint smoothly.
 */
function yieldToUI() {
  return new Promise((resolve) => setTimeout(resolve, 16));
}

/**
 * Stage a: 2nd-Order Butterworth High-Pass Filter (12 dB/octave)
 * Removes low-frequency room rumble, desk bumps, and plosives below `cutoffHz`.
 */
function runHighPassFilter(channels, sampleRate, cutoffHz) {
  const f0 = Math.max(20, Math.min(cutoffHz, sampleRate * 0.4));
  const w0 = (2 * Math.PI * f0) / sampleRate;
  const cosW0 = Math.cos(w0);
  const sinW0 = Math.sin(w0);
  // Q = 0.7071 (1/sqrt(2)) produces a maximally flat Butterworth passband
  const alpha = sinW0 / (2 * Math.SQRT1_2);
  const a0 = 1 + alpha;

  const coeffs = {
    b0: ((1 + cosW0) / 2) / a0,
    b1: -(1 + cosW0) / a0,
    b2: ((1 + cosW0) / 2) / a0,
    a1: (-2 * cosW0) / a0,
    a2: (1 - alpha) / a0,
  };

  return channels.map((ch) => applyBiquad(ch, coeffs));
}

/**
 * Stage b: Adaptive Noise Gate / Noise Reduction
 * 1. Measures RMS energy in 50 ms frames and takes the 10th percentile as the
 *    estimated background noise floor.
 * 2. Places the gate threshold `marginDb` above that noise floor (clamped to a
 *    safe range so loud recordings aren't over-gated).
 * 3. Smoothly attenuates frames below threshold by `reductionDb` using
 *    10 ms attack, 60 ms hold, and 140 ms release to prevent choppy syllables.
 */
function runNoiseGate(channels, sampleRate, reductionDb, marginDb = 6) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const frameSamples = Math.max(1, Math.round(0.05 * sampleRate));
  const numFrames = Math.max(1, Math.ceil(length / frameSamples));
  const frameRmsDb = new Float32Array(numFrames);

  for (let f = 0; f < numFrames; f++) {
    const start = f * frameSamples;
    const end = Math.min(length, start + frameSamples);
    let sumSq = 0;
    let count = 0;
    for (let i = start; i < end; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        const s = channels[ch][i];
        sumSq += s * s;
        count++;
      }
    }
    frameRmsDb[f] = linearToDb(Math.sqrt(sumSq / Math.max(1, count)));
  }

  // Estimate noise floor from the quietest 10% of frames (ignoring digital dead zero)
  const nonZeroFrames = [];
  for (let i = 0; i < numFrames; i++) {
    if (frameRmsDb[i] > -95) nonZeroFrames.push(frameRmsDb[i]);
  }
  nonZeroFrames.sort((a, b) => a - b);

  const noiseFloorDb =
    nonZeroFrames.length > 0
      ? nonZeroFrames[Math.floor(nonZeroFrames.length * 0.1)]
      : -65;

  // Compute dynamic gate threshold above the measured noise floor
  const thresholdDb = Math.max(-68, Math.min(-24, noiseFloorDb + marginDb));
  const minGain = dbToLinear(-Math.abs(reductionDb));

  // Sample-level envelope smoothing constants
  const attackCoeff = Math.exp(-1 / (0.01 * sampleRate)); // 10 ms open
  const releaseCoeff = Math.exp(-1 / (0.14 * sampleRate)); // 140 ms close
  const rmsDetectCoeff = Math.exp(-1 / (0.015 * sampleRate)); // 15 ms detector
  const holdSamples = Math.round(0.06 * sampleRate); // 60 ms hold

  let envSq = 0;
  let gain = 1.0;
  let holdCounter = 0;

  const out = channels.map((ch) => new Float32Array(ch.length));

  for (let i = 0; i < length; i++) {
    let linkedSq = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      linkedSq += channels[ch][i] * channels[ch][i];
    }
    linkedSq /= numChannels;
    envSq = rmsDetectCoeff * envSq + (1 - rmsDetectCoeff) * linkedSq;
    const levelDb = linearToDb(Math.sqrt(envSq));

    let targetGain;
    if (levelDb >= thresholdDb) {
      targetGain = 1.0;
      holdCounter = holdSamples;
    } else if (holdCounter > 0) {
      targetGain = 1.0;
      holdCounter--;
    } else {
      // Soft knee transition across 6 dB below threshold for natural gating
      const belowDb = thresholdDb - levelDb;
      const ratio = Math.min(1, belowDb / 6);
      targetGain = 1.0 - ratio * (1.0 - minGain);
    }

    if (targetGain > gain) {
      gain = attackCoeff * gain + (1 - attackCoeff) * targetGain;
    } else {
      gain = releaseCoeff * gain + (1 - releaseCoeff) * targetGain;
    }

    for (let ch = 0; ch < numChannels; ch++) {
      out[ch][i] = channels[ch][i] * gain;
    }
  }

  return { channels: out, noiseFloorDb, thresholdDb };
}

/**
 * Stage b (Studio Upgrade): STFT Spectral Wiener / Ephraim-Malah Noise Reduction
 * Removes continuous fan hum, AC noise, and broadband room hiss *even while
 * someone is speaking* using 1024-point Hann 50% overlap-add STFT, automatic
 * per-frequency noise floor profiling (15th percentile across quiet frames),
 * decision-directed SNR smoothing (eliminates robotic "musical noise"), and
 * a subtle finishing pause gate so silences stay studio-black.
 */
function runSpectralDenoiser(channels, sampleRate, reductionDb = 18) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const fftSize = 1024;
  const hop = fftSize >> 1; // 50% overlap (512 samples)
  const halfBins = (fftSize >> 1) + 1;

  // Periodic Hann window for exact COLA (Constant Overlap-Add) reconstruction
  const win = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / fftSize));
  }
  // Using sqrt(Hann) for both analysis and synthesis guarantees exact 1.0 overlap sum
  const sqrtWin = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    sqrtWin[i] = Math.sqrt(win[i]);
  }

  const minGain = dbToLinear(-Math.max(4, Math.min(36, reductionDb)));
  const overSub = 1.15 + Math.min(0.85, reductionDb / 30); // Gentle oversubtraction scaled by strength
  const alphaDD = 0.92; // Decision-directed smoothing prevents musical noise artifacts

  const outChannels = [];
  let measuredFloorDb = -60;

  for (let ch = 0; ch < numChannels; ch++) {
    const src = channels[ch];
    const numFrames = Math.max(1, Math.ceil((length + hop) / hop));

    // Pass 1: Measure per-frame broadband energy to locate quietest 15% of frames
    const frameEnergy = new Float32Array(numFrames);
    for (let f = 0; f < numFrames; f++) {
      const start = f * hop;
      let sq = 0;
      let cnt = 0;
      for (let i = 0; i < fftSize; i++) {
        const idx = start + i;
        if (idx < length) {
          const s = src[idx];
          sq += s * s;
          cnt++;
        }
      }
      frameEnergy[f] = cnt > 0 ? sq / cnt : 0;
    }

    const frameOrder = Array.from({ length: numFrames }, (_, i) => i).filter(
      (i) => frameEnergy[i] > 1e-11
    );
    frameOrder.sort((a, b) => frameEnergy[a] - frameEnergy[b]);

    // Take quietest 15% of frames (at least 4 frames) to estimate noise PSD per bin
    const quietCount = Math.max(4, Math.min(frameOrder.length, Math.ceil(frameOrder.length * 0.15)));
    const noisePsd = new Float32Array(halfBins);
    const real = new Float32Array(fftSize);
    const imag = new Float32Array(fftSize);

    if (quietCount > 0) {
      let quietAvgSq = 0;
      for (let q = 0; q < quietCount; q++) {
        const fIdx = frameOrder[q];
        quietAvgSq += frameEnergy[fIdx];
        const start = fIdx * hop;
        for (let i = 0; i < fftSize; i++) {
          const idx = start + i;
          real[i] = idx < length ? src[idx] * sqrtWin[i] : 0;
          imag[i] = 0;
        }
        fftRadix2(real, imag);
        for (let b = 0; b < halfBins; b++) {
          noisePsd[b] += real[b] * real[b] + imag[b] * imag[b];
        }
      }
      for (let b = 0; b < halfBins; b++) {
        noisePsd[b] = (noisePsd[b] / quietCount) * overSub;
      }
      if (ch === 0) {
        measuredFloorDb = linearToDb(Math.sqrt(quietAvgSq / quietCount));
      }
    }

    // Pass 2: Decision-directed Wiener filtering + Overlap-Add synthesis
    const out = new Float32Array(length);
    const prevPostSnr = new Float32Array(halfBins).fill(1);
    const prevGain = new Float32Array(halfBins).fill(1);
    const binGain = new Float32Array(halfBins);
    const smoothGain = new Float32Array(halfBins);

    for (let f = 0; f < numFrames; f++) {
      const start = f * hop;
      for (let i = 0; i < fftSize; i++) {
        const idx = start + i;
        real[i] = idx < length ? src[idx] * sqrtWin[i] : 0;
        imag[i] = 0;
      }

      fftRadix2(real, imag);

      for (let b = 0; b < halfBins; b++) {
        const power = real[b] * real[b] + imag[b] * imag[b];
        const nPow = Math.max(noisePsd[b], 1e-12);
        const postSnr = power / nPow;

        // Decision-directed a-priori SNR estimate (Ephraim-Malah)
        const priorSnr =
          alphaDD * (prevGain[b] * prevGain[b] * prevPostSnr[b]) +
          (1 - alphaDD) * Math.max(postSnr - 1, 0);

        // Wiener spectral gain with protective speech floor
        const wiener = priorSnr / (priorSnr + 1);
        binGain[b] = Math.max(minGain, Math.min(1, wiener));
        prevPostSnr[b] = Math.min(postSnr, 50);
      }

      // 3-bin frequency smoothing suppresses isolated musical-noise pings
      for (let b = 0; b < halfBins; b++) {
        const gPrev = binGain[Math.max(0, b - 1)];
        const gCurr = binGain[b];
        const gNext = binGain[Math.min(halfBins - 1, b + 1)];
        const g = 0.15 * gPrev + 0.7 * gCurr + 0.15 * gNext;
        smoothGain[b] = g;
        prevGain[b] = g;
      }

      // Apply symmetric spectral gain to positive and negative frequency halves
      real[0] *= smoothGain[0];
      imag[0] *= smoothGain[0];
      real[fftSize >> 1] *= smoothGain[halfBins - 1];
      imag[fftSize >> 1] *= smoothGain[halfBins - 1];

      for (let b = 1; b < fftSize >> 1; b++) {
        const g = smoothGain[b];
        real[b] *= g;
        imag[b] *= g;
        real[fftSize - b] *= g;
        imag[fftSize - b] *= g;
      }

      ifftRadix2(real, imag);

      for (let i = 0; i < fftSize; i++) {
        const idx = start + i;
        if (idx < length) {
          out[idx] += real[i] * sqrtWin[i];
        }
      }
    }

    outChannels.push(out);
  }

  // Subtle finishing gate (-8 dB max on pure dead pauses) for studio-clean silence
  const polished = runNoiseGate(
    outChannels,
    sampleRate,
    Math.min(10, Math.round(reductionDb * 0.4)),
    5
  );

  return {
    channels: polished.channels,
    noiseFloorDb: measuredFloorDb,
    thresholdDb: polished.thresholdDb,
  };
}

/**
 * Helper: RBJ Peaking EQ biquad coefficients.
 */
function makePeakingCoeffs(sampleRate, freqHz, gainDb, q) {
  const f0 = Math.max(20, Math.min(freqHz, sampleRate * 0.45));
  const A = Math.pow(10, gainDb / 40);
  const w0 = (2 * Math.PI * f0) / sampleRate;
  const cosW0 = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  const a0 = 1 + alpha / A;
  return {
    b0: (1 + alpha * A) / a0,
    b1: (-2 * cosW0) / a0,
    b2: (1 - alpha * A) / a0,
    a1: (-2 * cosW0) / a0,
    a2: (1 - alpha / A) / a0,
  };
}

/**
 * Helper: RBJ High-Shelf EQ biquad coefficients.
 */
function makeHighShelfCoeffs(sampleRate, freqHz, gainDb, slope = 0.8) {
  const f0 = Math.max(100, Math.min(freqHz, sampleRate * 0.45));
  const A = Math.pow(10, gainDb / 40);
  const w0 = (2 * Math.PI * f0) / sampleRate;
  const cosW0 = Math.cos(w0);
  const sinW0 = Math.sin(w0);
  const alpha = (sinW0 / 2) * Math.sqrt((A + 1 / A) * (1 / slope - 1) + 2);
  const twoSqrtAAlpha = 2 * Math.sqrt(A) * alpha;
  const a0 = A + 1 - (A - 1) * cosW0 + twoSqrtAAlpha;
  return {
    b0: (A * (A + 1 + (A - 1) * cosW0 + twoSqrtAAlpha)) / a0,
    b1: (-2 * A * (A - 1 + (A + 1) * cosW0)) / a0,
    b2: (A * (A + 1 + (A - 1) * cosW0 - twoSqrtAAlpha)) / a0,
    a1: (2 * (A - 1 - (A + 1) * cosW0)) / a0,
    a2: (A + 1 - (A - 1) * cosW0 - twoSqrtAAlpha) / a0,
  };
}

/**
 * Studio Voice Clarity EQ + Split-Band De-Esser
 * Shapes raw mic recordings into polished broadcast vocals:
 *   1. Gentle low warmth (+1.3 dB @ 135 Hz)
 *   2. De-mud scoop (-2.8 dB @ 260 Hz) to remove boxy room resonance
 *   3. Vocal presence boost (+3.2 dB @ 3.2 kHz) for crisp speech intelligibility
 *   4. Condenser air shelf (+1.8 dB @ 9.5 kHz) for open studio sheen
 *   5. Split-band dynamic de-esser (tames harsh 6–9 kHz "s/sh" spikes only)
 */
function runVoiceClarityEqAndDeEsser(channels, sampleRate, amount = 0.7) {
  const amt = Math.max(0, Math.min(1, amount));
  if (amt <= 0.01) {
    return channels;
  }

  const warmth = makePeakingCoeffs(sampleRate, 135, 1.3 * amt, 1.0);
  const deMud = makePeakingCoeffs(sampleRate, 260, -2.8 * amt, 1.2);
  const presence = makePeakingCoeffs(sampleRate, 3200, 3.2 * amt, 0.95);
  const air = makeHighShelfCoeffs(sampleRate, 9500, 1.8 * amt, 0.8);

  // Split-band sibilance high-pass (5.8 kHz) for dynamic de-essing
  const fSib = Math.min(5800, sampleRate * 0.42);
  const w0 = (2 * Math.PI * fSib) / sampleRate;
  const cosW0 = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
  const a0 = 1 + alpha;
  const sibHp = {
    b0: ((1 + cosW0) / 2) / a0,
    b1: -(1 + cosW0) / a0,
    b2: ((1 + cosW0) / 2) / a0,
    a1: (-2 * cosW0) / a0,
    a2: (1 - alpha) / a0,
  };

  const eqChannels = channels.map((ch) => {
    let x = applyBiquad(ch, warmth);
    x = applyBiquad(x, deMud);
    x = applyBiquad(x, presence);
    x = applyBiquad(x, air);
    return x;
  });

  // Dynamic split-band de-esser: subtract excess sibilance above -24 dBFS
  const attackCoeff = Math.exp(-1 / Math.max(1, 0.002 * sampleRate)); // 2 ms
  const releaseCoeff = Math.exp(-1 / Math.max(1, 0.04 * sampleRate)); // 40 ms
  const sibThresholdDb = -24;
  const maxDeEssDb = 6 * amt;

  return eqChannels.map((ch) => {
    const highs = applyBiquad(ch, sibHp);
    const out = new Float32Array(ch.length);
    let env = 0;

    for (let i = 0; i < ch.length; i++) {
      const a = Math.abs(highs[i]);
      env = a > env ? attackCoeff * env + (1 - attackCoeff) * a : releaseCoeff * env + (1 - releaseCoeff) * a;
      const sibDb = linearToDb(env);
      let attenGain = 1.0;
      if (sibDb > sibThresholdDb) {
        const grDb = Math.min(maxDeEssDb, (sibDb - sibThresholdDb) * 0.65);
        attenGain = dbToLinear(-grDb);
      }
      // Attenuate ONLY the high-frequency sibilant band; keep low/mid speech untouched
      const lowMid = ch[i] - highs[i];
      out[i] = lowMid + highs[i] * attenGain;
    }
    return out;
  });
}

/**
 * Stage c: Soft-Knee Feed-Forward Dynamics Compressor
 * Compresses peaks exceeding `thresholdDb` by `ratio`:1 with a 6 dB soft knee,
 * stereo-linked peak detection, and smooth attack/release ballistics.
 */
function runCompressor(
  channels,
  sampleRate,
  { thresholdDb, ratio, attackMs, releaseMs }
) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const kneeDb = 6;
  const halfKnee = kneeDb / 2;

  const attackCoeff = Math.exp(-1 / Math.max(1, (attackMs / 1000) * sampleRate));
  const releaseCoeff = Math.exp(-1 / Math.max(1, (releaseMs / 1000) * sampleRate));

  let smoothedGrDb = 0;
  let maxGrDb = 0;
  const out = channels.map((ch) => new Float32Array(ch.length));

  for (let i = 0; i < length; i++) {
    // Stereo-linked peak detector so stereo imaging stays centered
    let peak = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const a = Math.abs(channels[ch][i]);
      if (a > peak) peak = a;
    }
    const inputDb = linearToDb(peak);

    // Static soft-knee gain reduction curve
    let targetGrDb = 0;
    const overshoot = inputDb - thresholdDb;
    if (overshoot > halfKnee) {
      targetGrDb = overshoot * (1 - 1 / ratio);
    } else if (overshoot > -halfKnee) {
      const x = overshoot + halfKnee;
      targetGrDb = ((1 - 1 / ratio) * x * x) / (2 * kneeDb);
    }

    // Attack when gain reduction increases; release when it relaxes
    if (targetGrDb > smoothedGrDb) {
      smoothedGrDb = attackCoeff * smoothedGrDb + (1 - attackCoeff) * targetGrDb;
    } else {
      smoothedGrDb = releaseCoeff * smoothedGrDb + (1 - releaseCoeff) * targetGrDb;
    }

    if (smoothedGrDb > maxGrDb) maxGrDb = smoothedGrDb;

    const gain = dbToLinear(-smoothedGrDb);
    for (let ch = 0; ch < numChannels; ch++) {
      out[ch][i] = channels[ch][i] * gain;
    }
  }

  return { channels: out, maxGainReductionDb: maxGrDb };
}

/**
 * Stage d: Speech-Aware Auto-Leveler (~0.5 s windows)
 * Splits audio into 0.5 s segments, measures each segment's RMS level, and
 * gently rides volume toward the median active speech level. Silent or
 * noise-only windows are ignored (gain glides toward 0 dB) so pauses never
 * swell with background hiss.
 */
function runAutoLeveler(channels, sampleRate, strength = 0.75) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const winSamples = Math.max(1, Math.round(0.5 * sampleRate));
  const numWindows = Math.max(1, Math.ceil(length / winSamples));
  const winRmsDb = new Float32Array(numWindows);

  for (let w = 0; w < numWindows; w++) {
    const start = w * winSamples;
    const end = Math.min(length, start + winSamples);
    let sumSq = 0;
    let count = 0;
    for (let i = start; i < end; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        const s = channels[ch][i];
        sumSq += s * s;
        count++;
      }
    }
    winRmsDb[w] = linearToDb(Math.sqrt(sumSq / Math.max(1, count)));
  }

  // Identify active speech windows above -42 dBFS (or relative to peak - 28 dB)
  const sorted = Float32Array.from(winRmsDb).sort();
  const p90 = sorted[Math.floor(sorted.length * 0.9)];
  const p15 = sorted[Math.floor(sorted.length * 0.15)];
  const speechGateDb = Math.max(-44, Math.min(p90 - 24, p15 + 7));

  const activeLevels = [];
  for (let w = 0; w < numWindows; w++) {
    if (winRmsDb[w] > speechGateDb) activeLevels.push(winRmsDb[w]);
  }

  if (activeLevels.length === 0) {
    return { channels: channels.map((ch) => Float32Array.from(ch)), targetSpeechDb: -20 };
  }

  activeLevels.sort((a, b) => a - b);
  const targetSpeechDb = activeLevels[Math.floor(activeLevels.length * 0.65)];

  // Compute raw gain adjustment per 0.5 s window
  const maxBoostDb = 10 * strength;
  const maxCutDb = -10 * strength;
  const winGainDb = new Float32Array(numWindows);

  for (let w = 0; w < numWindows; w++) {
    if (winRmsDb[w] <= speechGateDb) {
      // Do NOT boost silence or room tone
      winGainDb[w] = 0;
    } else {
      const diff = (targetSpeechDb - winRmsDb[w]) * strength;
      winGainDb[w] = Math.max(maxCutDb, Math.min(maxBoostDb, diff));
    }
  }

  // Smooth window-to-window gain transitions with a 3-point moving average
  const smoothWinGainDb = new Float32Array(numWindows);
  for (let w = 0; w < numWindows; w++) {
    const prev = winGainDb[Math.max(0, w - 1)];
    const curr = winGainDb[w];
    const next = winGainDb[Math.min(numWindows - 1, w + 1)];
    smoothWinGainDb[w] = 0.25 * prev + 0.5 * curr + 0.25 * next;
  }

  // Apply linearly interpolated + 1-pole smoothed gain per sample
  const out = channels.map((ch) => new Float32Array(ch.length));
  const slewCoeff = Math.exp(-1 / (0.12 * sampleRate)); // 120 ms gentle fader ride
  let currentGainDb = smoothWinGainDb[0];

  for (let i = 0; i < length; i++) {
    const pos = i / winSamples;
    const w0 = Math.min(numWindows - 1, Math.floor(pos));
    const w1 = Math.min(numWindows - 1, w0 + 1);
    const frac = pos - w0;
    const desiredDb =
      smoothWinGainDb[w0] * (1 - frac) + smoothWinGainDb[w1] * frac;

    currentGainDb = slewCoeff * currentGainDb + (1 - slewCoeff) * desiredDb;
    const gain = dbToLinear(currentGainDb);

    for (let ch = 0; ch < numChannels; ch++) {
      out[ch][i] = channels[ch][i] * gain;
    }
  }

  return { channels: out, targetSpeechDb };
}

/**
 * Stage e: Loudness Normalization + 4x Oversampled Lookahead True-Peak Limiter
 * 1. Measures current loudness (ITU-R BS.1770 LUFS or RMS dBFS) and applies
 *    uniform makeup gain to reach the exact target.
 * 2. Runs a 5 ms lookahead true-peak limiter with 4x cubic inter-sample peak
 *    detection so output never clips above `ceilingDbtp`.
 */
function runNormalizeAndLimit(
  channels,
  sampleRate,
  { mode = 'lufs', targetLoudness = -16, ceilingDbtp = -1.0 }
) {
  const numChannels = channels.length;
  const length = channels[0].length;

  // Step 1: Measure current loudness
  let currentLevel;
  if (mode === 'rms') {
    let sumSq = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const d = channels[ch];
      for (let i = 0; i < length; i++) sumSq += d[i] * d[i];
    }
    currentLevel = linearToDb(Math.sqrt(sumSq / Math.max(1, length * numChannels)));
  } else {
    const { integratedLufs } = measureBS1770Loudness(channels, sampleRate);
    currentLevel = integratedLufs;
  }

  // Cap extreme makeup gain on silent files (+24 dB max)
  const makeupDb =
    currentLevel <= -68 ? 0 : Math.max(-24, Math.min(24, targetLoudness - currentLevel));
  const makeupLinear = dbToLinear(makeupDb);

  const normalized = channels.map((ch) => {
    const copy = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      copy[i] = ch[i] * makeupLinear;
    }
    return copy;
  });

  // Apply 4x Oversampled Lookahead True-Peak Limiter (reusable helper for 2-pass loudness lock)
  const ceilingLinear = dbToLinear(ceilingDbtp);
  const lookaheadSamples = Math.max(1, Math.round(0.005 * sampleRate)); // 5 ms
  const releaseCoeff = Math.exp(-1 / (0.06 * sampleRate)); // 60 ms release
  const attackCoeff = Math.exp(-1 / Math.max(1, 0.0015 * sampleRate));
  const fracs = [0.25, 0.5, 0.75];

  const applyTruePeakLimiter = (inputChs) => {
    const peakEnv = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      let maxP = 0;
      for (let ch = 0; ch < numChannels; ch++) {
        const d = inputChs[ch];
        const absV = Math.abs(d[i]);
        if (absV > maxP) maxP = absV;

        if (absV > ceilingLinear * 0.75 && i >= 1 && i + 2 < length) {
          const y0 = d[i - 1];
          const y1 = d[i];
          const y2 = d[i + 1];
          const y3 = d[i + 2];
          for (let f = 0; f < 3; f++) {
            const t = fracs[f];
            const t2 = t * t;
            const t3 = t2 * t;
            const interp =
              0.5 *
              (2 * y1 +
                (-y0 + y2) * t +
                (2 * y0 - 5 * y1 + 4 * y2 - y3) * t2 +
                (-y0 + 3 * y1 - 3 * y2 + y3) * t3);
            const absI = Math.abs(interp);
            if (absI > maxP) maxP = absI;
          }
        }
      }
      peakEnv[i] = maxP;
    }

    // Sliding maximum over [i, i + lookaheadSamples] using monotonic deque in O(N)
    const desiredGain = new Float32Array(length);
    const deque = new Int32Array(length);
    let head = 0;
    let tail = 0;

    for (let i = length - 1; i >= 0; i--) {
      const val = peakEnv[i];
      while (tail > head && peakEnv[deque[tail - 1]] <= val) {
        tail--;
      }
      deque[tail++] = i;
      if (deque[head] > i + lookaheadSamples) {
        head++;
      }
      const windowPeak = peakEnv[deque[head]];
      desiredGain[i] = windowPeak > ceilingLinear ? ceilingLinear / windowPeak : 1.0;
    }

    const limited = inputChs.map(() => new Float32Array(length));
    let gain = 1.0;
    let maxGrDb = 0;

    for (let i = 0; i < length; i++) {
      const target = desiredGain[i];
      if (target < gain) {
        gain = Math.min(target, attackCoeff * gain + (1 - attackCoeff) * target);
      } else {
        gain = releaseCoeff * gain + (1 - releaseCoeff) * target;
      }

      const grDb = -linearToDb(gain);
      if (grDb > maxGrDb) maxGrDb = grDb;

      for (let ch = 0; ch < numChannels; ch++) {
        let s = inputChs[ch][i] * gain;
        if (s > ceilingLinear) s = ceilingLinear;
        else if (s < -ceilingLinear) s = -ceilingLinear;
        limited[ch][i] = s;
      }
    }
    return { channels: limited, maxGrDb };
  };

  let pass1 = applyTruePeakLimiter(normalized);
  let out = pass1.channels;
  let limiterMaxGrDb = pass1.maxGrDb;
  let totalMakeupDb = makeupDb;

  // Step 3: 2nd-pass loudness target lock (if heavy limiting pulled integrated level below target)
  if (currentLevel > -68 && mode === 'lufs') {
    const postLufs = measureBS1770Loudness(out, sampleRate).integratedLufs;
    const shortfall = targetLoudness - postLufs;
    if (shortfall > 0.35 && shortfall < 8) {
      const topUpDb = Math.min(3.5, shortfall);
      const topUpLin = dbToLinear(topUpDb);
      const topped = out.map((ch) => {
        const arr = new Float32Array(length);
        for (let i = 0; i < length; i++) arr[i] = ch[i] * topUpLin;
        return arr;
      });
      const pass2 = applyTruePeakLimiter(topped);
      out = pass2.channels;
      limiterMaxGrDb = Math.max(limiterMaxGrDb, pass2.maxGrDb);
      totalMakeupDb += topUpDb;
    }
  }

  return { channels: out, makeupDb: totalMakeupDb, limiterMaxGrDb };
}

/**
 * Stage f: Optional Silence Trimming
 * Finds pauses quieter than -45 dBFS that last longer than `maxPauseSec` and
 * trims the excess middle portion, leaving a natural brief pause joined with
 * a 25 ms raised-cosine crossfade so edits are seamless and click-free.
 */
function runSilenceTrimming(channels, sampleRate, maxPauseSec = 1.0) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const frameSamples = Math.max(1, Math.round(0.025 * sampleRate)); // 25 ms frames
  const numFrames = Math.ceil(length / frameSamples);
  const silenceThreshDb = -45;

  const isSilent = new Uint8Array(numFrames);
  for (let f = 0; f < numFrames; f++) {
    const start = f * frameSamples;
    const end = Math.min(length, start + frameSamples);
    let sumSq = 0;
    let count = 0;
    for (let i = start; i < end; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        const s = channels[ch][i];
        sumSq += s * s;
        count++;
      }
    }
    isSilent[f] = linearToDb(Math.sqrt(sumSq / Math.max(1, count))) < silenceThreshDb ? 1 : 0;
  }

  const maxPauseFrames = Math.max(4, Math.round(maxPauseSec / 0.025));
  // Keep a natural residual pause (up to 0.40s) rather than slamming phrases together
  const keepHalfFrames = Math.max(2, Math.round(Math.min(0.4, maxPauseSec * 0.45) / 0.025 / 2));

  // Build keep segments [startSample, endSample]
  const segments = [];
  let segStart = 0;
  let f = 0;
  let trimmedSeconds = 0;

  while (f < numFrames) {
    if (isSilent[f] === 1) {
      const runStart = f;
      while (f < numFrames && isSilent[f] === 1) f++;
      const runLen = f - runStart;

      if (runLen > maxPauseFrames) {
        const cutStart = Math.min(length, (runStart + keepHalfFrames) * frameSamples);
        const cutEnd = Math.min(length, (f - keepHalfFrames) * frameSamples);
        if (cutEnd > cutStart + frameSamples * 2) {
          segments.push([segStart, cutStart]);
          trimmedSeconds += (cutEnd - cutStart) / sampleRate;
          segStart = cutEnd;
        }
      }
    } else {
      f++;
    }
  }
  if (segStart < length) {
    segments.push([segStart, length]);
  }

  // If nothing was trimmed (or entire file was silence), return original
  if (segments.length <= 1 && trimmedSeconds === 0) {
    return { channels, trimmedSeconds: 0 };
  }

  const fadeSamples = Math.round(0.025 * sampleRate); // 25 ms crossfade
  let totalLen = 0;
  for (let s = 0; s < segments.length; s++) {
    const segLen = segments[s][1] - segments[s][0];
    totalLen += segLen;
    if (s > 0) {
      totalLen -= Math.min(fadeSamples, segLen);
    }
  }

  if (totalLen < Math.round(0.1 * sampleRate)) {
    return { channels, trimmedSeconds: 0 };
  }

  const out = channels.map(() => new Float32Array(totalLen));
  let writePos = 0;

  for (let s = 0; s < segments.length; s++) {
    const [start, end] = segments[s];
    const segLen = end - start;
    const overlap = s > 0 ? Math.min(fadeSamples, segLen, writePos) : 0;

    // Crossfade overlapping splice region using raised-cosine curves
    for (let i = 0; i < overlap; i++) {
      const t = i / overlap;
      const fadeOut = 0.5 * (1 + Math.cos(Math.PI * t));
      const fadeIn = 0.5 * (1 - Math.cos(Math.PI * t));
      const destIdx = writePos - overlap + i;
      for (let ch = 0; ch < numChannels; ch++) {
        out[ch][destIdx] =
          out[ch][destIdx] * fadeOut + channels[ch][start + i] * fadeIn;
      }
    }

    // Copy remaining non-overlapping samples of this segment
    const remStart = overlap;
    for (let i = remStart; i < segLen; i++) {
      const destIdx = writePos - overlap + i;
      for (let ch = 0; ch < numChannels; ch++) {
        out[ch][destIdx] = channels[ch][start + i];
      }
    }
    writePos = writePos - overlap + segLen;
  }

  return { channels: out, trimmedSeconds };
}

/**
 * Executes the complete deterministic processing pipeline on an `AudioBuffer`.
 *
 * @param {AudioBuffer} inputBuffer
 * @param {object} options - Processing stage toggles & parameters
 * @param {(progress: { percent: number, stage: string }) => void} onProgress
 * @returns {Promise<{ audioBuffer: AudioBuffer, report: object }>}
 */
export async function processAudioBuffer(inputBuffer, options, onProgress) {
  const sampleRate = inputBuffer.sampleRate;
  const numChannels = inputBuffer.numberOfChannels;
  let channels = [];
  for (let ch = 0; ch < numChannels; ch++) {
    channels.push(Float32Array.from(inputBuffer.getChannelData(ch)));
  }

  const report = {
    estimatedNoiseFloorDb: null,
    gateThresholdDb: null,
    compressorMaxGrDb: 0,
    levelerTargetDb: null,
    normalizeMakeupDb: 0,
    limiterMaxGrDb: 0,
    trimmedSeconds: 0,
  };

  const notify = async (percent, stage) => {
    if (onProgress) onProgress({ percent, stage });
    await yieldToUI();
  };

  // Stage a: High-pass filter
  if (options.highPassEnabled) {
    await notify(12, `Stage A: High-pass filter (${options.highPassFreq} Hz)...`);
    channels = runHighPassFilter(channels, sampleRate, options.highPassFreq);
  }

  // Stage b: Noise reduction (Spectral Wiener STFT or Simple Noise Gate)
  if (options.noiseGateEnabled) {
    const useSpectral = (options.noiseMode || 'spectral') === 'spectral';
    await notify(
      26,
      useSpectral
        ? `Stage B: STFT Spectral Wiener denoising (-${options.noiseGateReductionDb} dB)...`
        : `Stage B: Estimating noise floor & gating (-${options.noiseGateReductionDb} dB)...`
    );
    const res = useSpectral
      ? runSpectralDenoiser(channels, sampleRate, options.noiseGateReductionDb)
      : runNoiseGate(
          channels,
          sampleRate,
          options.noiseGateReductionDb,
          options.noiseGateMarginDb ?? 6
        );
    channels = res.channels;
    report.estimatedNoiseFloorDb = res.noiseFloorDb;
    report.gateThresholdDb = res.thresholdDb;
  }

  // Stage b2: Studio Voice Clarity EQ & Split-Band De-Esser
  if (options.voiceClarityEnabled) {
    await notify(
      38,
      `Voice Clarity EQ & De-Esser (${Math.round((options.voiceClarityAmount ?? 0.7) * 100)}%)...`
    );
    channels = runVoiceClarityEqAndDeEsser(
      channels,
      sampleRate,
      options.voiceClarityAmount ?? 0.7
    );
  }

  // Stage c: Compressor
  if (options.compressorEnabled) {
    await notify(
      46,
      `Stage C: Dynamics compression (${options.compThresholdDb} dB, ${options.compRatio}:1)...`
    );
    const res = runCompressor(channels, sampleRate, {
      thresholdDb: options.compThresholdDb,
      ratio: options.compRatio,
      attackMs: options.compAttackMs,
      releaseMs: options.compReleaseMs,
    });
    channels = res.channels;
    report.compressorMaxGrDb = res.maxGainReductionDb;
  }

  // Stage d: Auto-leveler
  if (options.levelerEnabled) {
    await notify(64, 'Stage D: Auto-leveler (0.5 s speech windows)...');
    const res = runAutoLeveler(channels, sampleRate, options.levelerStrength ?? 0.75);
    channels = res.channels;
    report.levelerTargetDb = res.targetSpeechDb;
  }

  // Stage e: Loudness normalization + True-peak limiter
  if (options.normalizeEnabled) {
    const unit = options.normalizeMode === 'rms' ? 'dBFS RMS' : 'LUFS';
    await notify(
      80,
      `Stage E: Normalizing to ${options.targetLoudness} ${unit} + True-peak limiter...`
    );
    const res = runNormalizeAndLimit(channels, sampleRate, {
      mode: options.normalizeMode,
      targetLoudness: options.targetLoudness,
      ceilingDbtp: options.limiterCeilingDbtp,
    });
    channels = res.channels;
    report.normalizeMakeupDb = res.makeupDb;
    report.limiterMaxGrDb = res.limiterMaxGrDb;
  }

  // Stage f: Optional silence trimming
  if (options.silenceTrimEnabled) {
    await notify(
      92,
      `Stage F: Trimming pauses longer than ${options.silenceMaxPauseSec.toFixed(1)} s...`
    );
    const res = runSilenceTrimming(
      channels,
      sampleRate,
      options.silenceMaxPauseSec
    );
    channels = res.channels;
    report.trimmedSeconds = res.trimmedSeconds;
  }

  await notify(98, 'Rebuilding output AudioBuffer & analyzing...');

  // Construct deterministic Web Audio AudioBuffer via OfflineAudioContext
  const outLength = Math.max(1, channels[0].length);
  const offlineCtx = new OfflineAudioContext(numChannels, outLength, sampleRate);
  const outputBuffer = offlineCtx.createBuffer(numChannels, outLength, sampleRate);
  for (let ch = 0; ch < numChannels; ch++) {
    outputBuffer.copyToChannel(channels[ch], ch);
  }

  return { audioBuffer: outputBuffer, report };
}

/**
 * Encodes a Web Audio `AudioBuffer` into a standard 16-bit PCM RIFF/WAVE Blob.
 * Interleaves channels, converts [-1.0, 1.0] floats to signed 16-bit integers,
 * and writes the 44-byte WAV header.
 */
export function encodeWav16Bit(audioBuffer) {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const numFrames = audioBuffer.length;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataByteLength = numFrames * blockAlign;

  const buffer = new ArrayBuffer(44 + dataByteLength);
  const view = new DataView(buffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  // RIFF chunk descriptor
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataByteLength, true);
  writeString(8, 'WAVE');

  // fmt sub-chunk (PCM = format 1, 16-bit)
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);

  // data sub-chunk
  writeString(36, 'data');
  view.setUint32(40, dataByteLength, true);

  const channels = [];
  for (let ch = 0; ch < numChannels; ch++) {
    channels.push(audioBuffer.getChannelData(ch));
  }

  let offset = 44;
  for (let i = 0; i < numFrames; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const sample = Math.max(-1, Math.min(1, channels[ch][i]));
      const int16 = sample < 0 ? Math.round(sample * 0x8000) : Math.round(sample * 0x7fff);
      view.setInt16(offset, int16, true);
      offset += 2;
    }
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Generates an 8.5-second realistic synthetic speech-like test recording so users
 * can immediately experiment with every DSP stage without hunting for a file:
 * - Contains 50 Hz room rumble (visible in spectrogram, removed by High-Pass)
 * - Contains broadband background hiss (visible in spectrogram, reduced by Gate)
 * - Contains a quiet spoken phrase + a loud spoken phrase with a clipping spike
 *   (demonstrates Auto-Leveler, Compressor, Normalization & True-Peak Limiter)
 * - Contains a 2.2 s dead pause in the middle (demonstrates Silence Trimming)
 */
export function generateDemoAudioBuffer(audioContext) {
  const sampleRate = audioContext.sampleRate || 44100;
  const durationSec = 8.5;
  const length = Math.round(durationSec * sampleRate);
  const buffer = audioContext.createBuffer(2, length, sampleRate);
  const left = buffer.getChannelData(0);
  const right = buffer.getChannelData(1);

  // Seeded PRNG so demo audio is deterministic
  let seed = 42197;
  const nextRand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 0xffffffff - 0.5;
  };

  // Three speech phrases with uneven loudness and a 2.2s pause between phrase 2 and 3
  const phrases = [
    { start: 0.4, end: 2.3, amp: 0.14, baseFreq: 132 }, // Quiet phrase
    { start: 2.6, end: 4.3, amp: 0.62, baseFreq: 158 }, // Loud dynamic phrase
    { start: 6.5, end: 8.1, amp: 0.24, baseFreq: 142 }, // Medium phrase after 2.2s pause
  ];

  let hissFilter = 0;

  for (let i = 0; i < length; i++) {
    const t = i / sampleRate;

    // 1. Low-frequency HVAC / desk rumble (48 Hz + 62 Hz)
    const rumble =
      0.045 * Math.sin(2 * Math.PI * 48 * t) +
      0.025 * Math.sin(2 * Math.PI * 62 * t + 0.7);

    // 2. Steady room hiss (lightly shaped white noise)
    const white = nextRand();
    hissFilter = 0.65 * hissFilter + 0.35 * white;
    const hiss = (white * 0.55 + hissFilter * 0.45) * 0.032;

    // 3. Synthetic formant speech bursts (syllable modulation + harmonic stack)
    let voice = 0;
    for (let p = 0; p < phrases.length; p++) {
      const ph = phrases[p];
      if (t >= ph.start && t <= ph.end) {
        const localT = t - ph.start;
        const phraseLen = ph.end - ph.start;
        const edgeEnv =
          Math.min(1, localT / 0.06) * Math.min(1, (phraseLen - localT) / 0.08);
        // Syllable rhythm (~4.2 Hz)
        const syllable = Math.max(0, Math.sin(2 * Math.PI * 4.2 * localT)) ** 0.7;
        const pitch = ph.baseFreq * (1 + 0.08 * Math.sin(2 * Math.PI * 1.4 * localT));
        const harmonics =
          0.55 * Math.sin(2 * Math.PI * pitch * t) +
          0.32 * Math.sin(2 * Math.PI * pitch * 2.0 * t + 0.4) +
          0.22 * Math.sin(2 * Math.PI * pitch * 3.0 * t + 1.1) +
          0.16 * Math.sin(2 * Math.PI * pitch * 5.0 * t + 0.2) +
          0.09 * Math.sin(2 * Math.PI * pitch * 8.0 * t);

        // Add a brief hot transient spike at t = 3.35s to show clipping & limiting
        const spike =
          p === 1 && Math.abs(t - 3.35) < 0.03
            ? 0.55 * Math.cos(((t - 3.35) / 0.03) * (Math.PI / 2)) * Math.sin(2 * Math.PI * 380 * t)
            : 0;

        voice += ph.amp * edgeEnv * (0.25 + 0.75 * syllable) * harmonics + spike;
      }
    }

    const rawL = Math.max(-1, Math.min(1, rumble + hiss + voice));
    const rawR = Math.max(-1, Math.min(1, rumble + hiss * 0.92 + voice * 0.98));
    left[i] = rawL;
    right[i] = rawR;
  }

  return buffer;
}
