/**
 * processor.js — Broadcast-Grade Automatic Audio Mastering Pipeline
 *
 * Plain-language overview of the processing chain:
 *   a. 4th-Order High-Pass Filter (24 dB/oct) — Steeply eliminates sub-bass
 *      rumble, wind pops, and AC hum below the cutoff without thinning speech.
 *   b. Noise Reduction (STFT Spectral Wiener OR Speech-Relative Soft Expander/Gate) —
 *      Removes steady fan/AC hiss during speech AND softly pushes down quiet
 *      room chatter/breaths between phrases (referenced relative to active
 *      speech level so hardware-gated silence never breaks floor estimation).
 *   +. Adaptive Studio Voice Clarity EQ & Split-Band De-Esser — Measures the
 *      recording's actual low-mid vs. presence spectral tilt! Automatically
 *      trims boxy mud on dark laptop/Bluetooth mics, restores crisp 2.8–6 kHz
 *      vocal articulation & air, and dynamically softens harsh "s/sh" spikes.
 *   d. Smooth Phrase Auto-Leveler (runs BEFORE Compressor for zero pumping!) —
 *      Uses a slow continuous speech-activity follower that holds gain cleanly
 *      through syllables and gently rides quiet phrases up to match loud ones
 *      without ever surging background breaths or causing mid-word pumping.
 *   c. Soft-Knee Broadcast Compressor — Gently controls peaks with stereo-linked
 *      smooth ballistics so speech sounds dense, clear, and radio-ready.
 *   e. 2-Pass Loudness Normalization + 4x Oversampled Lookahead True-Peak Limiter —
 *      Nails exact ITU-R BS.1770-4 LUFS (or Audiobook RMS) while preventing
 *      inter-sample clipping above the chosen dBTP ceiling.
 *   f. Optional Silence Trimming — Shortens pauses longer than N seconds using
 *      smooth 25 ms raised-cosine crossfades.
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
    summary: 'Best quality: DeepFilterNet3 (speech-safe) + Adaptive Clarity EQ + -15 LUFS',
    recommendedEngine: 'deepfilternet',
    dfOptions: {
      attenuationLimit: 48,
      postFilterBeta: 0.0,
      chainMastering: true,
    },
    options: {
      highPassEnabled: true,
      highPassFreq: 85,
      noiseGateEnabled: true,
      noiseMode: 'spectral',
      noiseGateReductionDb: 16,
      noiseGateMarginDb: 6,
      voiceClarityEnabled: true,
      voiceClarityAmount: 0.8,
      compressorEnabled: true,
      compThresholdDb: -19,
      compRatio: 2.6,
      compAttackMs: 10,
      compReleaseMs: 120,
      levelerEnabled: true,
      levelerStrength: 0.75,
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
    summary: '-16 LUFS, spectral noise cleanup, adaptive voice clarity, smooth dynamics',
    dfOptions: {
      attenuationLimit: 45,
      postFilterBeta: 0.0,
      chainMastering: true,
    },
    options: {
      highPassEnabled: true,
      highPassFreq: 85,
      noiseGateEnabled: true,
      noiseMode: 'spectral',
      noiseGateReductionDb: 16,
      noiseGateMarginDb: 6,
      voiceClarityEnabled: true,
      voiceClarityAmount: 0.75,
      compressorEnabled: true,
      compThresholdDb: -20,
      compRatio: 2.5,
      compAttackMs: 10,
      compReleaseMs: 130,
      levelerEnabled: true,
      levelerStrength: 0.7,
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
      attenuationLimit: 32,
      postFilterBeta: 0.0,
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
      voiceClarityAmount: 0.45,
      compressorEnabled: true,
      compThresholdDb: -22,
      compRatio: 2.0,
      compAttackMs: 16,
      compReleaseMs: 170,
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
      attenuationLimit: 55,
      postFilterBeta: 0.005,
      chainMastering: true,
    },
    options: {
      highPassEnabled: true,
      highPassFreq: 90,
      noiseGateEnabled: true,
      noiseMode: 'spectral',
      noiseGateReductionDb: 20,
      noiseGateMarginDb: 6,
      voiceClarityEnabled: true,
      voiceClarityAmount: 0.9,
      compressorEnabled: true,
      compThresholdDb: -18,
      compRatio: 3.0,
      compAttackMs: 7,
      compReleaseMs: 110,
      levelerEnabled: true,
      levelerStrength: 0.8,
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
      attenuationLimit: 38,
      postFilterBeta: 0.0,
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
      voiceClarityAmount: 0.55,
      compressorEnabled: true,
      compThresholdDb: -24,
      compRatio: 2.2,
      compAttackMs: 15,
      compReleaseMs: 160,
      levelerEnabled: true,
      levelerStrength: 0.6,
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
    summary: 'Heavy rescue: 100 Hz HPF, strong denoising, clarity EQ, trims >1.0 s',
    dfOptions: {
      attenuationLimit: 72,
      postFilterBeta: 0.01,
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
      voiceClarityAmount: 0.8,
      compressorEnabled: true,
      compThresholdDb: -20,
      compRatio: 3.0,
      compAttackMs: 8,
      compReleaseMs: 120,
      levelerEnabled: true,
      levelerStrength: 0.8,
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

function yieldToUI() {
  return new Promise((resolve) => setTimeout(resolve, 16));
}

/**
 * Helper: Estimates active speech RMS level (85th percentile of non-silent 50ms frames).
 * Anchoring thresholds to active speech prevents hardware-muted dead silence (-80 dB)
 * from fooling the noise gate or auto-leveler.
 */
function measureActiveSpeechRmsDb(channels, sampleRate) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const frameSamples = Math.max(1, Math.round(0.05 * sampleRate));
  const numFrames = Math.max(1, Math.ceil(length / frameSamples));
  const activeRmsDbs = [];

  const loudSpeechDbs = [];
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
    const db = linearToDb(Math.sqrt(sumSq / Math.max(1, count)));
    if (db > -42) loudSpeechDbs.push(db);
    if (db > -55) activeRmsDbs.push(db);
  }

  const pool = loudSpeechDbs.length >= 6 ? loudSpeechDbs : activeRmsDbs;
  if (pool.length === 0) return -28;
  pool.sort((a, b) => a - b);
  return pool[Math.floor(pool.length * 0.8)];
}

/**
 * Stage a: 4th-Order Cascaded Butterworth High-Pass Filter (24 dB/octave)
 * Steeply eliminates sub-bass rumble, desk bumps, and AC hum below `cutoffHz`.
 */
function runHighPassFilter(channels, sampleRate, cutoffHz) {
  const f0 = Math.max(20, Math.min(cutoffHz, sampleRate * 0.4));
  const w0 = (2 * Math.PI * f0) / sampleRate;
  const cosW0 = Math.cos(w0);
  const sinW0 = Math.sin(w0);

  // Two cascaded 2nd-order sections with Butterworth 4th-order Q factors
  const qs = [0.5411961, 1.306563];
  let out = channels;
  for (let s = 0; s < qs.length; s++) {
    const alpha = sinW0 / (2 * qs[s]);
    const a0 = 1 + alpha;
    const coeffs = {
      b0: ((1 + cosW0) / 2) / a0,
      b1: -(1 + cosW0) / a0,
      b2: ((1 + cosW0) / 2) / a0,
      a1: (-2 * cosW0) / a0,
      a2: (1 - alpha) / a0,
    };
    out = out.map((ch) => applyBiquad(ch, coeffs));
  }
  return out;
}

/**
 * Stage b (Mode 1): Speech-Relative Soft Noise Gate / Expander
 * Uses BOTH estimated quiet-frame floor AND active-speech reference so quiet
 * background chatter/breaths (-45 to -60 dBFS) are smoothly attenuated even
 * when a microphone has digital dead-silence (-80 dBFS) pauses.
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

  // Ignore hardware-muted digital silence (< -72 dBFS) when estimating room floor
  const roomFrames = [];
  for (let i = 0; i < numFrames; i++) {
    if (frameRmsDb[i] > -72) roomFrames.push(frameRmsDb[i]);
  }
  roomFrames.sort((a, b) => a - b);

  const speechRefDb = measureActiveSpeechRmsDb(channels, sampleRate);
  const floorFromRoom =
    roomFrames.length > 0
      ? roomFrames[Math.floor(roomFrames.length * 0.12)]
      : -65;

  // Place threshold cleanly below active speech (silences -45..-60 dB background chatter/breaths
  // while keeping soft foreground speech down to speechRef - 18 dB completely intact)
  const speechRelativeThresh = speechRefDb - 20;
  const thresholdDb = Math.max(
    -56,
    Math.min(speechRelativeThresh, Math.max(floorFromRoom + marginDb, speechRefDb - 23))
  );
  const minGain = dbToLinear(-Math.abs(reductionDb));

  // 20 ms lookahead + 5 ms attack + 260 ms hold + 240 ms smooth release
  // ensures word onsets open BEFORE speech begins and mid-word dips never break.
  const lookaheadSamples = Math.round(0.02 * sampleRate);
  const attackCoeff = Math.exp(-1 / (0.005 * sampleRate));
  const releaseCoeff = Math.exp(-1 / (0.24 * sampleRate));
  const rmsDetectCoeff = Math.exp(-1 / (0.010 * sampleRate));
  const holdSamples = Math.round(0.26 * sampleRate);

  let envSq = 0;
  let gain = 1.0;
  let holdCounter = 0;

  const out = channels.map((ch) => new Float32Array(ch.length));

  for (let i = 0; i < length; i++) {
    const detIdx = Math.min(length - 1, i + lookaheadSamples);
    let linkedSq = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const s = channels[ch][detIdx];
      linkedSq += s * s;
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
      // Wide 12 dB soft knee prevents abrupt choppy word tails
      const belowDb = thresholdDb - levelDb;
      const ratio = Math.min(1, belowDb / 12);
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

  return { channels: out, noiseFloorDb: floorFromRoom, thresholdDb };
}

/**
 * Stage b (Mode 2): STFT Spectral Wiener / Ephraim-Malah Noise Reduction
 * Removes steady fan hum, AC noise, and broadband room hiss *even while speaking*
 * using 1024-pt Hann 50% overlap-add STFT, room-floor profiling (ignoring digital
 * dead-silence frames < -72 dBFS), decision-directed SNR smoothing, and a clean
 * speech-relative pause expander.
 */
function runSpectralDenoiser(channels, sampleRate, reductionDb = 18) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const fftSize = 1024;
  const hop = fftSize >> 1;
  const halfBins = (fftSize >> 1) + 1;

  const sqrtWin = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    sqrtWin[i] = Math.sqrt(0.5 * (1 - Math.cos((2 * Math.PI * i) / fftSize)));
  }

  const minGain = dbToLinear(-Math.max(4, Math.min(32, reductionDb)));
  const overSub = 1.15 + Math.min(0.75, reductionDb / 32);
  const alphaDD = 0.94; // Smooth decision-directed prior keeps vocals completely natural

  const outChannels = [];
  let measuredFloorDb = -60;

  for (let ch = 0; ch < numChannels; ch++) {
    const src = channels[ch];
    const numFrames = Math.max(1, Math.ceil((length + hop) / hop));

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

    // Ignore hardware-muted dead silence (< -72 dBFS) when locating real room noise frames
    const minRoomEnergy = Math.pow(10, -72 / 10);
    const frameOrder = Array.from({ length: numFrames }, (_, i) => i).filter(
      (i) => frameEnergy[i] > minRoomEnergy
    );
    frameOrder.sort((a, b) => frameEnergy[a] - frameEnergy[b]);

    const quietCount = Math.max(
      4,
      Math.min(frameOrder.length, Math.ceil(frameOrder.length * 0.15))
    );
    const noisePsd = new Float32Array(halfBins);
    const real = new Float32Array(fftSize);
    const imag = new Float32Array(fftSize);

    if (quietCount > 0 && frameOrder.length >= 4) {
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
        const priorSnr =
          alphaDD * (prevGain[b] * prevGain[b] * prevPostSnr[b]) +
          (1 - alphaDD) * Math.max(postSnr - 1, 0);

        const wiener = priorSnr / (priorSnr + 1);
        binGain[b] = Math.max(minGain, Math.min(1, wiener));
        prevPostSnr[b] = Math.min(postSnr, 50);
      }

      for (let b = 0; b < halfBins; b++) {
        const gPrev = binGain[Math.max(0, b - 1)];
        const gCurr = binGain[b];
        const gNext = binGain[Math.min(halfBins - 1, b + 1)];
        const g = 0.2 * gPrev + 0.6 * gCurr + 0.2 * gNext;
        smoothGain[b] = g;
        prevGain[b] = g;
      }

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

  // Smooth speech-relative pause expander cleans up distant TV/chatter between phrases
  const polished = runNoiseGate(
    outChannels,
    sampleRate,
    Math.min(24, Math.max(14, Math.round(reductionDb * 0.95))),
    6
  );

  return {
    channels: polished.channels,
    noiseFloorDb: measuredFloorDb,
    thresholdDb: polished.thresholdDb,
  };
}

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

function makeLowShelfCoeffs(sampleRate, freqHz, gainDb, slope = 0.8) {
  const f0 = Math.max(40, Math.min(freqHz, sampleRate * 0.4));
  const A = Math.pow(10, gainDb / 40);
  const w0 = (2 * Math.PI * f0) / sampleRate;
  const cosW0 = Math.cos(w0);
  const sinW0 = Math.sin(w0);
  const alpha = (sinW0 / 2) * Math.sqrt((A + 1 / A) * (1 / slope - 1) + 2);
  const twoSqrtAAlpha = 2 * Math.sqrt(A) * alpha;
  const a0 = A + 1 + (A - 1) * cosW0 + twoSqrtAAlpha;
  return {
    b0: (A * (A + 1 - (A - 1) * cosW0 + twoSqrtAAlpha)) / a0,
    b1: (2 * A * (A - 1 - (A + 1) * cosW0)) / a0,
    b2: (A * (A + 1 - (A - 1) * cosW0 - twoSqrtAAlpha)) / a0,
    a1: (-2 * (A - 1 + (A + 1) * cosW0)) / a0,
    a2: (A + 1 + (A - 1) * cosW0 - twoSqrtAAlpha) / a0,
  };
}

/**
 * Adaptive Studio Voice Clarity EQ + Split-Band De-Esser
 * Measures the recording's actual spectral tilt during active speech:
 *   - Dark / muffled laptop & Bluetooth mics (heavy 150–500 Hz vs weak 2–6 kHz)
 *     receive a deeper de-mud cut (-3.5 to -5.5 dB @ 230 Hz) and stronger
 *     presence & articulation lift (+4 to +7 dB @ 2.8 kHz + 6.2 kHz shelf).
 *   - Bright condenser mics receive gentle sculpting without becoming harsh.
 *   - Always followed by a split-band dynamic de-esser (5.8 kHz+) so sibilants
 *     remain smooth and natural.
 */
function runVoiceClarityEqAndDeEsser(channels, sampleRate, amount = 0.75) {
  const amt = Math.max(0, Math.min(1, amount));
  if (amt <= 0.01) {
    return channels;
  }

  // Analyze spectral balance during active speech frames
  const mono = channels[0];
  const N = 2048;
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  let lowMidPow = 0; // 150–500 Hz (mud / boxiness)
  let presencePow = 0; // 1800–5500 Hz (clarity / intelligibility)
  let speechFrames = 0;

  const speechThresh = dbToLinear(measureActiveSpeechRmsDb(channels, sampleRate) - 10);
  const step = Math.max(1024, Math.floor(mono.length / 80));

  for (let s = 0; s + N <= mono.length; s += step) {
    let sq = 0;
    for (let i = 0; i < N; i++) sq += mono[s + i] * mono[s + i];
    if (Math.sqrt(sq / N) < speechThresh) continue;

    for (let i = 0; i < N; i++) {
      re[i] = mono[s + i] * 0.5 * (1 - Math.cos((2 * Math.PI * i) / N));
      im[i] = 0;
    }
    fftRadix2(re, im);

    for (let k = 1; k < N / 2; k++) {
      const f = (k * sampleRate) / N;
      const p = re[k] * re[k] + im[k] * im[k];
      if (f >= 150 && f <= 500) lowMidPow += p;
      else if (f >= 1800 && f <= 5500) presencePow += p;
    }
    speechFrames++;
  }

  // Calculate how dark/muffled the mic is (typical studio mic ~10 dB tilt; muffled laptop mic ~18–24 dB)
  let darknessFactor = 0.35; // 0 = already bright, 1 = heavily muffled mic
  if (speechFrames > 0 && lowMidPow > 1e-10) {
    const tiltDb = 10 * Math.log10((lowMidPow + 1e-12) / (presencePow + 1e-12));
    darknessFactor = Math.max(0, Math.min(1, (tiltDb - 9) / 13));
  }

  const deMudDb = -(2.2 + 3.2 * darknessFactor) * amt; // -2.2 dB to -5.4 dB @ 260 Hz
  const deBoomDb = darknessFactor > 0.25 ? -(2.8 * (darknessFactor - 0.25) / 0.75) * amt : 0; // Gentle low-shelf cut below 190 Hz on boomy mics
  const presenceDb = (2.6 + 4.4 * darknessFactor) * amt; // +2.6 dB to +7.0 dB @ 2.9 kHz
  const brillianceDb = (1.6 + 2.8 * darknessFactor) * amt; // +1.6 dB to +4.4 dB @ 6.2 kHz shelf
  // Only add subtle warmth if mic isn't already boomy/dark
  const warmthDb = (1.2 * (1 - darknessFactor)) * amt;

  const deBoom = deBoomDb < -0.2 ? makeLowShelfCoeffs(sampleRate, 190, deBoomDb, 0.75) : null;
  const deMud = makePeakingCoeffs(sampleRate, 260, deMudDb, 0.95);
  const presence = makePeakingCoeffs(sampleRate, 2900, presenceDb, 0.85);
  const brilliance = makeHighShelfCoeffs(sampleRate, 6200, brillianceDb, 0.75);
  const warmth = warmthDb > 0.15 ? makePeakingCoeffs(sampleRate, 130, warmthDb, 1.0) : null;

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
    let x = deBoom ? applyBiquad(ch, deBoom) : ch;
    x = applyBiquad(x, deMud);
    if (warmth) x = applyBiquad(x, warmth);
    x = applyBiquad(x, presence);
    x = applyBiquad(x, brilliance);
    return x;
  });

  // Dynamic split-band de-esser: softly attenuates sharp 5.8 kHz+ sibilants only
  const attackCoeff = Math.exp(-1 / Math.max(1, 0.002 * sampleRate));
  const releaseCoeff = Math.exp(-1 / Math.max(1, 0.045 * sampleRate));
  const sibThresholdDb = -22;
  const maxDeEssDb = 6.5 * amt;

  return eqChannels.map((ch) => {
    const highs = applyBiquad(ch, sibHp);
    const out = new Float32Array(ch.length);
    let env = 0;

    for (let i = 0; i < ch.length; i++) {
      const a = Math.abs(highs[i]);
      env =
        a > env
          ? attackCoeff * env + (1 - attackCoeff) * a
          : releaseCoeff * env + (1 - releaseCoeff) * a;
      const sibDb = linearToDb(env);
      let attenGain = 1.0;
      if (sibDb > sibThresholdDb) {
        const grDb = Math.min(maxDeEssDb, (sibDb - sibThresholdDb) * 0.6);
        attenGain = dbToLinear(-grDb);
      }
      const lowMid = ch[i] - highs[i];
      out[i] = lowMid + highs[i] * attenGain;
    }
    return out;
  });
}

/**
 * Stage d: Smooth Phrase-Aware Auto-Leveler (Runs BEFORE Compressor — Zero Pumping!)
 * Measures active phrases in overlapping ~0.4 s windows, computes a smooth target
 * speech level, and glides gain slowly across phrases while HOLDING/FREEZING gain
 * gently across brief pauses (instead of snapping to 0 dB mid-word!). Prevents
 * volume pumping and prevents boosting breaths or quiet background noise.
 */
function runAutoLeveler(channels, sampleRate, strength = 0.75) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const hopSamples = Math.max(1, Math.round(0.2 * sampleRate)); // 200 ms step
  const winSamples = Math.max(1, Math.round(0.6 * sampleRate)); // 600 ms overlapping window
  const numSteps = Math.max(1, Math.ceil(length / hopSamples));
  const winRmsDb = new Float32Array(numSteps);

  for (let w = 0; w < numSteps; w++) {
    const center = w * hopSamples;
    const start = Math.max(0, center - (winSamples >> 1));
    const end = Math.min(length, center + (winSamples >> 1));
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

  // Anchor speech detection strictly to active speech level so breaths/noise are ignored
  const speechRefDb = measureActiveSpeechRmsDb(channels, sampleRate);
  const speechGateDb = Math.max(-42, speechRefDb - 14);

  const activeLevels = [];
  for (let w = 0; w < numSteps; w++) {
    if (winRmsDb[w] >= speechGateDb) activeLevels.push(winRmsDb[w]);
  }

  if (activeLevels.length === 0) {
    return { channels: channels.map((ch) => Float32Array.from(ch)), targetSpeechDb: -20 };
  }

  activeLevels.sort((a, b) => a - b);
  const targetSpeechDb = activeLevels[Math.floor(activeLevels.length * 0.75)];

  // Cap gain range tightly (±6.5 dB max) so leveling stays completely transparent
  const maxBoostDb = 6.5 * strength;
  const maxCutDb = -6.0 * strength;
  const rawGainDb = new Float32Array(numSteps);

  let lastSpeechGainDb = 0;
  let silentStreak = 0;

  for (let w = 0; w < numSteps; w++) {
    if (winRmsDb[w] >= speechGateDb) {
      const diff = (targetSpeechDb - winRmsDb[w]) * strength;
      const g = Math.max(maxCutDb, Math.min(maxBoostDb, diff));
      rawGainDb[w] = g;
      lastSpeechGainDb = g;
      silentStreak = 0;
    } else {
      // During brief gaps (<0.6s) hold last gain so syllables never dip;
      // during longer pauses, gently relax toward 0 dB (never boost silence!)
      silentStreak++;
      const decay = silentStreak <= 3 ? 0.92 : 0.72;
      lastSpeechGainDb = Math.min(lastSpeechGainDb, 0) === lastSpeechGainDb
        ? lastSpeechGainDb * decay
        : Math.max(0, lastSpeechGainDb * decay);
      rawGainDb[w] = Math.min(0, lastSpeechGainDb); // Never boost below speech gate
    }
  }

  // Zero-phase Gaussian-like smoothing across 7 steps (~1.4 s) removes all pumping
  const smoothGainDb = new Float32Array(numSteps);
  const weights = [0.05, 0.12, 0.2, 0.26, 0.2, 0.12, 0.05];
  for (let w = 0; w < numSteps; w++) {
    let acc = 0;
    for (let k = -3; k <= 3; k++) {
      const idx = Math.max(0, Math.min(numSteps - 1, w + k));
      acc += rawGainDb[idx] * weights[k + 3];
    }
    smoothGainDb[w] = acc;
  }

  const out = channels.map((ch) => new Float32Array(ch.length));
  const slewCoeff = Math.exp(-1 / (0.28 * sampleRate)); // 280 ms silky-smooth fader ride
  let currentGainDb = smoothGainDb[0];

  for (let i = 0; i < length; i++) {
    const pos = i / hopSamples;
    const w0 = Math.min(numSteps - 1, Math.floor(pos));
    const w1 = Math.min(numSteps - 1, w0 + 1);
    const frac = pos - w0;
    const desiredDb = smoothGainDb[w0] * (1 - frac) + smoothGainDb[w1] * frac;

    currentGainDb = slewCoeff * currentGainDb + (1 - slewCoeff) * desiredDb;
    const gain = dbToLinear(currentGainDb);

    for (let ch = 0; ch < numChannels; ch++) {
      out[ch][i] = channels[ch][i] * gain;
    }
  }

  return { channels: out, targetSpeechDb };
}

/**
 * Stage c: Soft-Knee Broadcast Dynamics Compressor
 * Uses a hybrid fast-RMS + peak detector (6 dB soft knee) so loud transients
 * and strong vowels are smoothed transparently with consistent radio presence.
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
  const rmsCoeff = Math.exp(-1 / Math.max(1, 0.008 * sampleRate)); // 8 ms RMS smoothing

  let envSq = 0;
  let smoothedGrDb = 0;
  let maxGrDb = 0;
  const out = channels.map((ch) => new Float32Array(ch.length));

  for (let i = 0; i < length; i++) {
    let peak = 0;
    let sq = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const s = channels[ch][i];
      const a = Math.abs(s);
      if (a > peak) peak = a;
      sq += s * s;
    }
    sq /= numChannels;
    envSq = rmsCoeff * envSq + (1 - rmsCoeff) * sq;
    // Blend 65% fast-RMS + 35% peak for musical vocal compression without harsh transient clicks
    const detector = 0.65 * Math.sqrt(envSq) + 0.35 * peak;
    const inputDb = linearToDb(detector);

    let targetGrDb = 0;
    const overshoot = inputDb - thresholdDb;
    if (overshoot > halfKnee) {
      targetGrDb = overshoot * (1 - 1 / ratio);
    } else if (overshoot > -halfKnee) {
      const x = overshoot + halfKnee;
      targetGrDb = ((1 - 1 / ratio) * x * x) / (2 * kneeDb);
    }

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
 * Stage e: 2-Pass Loudness Normalization + 4x Oversampled Lookahead True-Peak Limiter
 */
function runNormalizeAndLimit(
  channels,
  sampleRate,
  { mode = 'lufs', targetLoudness = -16, ceilingDbtp = -1.0 }
) {
  const numChannels = channels.length;
  const length = channels[0].length;

  let currentLevel;
  if (mode === 'rms') {
    let sumSq = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const d = channels[ch];
      for (let i = 0; i < length; i++) sumSq += d[i] * d[i];
    }
    currentLevel = linearToDb(Math.sqrt(sumSq / Math.max(1, length * numChannels)));
  } else {
    currentLevel = measureBS1770Loudness(channels, sampleRate).integratedLufs;
  }

  const makeupDb =
    currentLevel <= -68 ? 0 : Math.max(-24, Math.min(26, targetLoudness - currentLevel));
  const makeupLinear = dbToLinear(makeupDb);

  const normalized = channels.map((ch) => {
    const copy = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      copy[i] = ch[i] * makeupLinear;
    }
    return copy;
  });

  const ceilingLinear = dbToLinear(ceilingDbtp);
  const lookaheadSamples = Math.max(1, Math.round(0.005 * sampleRate)); // 5 ms
  const releaseCoeff = Math.exp(-1 / (0.065 * sampleRate)); // 65 ms release
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

  // 2nd-pass target lock if heavy peak limiting pulled integrated LUFS below target
  if (currentLevel > -68 && mode === 'lufs') {
    const postLufs = measureBS1770Loudness(out, sampleRate).integratedLufs;
    const shortfall = targetLoudness - postLufs;
    if (shortfall > 0.3 && shortfall < 8) {
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
 * Stage f: Optional Silence Trimming with 25 ms Raised-Cosine Crossfades
 */
function runSilenceTrimming(channels, sampleRate, maxPauseSec = 1.0) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const frameSamples = Math.max(1, Math.round(0.025 * sampleRate));
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
    isSilent[f] =
      linearToDb(Math.sqrt(sumSq / Math.max(1, count))) < silenceThreshDb ? 1 : 0;
  }

  const maxPauseFrames = Math.max(4, Math.round(maxPauseSec / 0.025));
  const keepHalfFrames = Math.max(
    2,
    Math.round(Math.min(0.4, maxPauseSec * 0.45) / 0.025 / 2)
  );

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

  if (segments.length <= 1 && trimmedSeconds === 0) {
    return { channels, trimmedSeconds: 0 };
  }

  const fadeSamples = Math.round(0.025 * sampleRate);
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

    for (let i = overlap; i < segLen; i++) {
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

  // Stage a: 4th-order High-pass filter (24 dB/oct)
  if (options.highPassEnabled) {
    await notify(12, `Stage A: 24 dB/oct High-pass filter (${options.highPassFreq} Hz)...`);
    channels = runHighPassFilter(channels, sampleRate, options.highPassFreq);
  }

  // Stage b: Noise reduction (Spectral Wiener STFT or Speech-Relative Gate/Expander)
  if (options.noiseGateEnabled) {
    const useSpectral = (options.noiseMode || 'spectral') === 'spectral';
    await notify(
      26,
      useSpectral
        ? `Stage B: STFT Spectral Wiener denoising (-${options.noiseGateReductionDb} dB)...`
        : `Stage B: Speech-relative noise expander (-${options.noiseGateReductionDb} dB)...`
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
  } else if (options.postNeuralCleanup) {
    // Gentle speech-relative expander after DeepFilterNet3 suppresses residual distant TV/breaths
    await notify(26, 'Cleaning residual pause chatter & breaths...');
    const res = runNoiseGate(channels, sampleRate, 14, 5);
    channels = res.channels;
    report.estimatedNoiseFloorDb = res.noiseFloorDb;
    report.gateThresholdDb = res.thresholdDb;
  }

  // Stage +: Adaptive Studio Voice Clarity EQ & Split-Band De-Esser
  if (options.voiceClarityEnabled) {
    await notify(
      40,
      `Adaptive Voice Clarity EQ & De-Esser (${Math.round((options.voiceClarityAmount ?? 0.75) * 100)}%)...`
    );
    channels = runVoiceClarityEqAndDeEsser(
      channels,
      sampleRate,
      options.voiceClarityAmount ?? 0.75
    );
  }

  // Stage d: Smooth Phrase Auto-Leveler (runs BEFORE compressor for zero pumping!)
  if (options.levelerEnabled) {
    await notify(54, 'Smooth phrase auto-leveler (holding gain cleanly across pauses)...');
    const res = runAutoLeveler(channels, sampleRate, options.levelerStrength ?? 0.75);
    channels = res.channels;
    report.levelerTargetDb = res.targetSpeechDb;
  }

  // Stage c: Soft-Knee Broadcast Compressor
  if (options.compressorEnabled) {
    await notify(
      68,
      `Dynamics compression (${options.compThresholdDb} dB, ${options.compRatio}:1)...`
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

  // Stage e: 2-Pass Loudness Normalization + True-Peak Limiter
  if (options.normalizeEnabled) {
    const unit = options.normalizeMode === 'rms' ? 'dBFS RMS' : 'LUFS';
    await notify(
      82,
      `Normalizing to ${options.targetLoudness} ${unit} + 4x True-peak limiter...`
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

  // Stage f: Optional Silence Trimming
  if (options.silenceTrimEnabled) {
    await notify(
      92,
      `Trimming pauses longer than ${options.silenceMaxPauseSec.toFixed(1)} s...`
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

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataByteLength, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
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
      const int16 =
        sample < 0 ? Math.round(sample * 0x8000) : Math.round(sample * 0x7fff);
      view.setInt16(offset, int16, true);
      offset += 2;
    }
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Generates an 8.5-second realistic synthetic speech-like test recording.
 */
export function generateDemoAudioBuffer(audioContext) {
  const sampleRate = audioContext.sampleRate || 44100;
  const durationSec = 8.5;
  const length = Math.round(durationSec * sampleRate);
  const buffer = audioContext.createBuffer(2, length, sampleRate);
  const left = buffer.getChannelData(0);
  const right = buffer.getChannelData(1);

  let seed = 42197;
  const nextRand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 0xffffffff - 0.5;
  };

  const phrases = [
    { start: 0.4, end: 2.3, amp: 0.14, baseFreq: 132 },
    { start: 2.6, end: 4.3, amp: 0.62, baseFreq: 158 },
    { start: 6.5, end: 8.1, amp: 0.24, baseFreq: 142 },
  ];

  let hissFilter = 0;

  for (let i = 0; i < length; i++) {
    const t = i / sampleRate;
    const rumble =
      0.045 * Math.sin(2 * Math.PI * 48 * t) +
      0.025 * Math.sin(2 * Math.PI * 62 * t + 0.7);

    const white = nextRand();
    hissFilter = 0.65 * hissFilter + 0.35 * white;
    const hiss = (white * 0.55 + hissFilter * 0.45) * 0.032;

    let voice = 0;
    for (let p = 0; p < phrases.length; p++) {
      const ph = phrases[p];
      if (t >= ph.start && t <= ph.end) {
        const localT = t - ph.start;
        const phraseLen = ph.end - ph.start;
        const edgeEnv =
          Math.min(1, localT / 0.06) * Math.min(1, (phraseLen - localT) / 0.08);
        const syllable = Math.max(0, Math.sin(2 * Math.PI * 4.2 * localT)) ** 0.7;
        const pitch =
          ph.baseFreq * (1 + 0.08 * Math.sin(2 * Math.PI * 1.4 * localT));
        const harmonics =
          0.55 * Math.sin(2 * Math.PI * pitch * t) +
          0.32 * Math.sin(2 * Math.PI * pitch * 2.0 * t + 0.4) +
          0.22 * Math.sin(2 * Math.PI * pitch * 3.0 * t + 1.1) +
          0.16 * Math.sin(2 * Math.PI * pitch * 5.0 * t + 0.2) +
          0.09 * Math.sin(2 * Math.PI * pitch * 8.0 * t);

        const spike =
          p === 1 && Math.abs(t - 3.35) < 0.03
            ? 0.55 *
              Math.cos(((t - 3.35) / 0.03) * (Math.PI / 2)) *
              Math.sin(2 * Math.PI * 380 * t)
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
