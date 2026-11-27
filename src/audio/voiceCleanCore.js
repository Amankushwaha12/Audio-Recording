/**
 * voiceCleanCore.js — Pure, Web-Worker-Safe Voice Clean DSP Engine
 *
 * Plain-language overview of every stage (designed for clarity and learning):
 *   1. Pre-clean High-Pass (75 Hz, gentle 12 dB/oct Butterworth):
 *      Removes sub-bass rumble, AC hum, and mic handling bumps without
 *      thinning out the fundamental warmth of human speech.
 *   2. Spectral Noise Reduction (2048-pt STFT, 75% overlap, Hann window):
 *      Estimates background noise automatically from the quietest ~10% of
 *      non-zero frames (or from a user-selected noise region on the waveform).
 *      Computes a stereo-linked Wiener / decision-directed gain with both
 *      temporal and frequency smoothing to prevent watery "musical noise",
 *      with a strict gain floor (max attenuation ~6–18 dB) so bins are never
 *      completely zeroed. (Optionally replaced by Stage 2 DeepFilterNet3.)
 *   3. Late-Reverberation Suppression (2048-pt STFT exponential decay model):
 *      Estimates the room's decay rate (approximate RT60) from post-transient
 *      energy slopes. If the audio is already dry (RT60 < 0.22 s), skips
 *      automatically and reports it. Otherwise predicts late-reverb tail power
 *      after the early-reflection window (~40 ms) and gently subtracts it with
 *      a safe gain floor.
 *   4. Natural-Voice Blend ("Keep natural voice", default 15%):
 *      Mixes the time-aligned pre-cleaned dry voice back in so micro-transients
 *      and natural breath stay intact, preventing any robotic character.
 *   5. Gentle Dynamic De-Esser (5–9 kHz bandpass sidechain):
 *      Dynamically softens sharp "s / sh / t" spikes only when they exceed
 *      threshold, leaving normal vocal clarity untouched.
 *   6. Gentle Tone Shaping (strictly within ±3 dB):
 *      -2.0 dB cut at 250 Hz (reduces boxy room mud) and +1.5 dB lift at
 *      3.5 kHz (adds natural vocal presence).
 *   7. Loudness Match (default ON) + 4x Lookahead True-Peak Limiter (-1.5 dBTP):
 *      Measures ITU-R BS.1770-4 integrated LUFS before and after cleaning and
 *      matches output loudness within ±0.5 LU of input (or skips matching when
 *      an external target LUFS preset will run afterward), then limits true
 *      peaks cleanly at -1.5 dBTP so the result never clips.
 *
 * Quality guarantees:
 *   - Never mutates input channel buffers; always processes fresh copies.
 *   - Stereo-linked spectral gains prevent inter-channel phase shifts.
 *   - Chunked processing with 50 ms raised-cosine crossfades keeps memory low
 *     on long recordings while preserving exact duration and sample rate.
 */

import {
  applyBiquad,
  dbToLinear,
  estimateTruePeak,
  linearToDb,
  measureBS1770Loudness,
} from './analyzer.js';
import { fftRadix2, ifftRadix2 } from './fft.js';
import {
  VOICE_TARGET,
  detectAndRemoveMainsHum,
  evaluateVoiceTargetDenoise,
} from './voiceTarget.js';

/**
 * One-click Voice Clean presets (all sliders remain freely overridable).
 */
export const VOICE_CLEAN_PRESETS = {
  natural: {
    id: 'natural',
    label: 'Natural',
    description: 'Light cleanup that preserves maximum room feel and voice texture',
    noiseReduction: 40,
    reverbReduction: 20,
    naturalBlend: 25,
    adaptiveDenoise: false,
    humDetect: false,
    keepOriginalLoudness: true,
  },
  clear: {
    id: 'clear',
    label: 'Clear (Default)',
    description: 'Balanced speech clarity, room echo control, and natural presence',
    noiseReduction: 60,
    reverbReduction: 40,
    naturalBlend: 15,
    adaptiveDenoise: false,
    humDetect: false,
    keepOriginalLoudness: true,
  },
  strong: {
    id: 'strong',
    label: 'Strong',
    description: 'Stronger noise and echo reduction for challenging untreated rooms',
    noiseReduction: 80,
    reverbReduction: 65,
    naturalBlend: 10,
    adaptiveDenoise: false,
    humDetect: false,
    keepOriginalLoudness: true,
  },
  voiceTarget: {
    id: 'voiceTarget',
    label: 'Voice Target (-16 LUFS)',
    description:
      'SNR-adaptive denoise, 50/60 Hz hum notch (4 harmonics), ±3 dB Auto-EQ, gentle ±6 dB leveler, 15% dry mix, -16 LUFS (±0.5 LU), -1.0 dBTP, LRA 4–8 LU',
    noiseReduction: 60,
    reverbReduction: 0,
    naturalBlend: Math.round(VOICE_TARGET.dryMix * 100),
    adaptiveDenoise: true,
    humDetect: true,
    keepOriginalLoudness: false,
  },
  voiceIsolation: {
    id: 'voiceIsolation',
    label: 'Voice Isolation',
    description:
      'Isolates foreground speaker from stereo music, chatter & room bleed using center-voice extraction, harmonic speech masking, 85 Hz–11.5 kHz vocal focus & -16 LUFS polish',
    noiseReduction: 80,
    reverbReduction: 35,
    naturalBlend: 8,
    adaptiveDenoise: false,
    humDetect: true,
    isVoiceIsolation: true,
    keepOriginalLoudness: false,
  },
};

export const DEFAULT_VOICE_CLEAN_OPTIONS = {
  enabled: true,
  preset: 'clear',
  noiseReduction: 60, // 0–100% mapped to ~6–18 dB max attenuation
  reverbReduction: 40, // 0–100% late-reverb suppression
  naturalBlend: 15, // 0–100% dry voice mixed back in
  keepOriginalLoudness: true, // Match output LUFS to input within 0.5 LU
  useAiDenoise: false, // Stage 2: optional DeepFilterNet AI denoise
  adaptiveDenoise: false, // Voice Target: SNR-adaptive denoise strength
  humDetect: false, // Voice Target: 50/60 Hz mains hum + 4 harmonics notch
  noiseSampleRegion: null, // Optional { startSec, endSec }
};

const STFT_SIZE = 2048;
const STFT_HOP = 512; // 75% overlap (2048 / 4)
const HALF_BINS = (STFT_SIZE >> 1) + 1;
const CHUNK_DURATION_SEC = 16; // Chunk long recordings to keep memory low
const CHUNK_OVERLAP_SEC = 0.08; // 80 ms raised-cosine crossfade between chunks
const DRY_RT60_THRESHOLD_SEC = 0.22; // Auto-skip reverb reduction below this RT60
const SILENT_PEAK_THRESHOLD_DB = -72; // Treat files below this peak as silent

/**
 * Precomputes a periodic Hann window and exact COLA normalization factor
 * for 2048-pt STFT at 75% overlap (hop = 512).
 * With analysis Hann + synthesis Hann at 75% overlap, sum(w^2) === 1.5 exactly!
 */
function buildHannWindow(size) {
  const win = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / size));
  }
  return win;
}

const HANN_2048 = buildHannWindow(STFT_SIZE);
// For Hann^2 summed every N/4 samples, constant overlap-add gain is exactly 1.5
const HANN_75_NORM = 1 / 1.5;

function makeHighPass2ndOrder(sampleRate, freqHz) {
  const f0 = Math.max(20, Math.min(freqHz, sampleRate * 0.4));
  const w0 = (2 * Math.PI * f0) / sampleRate;
  const cosW0 = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * Math.SQRT1_2); // Q = 0.707 (gentle 12 dB/oct Butterworth)
  const a0 = 1 + alpha;
  return {
    b0: ((1 + cosW0) / 2) / a0,
    b1: -(1 + cosW0) / a0,
    b2: ((1 + cosW0) / 2) / a0,
    a1: (-2 * cosW0) / a0,
    a2: (1 - alpha) / a0,
  };
}

function makePeakingBiquad(sampleRate, freqHz, gainDb, q) {
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

function makeBandPassBiquad(sampleRate, freqHz, q) {
  const f0 = Math.max(80, Math.min(freqHz, sampleRate * 0.45));
  const w0 = (2 * Math.PI * f0) / sampleRate;
  const cosW0 = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: alpha / a0,
    b1: 0,
    b2: -alpha / a0,
    a1: (-2 * cosW0) / a0,
    a2: (1 - alpha) / a0,
  };
}

/**
 * Estimates peak, active speech RMS, and room noise floor (dBFS) across all channels.
 */
function inspectSignalLevels(channels, sampleRate) {
  const numChannels = channels.length;
  const length = channels[0].length;

  let peak = 0;
  for (let ch = 0; ch < numChannels; ch++) {
    const d = channels[ch];
    for (let i = 0; i < length; i++) {
      const a = Math.abs(d[i]);
      if (a > peak) peak = a;
    }
  }
  const peakDb = linearToDb(peak);

  const frameSamples = Math.max(1, Math.round(0.05 * sampleRate));
  const numFrames = Math.max(1, Math.ceil(length / frameSamples));
  const nonZeroFrameDbs = [];
  const speechFrameDbs = [];

  for (let f = 0; f < numFrames; f++) {
    const start = f * frameSamples;
    const end = Math.min(length, start + frameSamples);
    let sumSq = 0;
    let cnt = 0;
    for (let i = start; i < end; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        const s = channels[ch][i];
        sumSq += s * s;
        cnt++;
      }
    }
    const db = linearToDb(Math.sqrt(sumSq / Math.max(1, cnt)));
    if (db > -85) nonZeroFrameDbs.push(db);
    if (db > -45) speechFrameDbs.push(db);
  }

  nonZeroFrameDbs.sort((a, b) => a - b);
  speechFrameDbs.sort((a, b) => a - b);

  const noiseFloorDb =
    nonZeroFrameDbs.length > 0
      ? nonZeroFrameDbs[Math.floor(nonZeroFrameDbs.length * 0.1)]
      : -100;
  const speechRmsDb =
    speechFrameDbs.length > 0
      ? speechFrameDbs[Math.floor(speechFrameDbs.length * 0.8)]
      : nonZeroFrameDbs.length > 0
        ? nonZeroFrameDbs[Math.floor(nonZeroFrameDbs.length * 0.85)]
        : -100;

  return {
    peak,
    peakDb,
    noiseFloorDb,
    speechRmsDb,
    snrDb: speechRmsDb - noiseFloorDb,
  };
}

/**
 * Estimates room reverberation decay time (approximate RT60 in seconds) by
 * locating sharp speech offsets (energy drops after vocal peaks) and measuring
 * the downward dB slope in the 300–4000 Hz vocal band.
 *
 * Returns `{ rt60Sec, isDry, confidence }`.
 */
export function estimateRoomReverb(channels, sampleRate) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const frameSamples = Math.max(1, Math.round(0.02 * sampleRate)); // 20 ms frames
  const numFrames = Math.floor(length / frameSamples);

  if (numFrames < 20) {
    return { rt60Sec: 0.15, isDry: true, confidence: 'low' };
  }

  // Bandpass 300–3600 Hz (core vocal reverberation region) on mono sum
  const mono = new Float32Array(length);
  const invCh = 1 / numChannels;
  for (let ch = 0; ch < numChannels; ch++) {
    const d = channels[ch];
    for (let i = 0; i < length; i++) mono[i] += d[i] * invCh;
  }
  const vocalBand = applyBiquad(mono, makeBandPassBiquad(sampleRate, 1100, 0.7));

  const envDb = new Float32Array(numFrames);
  const allDbs = [];
  const activeDbs = [];
  for (let f = 0; f < numFrames; f++) {
    const start = f * frameSamples;
    let sq = 0;
    for (let i = 0; i < frameSamples; i++) {
      const s = vocalBand[start + i];
      sq += s * s;
    }
    const db = Math.max(-75, linearToDb(Math.sqrt(sq / frameSamples)));
    envDb[f] = db;
    allDbs.push(db);
    if (db > -65) activeDbs.push(db);
  }

  if (activeDbs.length < 10) {
    return { rt60Sec: 0.15, isDry: true, confidence: 'low' };
  }

  allDbs.sort((a, b) => a - b);
  activeDbs.sort((a, b) => a - b);
  const floorDb = allDbs[Math.floor(allDbs.length * 0.1)];
  const peakSpeechDb = activeDbs[Math.floor(activeDbs.length * 0.9)];
  if (peakSpeechDb - floorDb < 14) {
    return { rt60Sec: 0.18, isDry: true, confidence: 'low' };
  }

  // Find genuine phrase/word-ending decays into quiet pauses.
  // Measure slope strictly across the lower decay tail (below peakSpeechDb - 10 dB)
  // so slow mid-speech vowel inflections never masquerade as acoustic room reverb.
  const decayEstimates = [];
  const frameDurSec = frameSamples / sampleRate; // 0.02 s
  const stopFloor = floorDb + 6;
  const tailStartThresh = peakSpeechDb - 10;

  for (let f = 2; f < numFrames - 6; f++) {
    if (
      envDb[f] >= peakSpeechDb - 16 &&
      envDb[f] >= envDb[f - 1] - 0.5 &&
      envDb[f] > envDb[f + 1] + 1.5
    ) {
      let endF = f + 1;
      while (
        endF < Math.min(numFrames - 1, f + 24) &&
        envDb[endF] > stopFloor &&
        envDb[endF] <= envDb[endF - 1] + 1.2
      ) {
        endF++;
      }

      const hitFloor = endF < numFrames && envDb[endF] <= stopFloor;
      const tailEnd = hitFloor ? endF : Math.max(f + 1, endF - 1);
      const totalDropDb = envDb[f] - envDb[tailEnd];

      // Must drop at least 16 dB and remain quiet for >= 60 ms (3 frames)
      if (totalDropDb < 16) continue;
      const checkEnd = Math.min(numFrames, tailEnd + 3);
      let staysLow = true;
      for (let k = tailEnd; k < checkEnd; k++) {
        if (envDb[k] > envDb[f] - 12) {
          staysLow = false;
          break;
        }
      }
      if (!staysLow) continue;

      // Start measuring where the true post-speech tail drops below tailStartThresh
      let startF = f;
      while (startF < tailEnd - 1 && envDb[startF + 1] > tailStartThresh) {
        startF++;
      }

      const spanFrames = tailEnd - startF;
      const tailDropDb = envDb[startF] - envDb[tailEnd];
      if (tailDropDb < 12) {
        f = tailEnd;
        continue;
      }

      if (spanFrames <= 2) {
        // Sharp dry cutoff (>= 12 dB drop in <= 40 ms -> dry close-mic/studio speech)
        const slopeDbPerSec = -tailDropDb / Math.max(frameDurSec, spanFrames * frameDurSec);
        const rt60 = Math.max(0.06, Math.min(0.2, -60 / slopeDbPerSec));
        decayEstimates.push(rt60);
      } else {
        const count = spanFrames + 1;
        let sumX = 0;
        let sumY = 0;
        let sumXX = 0;
        let sumXY = 0;
        for (let k = 0; k < count; k++) {
          const x = k * frameDurSec;
          const y = Math.max(floorDb, envDb[startF + k]);
          sumX += x;
          sumY += y;
          sumXX += x * x;
          sumXY += x * y;
        }
        const denom = count * sumXX - sumX * sumX;
        if (denom > 1e-8) {
          const slopeDbPerSec = (count * sumXY - sumX * sumY) / denom;
          if (slopeDbPerSec < -15) {
            const rt60 = Math.max(0.08, Math.min(2.5, -60 / slopeDbPerSec));
            decayEstimates.push(rt60);
          }
        }
      }
      f = tailEnd;
    }
  }

  if (decayEstimates.length === 0) {
    return { rt60Sec: 0.16, isDry: true, confidence: 'low' };
  }

  decayEstimates.sort((a, b) => a - b);
  const rt60Sec = decayEstimates[Math.floor(decayEstimates.length * 0.35)];
  return {
    rt60Sec: Number(rt60Sec.toFixed(2)),
    isDry: rt60Sec < DRY_RT60_THRESHOLD_SEC,
    confidence: decayEstimates.length >= 3 ? 'high' : 'medium',
  };
}

/**
 * Estimates the per-bin noise Power Spectral Density (PSD) across 2048-pt Hann frames.
 * Uses either a user-selected waveform noise region `{ startSec, endSec }` OR
 * automatically locates the quietest ~10% of non-zero frames.
 */
function estimateGlobalNoisePsd(channels, sampleRate, noiseSampleRegion) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const real = new Float32Array(STFT_SIZE);
  const imag = new Float32Array(STFT_SIZE);
  const noisePsd = new Float32Array(HALF_BINS);

  // Case 1: User explicitly selected a noise sample region on the waveform
  if (
    noiseSampleRegion &&
    Number.isFinite(noiseSampleRegion.startSec) &&
    Number.isFinite(noiseSampleRegion.endSec) &&
    noiseSampleRegion.endSec > noiseSampleRegion.startSec + 0.04
  ) {
    const startSample = Math.max(
      0,
      Math.min(length - 1, Math.floor(noiseSampleRegion.startSec * sampleRate))
    );
    const endSample = Math.max(
      startSample + 1,
      Math.min(length, Math.floor(noiseSampleRegion.endSec * sampleRate))
    );
    const regionLen = endSample - startSample;
    const regionFrames = Math.max(1, Math.floor((regionLen - STFT_SIZE) / STFT_HOP) + 1);
    let totalFrames = 0;
    let sumSq = 0;

    for (let f = 0; f < regionFrames; f++) {
      const base = startSample + f * STFT_HOP;
      for (let ch = 0; ch < numChannels; ch++) {
        const src = channels[ch];
        for (let i = 0; i < STFT_SIZE; i++) {
          const idx = base + i;
          const v = idx < endSample ? src[idx] : 0;
          real[i] = v * HANN_2048[i];
          imag[i] = 0;
          sumSq += v * v;
        }
        fftRadix2(real, imag);
        for (let b = 0; b < HALF_BINS; b++) {
          noisePsd[b] += real[b] * real[b] + imag[b] * imag[b];
        }
        totalFrames++;
      }
    }

    if (totalFrames > 0) {
      for (let b = 0; b < HALF_BINS; b++) {
        noisePsd[b] /= totalFrames;
      }
      const floorDb = linearToDb(Math.sqrt(sumSq / Math.max(1, totalFrames * STFT_SIZE)));
      return { noisePsd, noiseFloorDb: floorDb, source: 'selected-region' };
    }
  }

  // Case 2: Automatic noise profile from the quietest ~10% of frames
  const totalFrames = Math.max(1, Math.floor((length - STFT_SIZE) / STFT_HOP) + 1);
  // Stride very long files when scanning for the quietest 10% to stay fast & low-memory
  const scanStride = Math.max(1, Math.floor(totalFrames / 1200));
  const candidates = [];

  for (let f = 0; f < totalFrames; f += scanStride) {
    const start = f * STFT_HOP;
    let sq = 0;
    let cnt = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const src = channels[ch];
      for (let i = 0; i < STFT_SIZE; i += 4) {
        const idx = start + i;
        if (idx < length) {
          const s = src[idx];
          sq += s * s;
          cnt++;
        }
      }
    }
    const meanSq = cnt > 0 ? sq / cnt : 0;
    candidates.push({ f, meanSq });
  }

  // If >= 10% of frames are clean digital silence (< -75 dBFS), keep them in the pool
  // so studio/gated clean files report a true ultra-low noise floor instead of mistaking
  // quiet speech syllables for background noise.
  const quietDigitalSq = Math.pow(10, -75 / 10);
  const digitalSilentCount = candidates.filter((c) => c.meanSq <= quietDigitalSq).length;
  const minRoomSq = Math.pow(10, -85 / 10);
  const nonZero = candidates.filter((c) => c.meanSq > minRoomSq);
  const pool =
    digitalSilentCount >= Math.max(3, Math.ceil(candidates.length * 0.1))
      ? candidates
      : nonZero.length >= 4
        ? nonZero
        : candidates;
  pool.sort((a, b) => a.meanSq - b.meanSq);

  const quietCount = Math.max(2, Math.min(pool.length, Math.ceil(pool.length * 0.1)));
  let avgSq = 0;
  let accumCount = 0;

  for (let q = 0; q < quietCount; q++) {
    const start = pool[q].f * STFT_HOP;
    avgSq += pool[q].meanSq;
    for (let ch = 0; ch < numChannels; ch++) {
      const src = channels[ch];
      for (let i = 0; i < STFT_SIZE; i++) {
        const idx = start + i;
        real[i] = idx < length ? src[idx] * HANN_2048[i] : 0;
        imag[i] = 0;
      }
      fftRadix2(real, imag);
      for (let b = 0; b < HALF_BINS; b++) {
        noisePsd[b] += real[b] * real[b] + imag[b] * imag[b];
      }
      accumCount++;
    }
  }

  for (let b = 0; b < HALF_BINS; b++) {
    noisePsd[b] /= Math.max(1, accumCount);
  }

  const noiseFloorDb = linearToDb(Math.sqrt(avgSq / Math.max(1, quietCount)));
  return { noisePsd, noiseFloorDb, source: 'auto' };
}

/**
 * Processes a single audio chunk through the stereo-linked 2048-pt 75%-overlap STFT
 * engine for:
 *   - Stage 2: Wiener spectral noise reduction (when `runNoise` is true)
 *   - Stage 3: Exponential-decay late-reverb suppression (when `runReverb` is true)
 *
 * Computing ONE shared spectral gain curve across stereo channels (from L+R power)
 * guarantees zero inter-channel phase shift or stereo image wobble.
 */
function processStftChunk(
  chunkChannels,
  sampleRate,
  {
    runNoise,
    noisePsd,
    noiseReductionPct,
    runReverb,
    reverbReductionPct,
    estimatedRt60Sec,
  }
) {
  const numChannels = chunkChannels.length;
  const chunkLen = chunkChannels[0].length;
  const pad = STFT_SIZE - STFT_HOP; // Center-pad edges so start/end frames reconstruct cleanly
  const paddedLen = chunkLen + 2 * pad;
  const numFrames = Math.max(1, Math.ceil((paddedLen - STFT_SIZE) / STFT_HOP) + 1);

  // Noise reduction mapping: 0–100% -> 0 dB or 6–18 dB max attenuation (strict gain floor)
  const noiseMaxAttenDb =
    runNoise && noiseReductionPct > 0
      ? 6 + (Math.min(100, noiseReductionPct) / 100) * 12
      : 0;
  const noiseGainFloor = dbToLinear(-noiseMaxAttenDb);
  const noiseOverSub = 1.05 + (Math.min(100, noiseReductionPct) / 100) * 0.7;
  const alphaDD = 0.94; // Decision-directed smoothing eliminates musical noise

  // Reverb suppression mapping: 0–100% -> exponential late-reverb subtraction (max -12 dB floor)
  const reverbStrength = runReverb ? Math.min(1, Math.max(0, reverbReductionPct / 100)) : 0;
  const reverbGainFloor = dbToLinear(-(2 + reverbStrength * 10)); // Never zero a bin
  const reverbBeta = 0.35 + reverbStrength * 0.75; // Late-reverb subtraction weight
  const hopSec = STFT_HOP / sampleRate;
  const rt60 = Math.max(0.22, Math.min(1.8, estimatedRt60Sec || 0.45));
  // Exponential power decay per STFT hop: exp(-6 * ln(10) * hopSec / RT60)
  const baseDecay = Math.exp((-13.8155 * hopSec) / rt60);
  // Wait ~40 ms (early reflection delay) before feeding speech energy into late-reverb model
  const earlyDelayFrames = Math.max(2, Math.round(0.04 / hopSec));
  const ringSize = earlyDelayFrames + 1;
  const powerRing = Array.from({ length: ringSize }, () => new Float32Array(HALF_BINS));
  let ringHead = 0;

  // Per-channel FFT work buffers and padded output accumulators
  const reals = Array.from({ length: numChannels }, () => new Float32Array(STFT_SIZE));
  const imags = Array.from({ length: numChannels }, () => new Float32Array(STFT_SIZE));
  const paddedOuts = Array.from({ length: numChannels }, () => new Float32Array(paddedLen));

  // Shared stereo-linked spectral state (prevents inter-channel phase shift!)
  const framePower = new Float32Array(HALF_BINS);
  const prevPostSnr = new Float32Array(HALF_BINS).fill(1);
  const prevNoiseGain = new Float32Array(HALF_BINS).fill(1);
  const lateReverbPsd = new Float32Array(HALF_BINS);
  const prevTotalGain = new Float32Array(HALF_BINS).fill(1);
  const rawGain = new Float32Array(HALF_BINS);
  const smoothGain = new Float32Array(HALF_BINS);

  // Per-bin decay curve (high frequencies decay slightly faster in real rooms)
  const binDecay = new Float32Array(HALF_BINS);
  for (let b = 0; b < HALF_BINS; b++) {
    const freqHz = (b * sampleRate) / STFT_SIZE;
    const hfScale = freqHz > 3000 ? 0.88 : 1.0;
    binDecay[b] = Math.pow(baseDecay, 1 / hfScale);
  }

  for (let f = 0; f < numFrames; f++) {
    const frameStart = f * STFT_HOP;

    // 1. Window & FFT all channels
    for (let ch = 0; ch < numChannels; ch++) {
      const src = chunkChannels[ch];
      const re = reals[ch];
      const im = imags[ch];
      for (let i = 0; i < STFT_SIZE; i++) {
        const srcIdx = frameStart + i - pad;
        const sample = srcIdx >= 0 && srcIdx < chunkLen ? src[srcIdx] : 0;
        re[i] = sample * HANN_2048[i];
        im[i] = 0;
      }
      fftRadix2(re, im);
    }

    // 2. Compute stereo-linked power per bin (average across channels)
    const invCh = 1 / numChannels;
    for (let b = 0; b < HALF_BINS; b++) {
      let p = 0;
      for (let ch = 0; ch < numChannels; ch++) {
        const r = reals[ch][b];
        const m = imags[ch][b];
        p += r * r + m * m;
      }
      framePower[b] = p * invCh;
    }

    // 3. Spectral Noise Reduction (Decision-Directed Wiener) + Late-Reverb Subtraction
    const delayedPower = powerRing[(ringHead + 1) % ringSize];

    for (let b = 0; b < HALF_BINS; b++) {
      const power = framePower[b];
      let gNoise = 1.0;

      if (runNoise && noiseMaxAttenDb > 0) {
        const nPow = Math.max(noisePsd[b] * noiseOverSub, 1e-12);
        const postSnr = power / nPow;
        const priorSnr =
          alphaDD * (prevNoiseGain[b] * prevNoiseGain[b] * prevPostSnr[b]) +
          (1 - alphaDD) * Math.max(postSnr - 1, 0);
        const wiener = priorSnr / (priorSnr + 1);
        gNoise = Math.max(noiseGainFloor, Math.min(1, wiener));
        prevPostSnr[b] = Math.min(postSnr, 50);
        prevNoiseGain[b] = gNoise;
      }

      // Power remaining after noise suppression feeds the late-reverb decay model
      const denoisedPower = power * gNoise * gNoise;
      powerRing[ringHead][b] = denoisedPower;

      let gReverb = 1.0;
      if (runReverb && reverbStrength > 0) {
        // Exponential decay model of late reverberation energy
        const dec = binDecay[b];
        lateReverbPsd[b] = Math.max(
          dec * lateReverbPsd[b],
          dec * delayedPower[b] * 0.55
        );
        // Only suppress in the 120 Hz – 8 kHz band where room echo muddies speech
        const freqHz = (b * sampleRate) / STFT_SIZE;
        if (freqHz >= 120 && freqHz <= 8500 && denoisedPower > 1e-12) {
          const predictedLate = lateReverbPsd[b] * reverbBeta;
          const cleanRatio = Math.max(0, 1 - predictedLate / (denoisedPower + 1e-12));
          gReverb = Math.max(reverbGainFloor, Math.min(1, Math.sqrt(cleanRatio)));
        }
      }

      rawGain[b] = gNoise * gReverb;
    }

    ringHead = (ringHead + 1) % ringSize;

    // 4. Frequency + Temporal gain smoothing (prevents musical noise & watery artifacts)
    for (let b = 0; b < HALF_BINS; b++) {
      const gPrev = rawGain[Math.max(0, b - 1)];
      const gCurr = rawGain[b];
      const gNext = rawGain[Math.min(HALF_BINS - 1, b + 1)];
      const freqSmooth = 0.2 * gPrev + 0.6 * gCurr + 0.2 * gNext;
      // Fast rise on speech attack (0.35), smooth gentle release (0.72)
      const alphaTime = freqSmooth > prevTotalGain[b] ? 0.35 : 0.72;
      const finalG = alphaTime * prevTotalGain[b] + (1 - alphaTime) * freqSmooth;
      smoothGain[b] = finalG;
      prevTotalGain[b] = finalG;
    }

    // 5. Apply identical real-valued gain to all channels (preserves exact stereo phase!)
    for (let ch = 0; ch < numChannels; ch++) {
      const re = reals[ch];
      const im = imags[ch];
      const out = paddedOuts[ch];

      re[0] *= smoothGain[0];
      im[0] *= smoothGain[0];
      re[STFT_SIZE >> 1] *= smoothGain[HALF_BINS - 1];
      im[STFT_SIZE >> 1] *= smoothGain[HALF_BINS - 1];

      for (let b = 1; b < STFT_SIZE >> 1; b++) {
        const g = smoothGain[b];
        re[b] *= g;
        im[b] *= g;
        re[STFT_SIZE - b] *= g;
        im[STFT_SIZE - b] *= g;
      }

      ifftRadix2(re, im);

      for (let i = 0; i < STFT_SIZE; i++) {
        const dstIdx = frameStart + i;
        if (dstIdx < paddedLen) {
          out[dstIdx] += re[i] * HANN_2048[i] * HANN_75_NORM;
        }
      }
    }
  }

  // Strip padding so output chunk matches input chunk length sample-for-sample
  return paddedOuts.map((ch) => ch.subarray(pad, pad + chunkLen));
}

/**
 * Step 5: Gentle Dynamic De-Esser (5–9 kHz bandpass sidechain, stereo-linked).
 * Only attenuates sharp sibilance spikes above threshold; preserves natural air.
 */
function runGentleDeEsser(channels, sampleRate) {
  const numChannels = channels.length;
  const length = channels[0].length;

  // Bandpass centered at 6.8 kHz (Q = 1.6 -> ~5.0–9.0 kHz sibilance band)
  const sibCoeffs = makeBandPassBiquad(sampleRate, 6800, 1.6);
  const sibBands = channels.map((ch) => applyBiquad(ch, sibCoeffs));

  const attackCoeff = Math.exp(-1 / Math.max(1, 0.002 * sampleRate));
  const releaseCoeff = Math.exp(-1 / Math.max(1, 0.05 * sampleRate));
  const thresholdDb = -24; // Gentle default threshold
  const maxReductionDb = 5.0; // Never lispy

  const out = channels.map(() => new Float32Array(length));
  let env = 0;

  for (let i = 0; i < length; i++) {
    let linkedSib = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const a = Math.abs(sibBands[ch][i]);
      if (a > linkedSib) linkedSib = a;
    }

    env =
      linkedSib > env
        ? attackCoeff * env + (1 - attackCoeff) * linkedSib
        : releaseCoeff * env + (1 - releaseCoeff) * linkedSib;

    const sibDb = linearToDb(env);
    let atten = 1.0;
    if (sibDb > thresholdDb) {
      const grDb = Math.min(maxReductionDb, (sibDb - thresholdDb) * 0.5);
      atten = dbToLinear(-grDb);
    }

    const subtract = 1 - atten;
    for (let ch = 0; ch < numChannels; ch++) {
      out[ch][i] = channels[ch][i] - sibBands[ch][i] * subtract;
    }
  }

  return out;
}

/**
 * Step 6: Gentle Tone Shaping (strictly within ±3 dB).
 * -2.0 dB at 250 Hz (mud cleanup) and +1.5 dB at 3.5 kHz (gentle presence).
 */
function runGentleToneShaping(channels, sampleRate) {
  const mudCut = makePeakingBiquad(sampleRate, 250, -2.0, 1.0);
  const presenceLift = makePeakingBiquad(sampleRate, 3500, 1.5, 0.85);
  return channels.map((ch) => applyBiquad(applyBiquad(ch, mudCut), presenceLift));
}

/**
 * Step 7b: 4x Oversampled Lookahead True-Peak Limiter (default ceiling -1.5 dBTP).
 * Uses cubic Catmull-Rom interpolation to catch inter-sample peaks and a 5 ms
 * monotonic-deque lookahead envelope so limiting is smooth and never clips.
 */
function applyTruePeakLimiter(channels, sampleRate, ceilingDbtp = -1.5) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const ceilingLinear = dbToLinear(ceilingDbtp);
  const lookaheadSamples = Math.max(1, Math.round(0.005 * sampleRate));
  const releaseCoeff = Math.exp(-1 / (0.065 * sampleRate));
  const attackCoeff = Math.exp(-1 / Math.max(1, 0.0015 * sampleRate));
  const fracs = [0.25, 0.5, 0.75];

  const peakEnv = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    let maxP = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const d = channels[ch];
      const absV = Math.abs(d[i]);
      if (absV > maxP) maxP = absV;

      if (absV > ceilingLinear * 0.72 && i >= 1 && i + 2 < length) {
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
    while (tail > head && peakEnv[deque[tail - 1]] <= val) tail--;
    deque[tail++] = i;
    if (deque[head] > i + lookaheadSamples) head++;
    const winPeak = peakEnv[deque[head]];
    desiredGain[i] = winPeak > ceilingLinear ? ceilingLinear / winPeak : 1.0;
  }

  const out = channels.map(() => new Float32Array(length));
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
      let s = channels[ch][i] * gain;
      if (s > ceilingLinear) s = ceilingLinear;
      else if (s < -ceilingLinear) s = -ceilingLinear;
      out[ch][i] = s;
    }
  }

  return { channels: out, maxGrDb };
}

/**
 * Step 7a: Loudness Match to input integrated LUFS (within ±0.5 LU) + -1.5 dBTP limiter.
 */
function runLoudnessMatchAndLimit(
  channels,
  sampleRate,
  targetLufs,
  ceilingDbtp = -1.5
) {
  const numChannels = channels.length;
  const length = channels[0].length;

  const beforeLufs = measureBS1770Loudness(channels, sampleRate).integratedLufs;
  if (beforeLufs <= -68 || targetLufs <= -68) {
    const lim = applyTruePeakLimiter(channels, sampleRate, ceilingDbtp);
    return { channels: lim.channels, appliedGainDb: 0, limiterMaxGrDb: lim.maxGrDb };
  }

  let totalGainDb = Math.max(-18, Math.min(18, targetLufs - beforeLufs));
  const scale1 = dbToLinear(totalGainDb);
  let scaled = channels.map((ch) => {
    const arr = new Float32Array(length);
    for (let i = 0; i < length; i++) arr[i] = ch[i] * scale1;
    return arr;
  });

  let lim = applyTruePeakLimiter(scaled, sampleRate, ceilingDbtp);
  let out = lim.channels;
  let maxGrDb = lim.maxGrDb;

  // Iterative precision trim so output integrated LUFS lands strictly within ±0.5 LU
  // (never exceeding input loudness if peak-constrained).
  for (let iter = 0; iter < 2; iter++) {
    const measured = measureBS1770Loudness(out, sampleRate).integratedLufs;
    const diff = targetLufs - measured;
    if (Math.abs(diff) <= 0.25) break;
    const trimDb = Math.max(-6, Math.min(4, diff));
    const trimLin = dbToLinear(trimDb);
    const trimmed = out.map((ch) => {
      const arr = new Float32Array(length);
      for (let i = 0; i < length; i++) arr[i] = ch[i] * trimLin;
      return arr;
    });
    const lim2 = applyTruePeakLimiter(trimmed, sampleRate, ceilingDbtp);
    out = lim2.channels;
    maxGrDb = Math.max(maxGrDb, lim2.maxGrDb);
    totalGainDb += trimDb;
  }

  return { channels: out, appliedGainDb: totalGainDb, limiterMaxGrDb: maxGrDb };
}

/**
 * Runs the complete Stage 1 (and optional Stage 2 AI denoise) Voice Clean pipeline.
 *
 * @param {Float32Array[]} inputChannels - Unmodified input PCM channels
 * @param {number} sampleRate
 * @param {object} options - Voice Clean options
 * @param {object} [hooks] - Optional `{ onProgress, aiDenoiseFn, shouldAbort }`
 * @returns {Promise<{ channels: Float32Array[], report: object }>}
 */
export async function runVoiceCleanPipeline(
  inputChannels,
  sampleRate,
  options = {},
  hooks = {}
) {
  const { onProgress, aiDenoiseFn, shouldAbort } = hooks;
  const checkAbort = () => {
    if (shouldAbort?.()) {
      const err = new Error('Voice Clean cancelled by user.');
      err.name = 'AbortError';
      throw err;
    }
  };

  const numChannels = inputChannels.length;
  const length = inputChannels[0]?.length || 0;
  const warnings = [];

  if (length === 0 || numChannels === 0) {
    return {
      channels: inputChannels.map((c) => Float32Array.from(c)),
      report: {
        skipped: true,
        inputLufs: -70,
        outputLufs: -70,
        inputPeakDbtp: -100,
        outputPeakDbtp: -100,
        estimatedNoiseFloorDb: -100,
        outputNoiseFloorDb: -100,
        estimatedRt60Sec: null,
        reverbSkipped: true,
        noiseSource: 'auto',
        usedAiDenoise: false,
        warnings: ['Empty audio buffer — nothing to clean.'],
      },
    };
  }

  onProgress?.({ percent: 4, stage: 'Voice Clean: Inspecting input loudness, noise floor & room echo...' });
  checkAbort();

  // Never mutate input buffers — work strictly on copies
  const originalCopy = inputChannels.map((ch) => Float32Array.from(ch));
  const inputStats = inspectSignalLevels(originalCopy, sampleRate);
  const inputLufs = measureBS1770Loudness(originalCopy, sampleRate).integratedLufs;
  const inputPeakDbtp = linearToDb(estimateTruePeak(originalCopy));

  // Guard 1: Silent input must never crash or amplify digital silence
  if (inputStats.peakDb < SILENT_PEAK_THRESHOLD_DB) {
    warnings.push('Input audio is silent (peak below -72 dBFS) — Voice Clean skipped safely.');
    onProgress?.({ percent: 100, stage: 'Voice Clean: Silent input skipped safely.' });
    return {
      channels: originalCopy,
      report: {
        skipped: true,
        inputLufs,
        outputLufs: inputLufs,
        inputPeakDbtp,
        outputPeakDbtp: inputPeakDbtp,
        estimatedNoiseFloorDb: inputStats.noiseFloorDb,
        outputNoiseFloorDb: inputStats.noiseFloorDb,
        estimatedRt60Sec: null,
        reverbSkipped: true,
        noiseSource: 'auto',
        usedAiDenoise: false,
        warnings,
      },
    };
  }

  // Warn gently on very long files (> 10 minutes) that chunked mode is active
  const durationSec = length / sampleRate;
  if (durationSec > 600) {
    warnings.push(
      `Long recording (${(durationSec / 60).toFixed(1)} min) — running in low-memory chunked mode.`
    );
  }

  // Step 0b (optional): Automatic 50/60 Hz mains hum detection & 4-harmonic notch removal
  // Runs BEFORE the 75 Hz HPF so 50/60 Hz fundamental is accurately measured and its
  // 100/120/150/180/200/240 Hz harmonics are cleanly removed.
  let preHpfSource = originalCopy;
  let humReport = null;
  if (options.humDetect) {
    onProgress?.({
      percent: 8,
      stage: 'Voice Clean: Scanning for 50/60 Hz mains hum & harmonics...',
    });
    checkAbort();
    const humRes = detectAndRemoveMainsHum(originalCopy, sampleRate, VOICE_TARGET.hum);
    preHpfSource = humRes.channels;
    humReport = humRes.report;
  }

  // Step 1: Gentle 75 Hz High-Pass Pre-Clean (12 dB/oct Butterworth)
  onProgress?.({ percent: 10, stage: 'Voice Clean (1/7): Gentle 75 Hz high-pass rumble filter...' });
  checkAbort();
  const hpfCoeffs = makeHighPass2ndOrder(sampleRate, 75);
  const dryPreClean = preHpfSource.map((ch) => applyBiquad(ch, hpfCoeffs));

  // Estimate global noise profile (user selection or quietest ~10% of frames)
  const { noisePsd, noiseFloorDb: estimatedNoiseFloorDb, source: noiseSource } =
    estimateGlobalNoisePsd(dryPreClean, sampleRate, options.noiseSampleRegion);

  // Estimate room reverberation (RT60)
  const reverbInfo = estimateRoomReverb(dryPreClean, sampleRate);
  const requestedNoisePct = Math.max(0, Math.min(100, options.noiseReduction ?? 60));
  const requestedReverbPct = Math.max(0, Math.min(100, options.reverbReduction ?? 40));
  const naturalBlendFrac = Math.max(0, Math.min(1, (options.naturalBlend ?? 15) / 100));

  let effectiveNoisePct = requestedNoisePct;
  let adaptiveDenoiseStrength = null;

  if (options.adaptiveDenoise) {
    // Voice Target SNR-adaptive denoise curve:
    // snrDb > 50 ? 0 : snrDb > 30 ? 0.3 : snrDb > 15 ? 0.6 : 0.85
    adaptiveDenoiseStrength = evaluateVoiceTargetDenoise(inputStats.snrDb);
    effectiveNoisePct = Math.round(adaptiveDenoiseStrength * 100);
    if (adaptiveDenoiseStrength === 0) {
      warnings.push(
        `High SNR detected (~${inputStats.snrDb.toFixed(1)} dB > 50 dB) — Voice Target skipped denoising automatically.`
      );
    }
  } else {
    // Clean-file protection: if input already has very high SNR (> 38 dB) and low floor (< -60 dB),
    // gently cap unnecessary spectral noise reduction so clean recordings are never made worse.
    const isAlreadyVeryClean =
      !options.noiseSampleRegion &&
      inputStats.snrDb > 38 &&
      estimatedNoiseFloorDb < -60;
    if (isAlreadyVeryClean && requestedNoisePct > 0) {
      effectiveNoisePct = Math.min(requestedNoisePct, 25);
      warnings.push(
        `Recording is already very clean (floor ${estimatedNoiseFloorDb.toFixed(1)} dBFS, SNR ~${inputStats.snrDb.toFixed(0)} dB) — softened noise reduction automatically to preserve quality.`
      );
    }
  }

  // Auto-skip reverb reduction if audio is already dry
  let runReverb = requestedReverbPct > 0;
  let reverbSkipped = false;
  if (runReverb && reverbInfo.isDry) {
    runReverb = false;
    reverbSkipped = true;
    warnings.push(
      `Audio already sounds dry (est. RT60 ~${reverbInfo.rt60Sec.toFixed(2)} s) — reverb reduction skipped automatically.`
    );
  }

  let working = dryPreClean.map((ch) => Float32Array.from(ch));
  let usedAiDenoise = false;

  // Step 2 (Stage 2 Optional): DeepFilterNet AI Denoise replaces spectral noise step
  if (options.useAiDenoise && typeof aiDenoiseFn === 'function') {
    try {
      onProgress?.({
        percent: 16,
        stage: 'Voice Clean (2/7): Running AI denoise (DeepFilterNet3)...',
      });
      checkAbort();
      working = await aiDenoiseFn(working, sampleRate, {
        noiseReduction: effectiveNoisePct,
        onProgress: (p) => {
          const pct = Math.round(16 + (p.percent || 0) * 0.42);
          onProgress?.({ percent: pct, stage: p.stage || 'Voice Clean: AI denoise...' });
        },
        shouldAbort,
      });
      usedAiDenoise = true;
    } catch (aiErr) {
      if (aiErr?.name === 'AbortError') throw aiErr;
      warnings.push(
        `AI denoise unavailable (${aiErr?.message || 'WASM error'}) — fell back cleanly to browser Spectral Noise Reduction.`
      );
      usedAiDenoise = false;
    }
  }

  // Step 2 & 3 (Chunked STFT with 50–80 ms raised-cosine crossfade for low memory):
  // Runs Spectral Noise Reduction (if AI denoise wasn't used) + Late-Reverb Suppression
  const runSpectralNoise = !usedAiDenoise && effectiveNoisePct > 0;
  if (runSpectralNoise || runReverb) {
    const chunkSamples = Math.max(STFT_SIZE * 4, Math.round(CHUNK_DURATION_SEC * sampleRate));
    const overlapSamples = Math.min(
      Math.round(CHUNK_OVERLAP_SEC * sampleRate),
      Math.floor(chunkSamples / 4)
    );
    const stepSamples = Math.max(1, chunkSamples - overlapSamples);
    const numChunks = Math.max(1, Math.ceil(Math.max(0, length - overlapSamples) / stepSamples));

    const assembled = Array.from({ length: numChannels }, () => new Float32Array(length));
    const stageDesc =
      runSpectralNoise && runReverb
        ? 'Spectral noise + late-reverb suppression'
        : runSpectralNoise
          ? 'Spectral noise reduction (2048-pt STFT)'
          : 'Late-reverb suppression (2048-pt STFT)';

    const basePct = usedAiDenoise ? 60 : 18;
    const spanPct = usedAiDenoise ? 16 : 56;

    for (let c = 0; c < numChunks; c++) {
      checkAbort();
      const start = c * stepSamples;
      const end = Math.min(length, start + chunkSamples);
      const chunkIn = working.map((ch) => ch.subarray(start, end));

      onProgress?.({
        percent: Math.round(basePct + (c / numChunks) * spanPct),
        stage: `Voice Clean (2–3/7): ${stageDesc}${
          numChunks > 1 ? ` (chunk ${c + 1}/${numChunks})` : ''
        }...`,
      });

      const chunkOut = processStftChunk(chunkIn, sampleRate, {
        runNoise: runSpectralNoise,
        noisePsd,
        noiseReductionPct: effectiveNoisePct,
        runReverb,
        reverbReductionPct: requestedReverbPct,
        estimatedRt60Sec: reverbInfo.rt60Sec,
      });

      const len = end - start;
      const fadeInLen = c > 0 ? Math.min(overlapSamples, len) : 0;

      for (let ch = 0; ch < numChannels; ch++) {
        const dst = assembled[ch];
        const src = chunkOut[ch];

        // Raised-cosine crossfade over chunk overlap avoids any boundary clicks
        for (let i = 0; i < fadeInLen; i++) {
          const t = i / fadeInLen;
          const wNew = 0.5 * (1 - Math.cos(Math.PI * t));
          const wOld = 1 - wNew;
          dst[start + i] = dst[start + i] * wOld + src[i] * wNew;
        }
        for (let i = fadeInLen; i < len; i++) {
          dst[start + i] = src[i];
        }
      }
    }

    working = assembled;
  }

  checkAbort();

  // Step 4: Natural-Voice Blend ("Keep natural voice", default 15%)
  // Mixes the time-aligned pre-cleaned dry signal back in to prevent robotic artifacts.
  if (naturalBlendFrac > 0.001) {
    onProgress?.({
      percent: 78,
      stage: `Voice Clean (4/7): Blending ${Math.round(naturalBlendFrac * 100)}% natural voice...`,
    });
    const wetWeight = 1 - naturalBlendFrac;
    for (let ch = 0; ch < numChannels; ch++) {
      const w = working[ch];
      const d = dryPreClean[ch];
      for (let i = 0; i < length; i++) {
        w[i] = w[i] * wetWeight + d[i] * naturalBlendFrac;
      }
    }
  }

  checkAbort();

  // Step 5: Gentle Dynamic De-Esser (5–9 kHz)
  onProgress?.({ percent: 83, stage: 'Voice Clean (5/7): Gentle 5–9 kHz dynamic de-esser...' });
  working = runGentleDeEsser(working, sampleRate);
  checkAbort();

  // Step 6: Gentle Tone Shaping (-2 dB @ 250 Hz, +1.5 dB @ 3.5 kHz)
  // Skipped when an external Auto-EQ (such as Voice Target ±3 dB Auto-EQ) follows.
  if (!options.skipToneShaping) {
    onProgress?.({
      percent: 88,
      stage: 'Voice Clean (6/7): Gentle tone shaping (-2 dB mud, +1.5 dB presence)...',
    });
    working = runGentleToneShaping(working, sampleRate);
    checkAbort();
  }

  // Step 7: Loudness Match (within ±0.5 LU of input) + True-Peak Limiter (-1.5 dBTP)
  const shouldMatchLoudness = options.keepOriginalLoudness !== false;
  const safetyCeiling =
    typeof options.limiterCeilingDbtp === 'number' ? options.limiterCeilingDbtp : -1.5;
  let appliedGainDb = 0;
  let limiterMaxGrDb = 0;

  if (shouldMatchLoudness) {
    onProgress?.({
      percent: 93,
      stage: 'Voice Clean (7/7): Matching input LUFS (±0.5 LU) & limiting at -1.5 dBTP...',
    });
    const matched = runLoudnessMatchAndLimit(working, sampleRate, inputLufs, safetyCeiling);
    working = matched.channels;
    appliedGainDb = matched.appliedGainDb;
    limiterMaxGrDb = matched.limiterMaxGrDb;
  } else {
    onProgress?.({
      percent: 93,
      stage: `Voice Clean (7/7): Applying ${safetyCeiling.toFixed(1)} dBTP true-peak safety limiter...`,
    });
    const limited = applyTruePeakLimiter(working, sampleRate, safetyCeiling);
    working = limited.channels;
    limiterMaxGrDb = limited.maxGrDb;
  }

  checkAbort();

  const outputLufs = measureBS1770Loudness(working, sampleRate).integratedLufs;
  const outputPeakDbtp = linearToDb(estimateTruePeak(working));
  const outputStats = inspectSignalLevels(working, sampleRate);
  const outputReverb = estimateRoomReverb(working, sampleRate);

  onProgress?.({ percent: 100, stage: 'Voice Clean complete.' });

  return {
    channels: working,
    report: {
      skipped: false,
      inputLufs,
      outputLufs,
      loudnessDeltaLu: outputLufs - inputLufs,
      inputPeakDbtp,
      outputPeakDbtp,
      estimatedNoiseFloorDb,
      outputNoiseFloorDb: outputStats.noiseFloorDb,
      measuredSnrDb: inputStats.snrDb,
      adaptiveDenoiseStrength,
      effectiveNoisePct,
      humReport,
      estimatedRt60Sec: reverbInfo.rt60Sec,
      outputRt60Sec: outputReverb.rt60Sec,
      reverbSkipped,
      noiseSource,
      usedAiDenoise,
      appliedGainDb,
      limiterMaxGrDb,
      warnings,
    },
  };
}
