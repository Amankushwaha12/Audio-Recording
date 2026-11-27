/**
 * voiceTarget.js — Voice Target Mastering Profile & Mains Hum Detection/Removal
 *
 * Implements the exact Voice Target specification:
 *   - Integrated loudness: -16 LUFS (±0.5 LU tolerance)
 *   - True-peak ceiling: -1.0 dBTP (-1.5 dBTP option for MP3/AAC delivery)
 *   - Loudness Range (LRA): 4–8 LU (warns if processed audio falls outside)
 *   - Auto-Leveler: enabled, gentle strength, ±6 dB max gain
 *   - Auto-EQ: enabled, strictly bounded within ±3 dB max adjustment
 *   - SNR-Adaptive Denoise: snrDb > 50 ? 0 : > 30 ? 0.3 : > 15 ? 0.6 : 0.85
 *   - Mains Hum Removal: auto-detect 50 Hz / 60 Hz + up to 4 harmonics
 *   - Dry Mix: 15% original voice blended back in to prevent robotic artifacts
 */

import { applyBiquad, linearToDb } from './analyzer.js';
import { fftRadix2 } from './fft.js';

export const VOICE_TARGET = {
  loudnessLUFS: -16, // integrated target
  loudnessTolerance: 0.5, // LU
  truePeakDb: -1.0, // -1.5 for MP3/AAC
  lraRange: [4, 8], // LU, warn if outside
  leveler: { enabled: true, maxGainDb: 6, strength: 'gentle' },
  autoEQ: { enabled: true, maxAdjustDb: 3 },
  denoise: (snrDb) =>
    snrDb > 50 ? 0 : snrDb > 30 ? 0.3 : snrDb > 15 ? 0.6 : 0.85, // 0–1 strength
  hum: { detect: true, mains: [50, 60], harmonics: 4 },
  dryMix: 0.15, // keep some original to avoid robotic voice
};

/**
 * Evaluates the Voice Target SNR-adaptive denoise strength (0–1).
 */
export function evaluateVoiceTargetDenoise(snrDb) {
  if (!Number.isFinite(snrDb)) return 0.6;
  return VOICE_TARGET.denoise(snrDb);
}

/**
 * Maps symbolic leveler strength ('gentle' | 'moderate' | 'strong') to numeric [0..1].
 */
export function levelerStrengthToNumber(strength) {
  if (typeof strength === 'number' && Number.isFinite(strength)) {
    return Math.max(0.1, Math.min(1, strength));
  }
  if (strength === 'gentle') return 0.45;
  if (strength === 'strong') return 0.8;
  return 0.65;
}

/**
 * Designs a narrow-Q 2nd-order biquad notch filter centered at `freqHz`.
 */
function makeNotchCoeffs(sampleRate, freqHz, q = 26) {
  const f0 = Math.max(20, Math.min(freqHz, sampleRate * 0.45));
  const w0 = (2 * Math.PI * f0) / sampleRate;
  const cosW0 = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: 1 / a0,
    b1: (-2 * cosW0) / a0,
    b2: 1 / a0,
    a1: (-2 * cosW0) / a0,
    a2: (1 - alpha) / a0,
  };
}

/**
 * Detects 50 Hz or 60 Hz electrical mains hum and up to `harmonics` harmonics
 * using averaged high-resolution 16384-point Hann FFT frames (~2.7–2.9 Hz/bin),
 * then notches out only confirmed narrowband hum harmonics with surgical high-Q
 * biquad filters so clean speech harmonics are never hollowed out.
 *
 * @param {Float32Array[]} channels
 * @param {number} sampleRate
 * @param {{ detect?: boolean, mains?: number[], harmonics?: number }} [humConfig]
 * @returns {{
 *   channels: Float32Array[],
 *   report: {
 *     detected: boolean,
 *     mainsHz: number | null,
 *     harmonicsNotched: number[],
 *     prominenceDb: number
 *   }
 * }}
 */
export function detectAndRemoveMainsHum(channels, sampleRate, humConfig = VOICE_TARGET.hum) {
  const emptyReport = {
    detected: false,
    mainsHz: null,
    harmonicsNotched: [],
    prominenceDb: 0,
  };

  if (!humConfig?.detect || !channels?.length || !channels[0]?.length) {
    return { channels, report: emptyReport };
  }

  const numChannels = channels.length;
  const length = channels[0].length;
  const fftSize = 16384; // ~2.93 Hz bin width at 48 kHz -> cleanly separates 50 vs 60 Hz
  if (length < 2048) {
    return { channels, report: emptyReport };
  }

  const mainsCandidates =
    Array.isArray(humConfig.mains) && humConfig.mains.length > 0
      ? humConfig.mains
      : [50, 60];
  const numHarmonics = Math.max(1, Math.min(8, humConfig.harmonics || 4));

  // Build mono sum for consistent hum spectrum analysis
  const mono = new Float32Array(length);
  const invCh = 1 / numChannels;
  for (let ch = 0; ch < numChannels; ch++) {
    const src = channels[ch];
    for (let i = 0; i < length; i++) mono[i] += src[i] * invCh;
  }

  // Prefer the quietest analysis windows (where mains hum stands out clearest from voice)
  const step = Math.max(2048, Math.floor(Math.max(1, length - fftSize) / 16));
  const candidateStarts = [];
  if (length <= fftSize) {
    candidateStarts.push({ start: 0, energy: 1 });
  } else {
    for (let start = 0; start + fftSize <= length; start += step) {
      let sq = 0;
      for (let i = 0; i < fftSize; i += 8) {
        const s = mono[start + i];
        sq += s * s;
      }
      candidateStarts.push({ start, energy: sq / (fftSize / 8) });
    }
  }

  // Exclude pure digital zero frames (< -95 dBFS), keep quietest 40% (min 3 frames)
  const nonZero = candidateStarts.filter((c) => c.energy > 1e-10);
  const pool = nonZero.length > 0 ? nonZero : candidateStarts;
  pool.sort((a, b) => a.energy - b.energy);
  const framesToAverage = Math.max(1, Math.min(pool.length, Math.max(3, Math.ceil(pool.length * 0.4))));

  const halfBins = (fftSize >> 1) + 1;
  const avgPsd = new Float64Array(halfBins);
  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  const win = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / fftSize));
  }

  for (let f = 0; f < framesToAverage; f++) {
    const start = pool[f].start;
    for (let i = 0; i < fftSize; i++) {
      const idx = start + i;
      re[i] = idx < length ? mono[idx] * win[i] : 0;
      im[i] = 0;
    }
    fftRadix2(re, im);
    for (let b = 0; b < halfBins; b++) {
      avgPsd[b] += re[b] * re[b] + im[b] * im[b];
    }
  }

  for (let b = 0; b < halfBins; b++) {
    avgPsd[b] /= framesToAverage;
  }

  const binHz = sampleRate / fftSize;

  // Measure tonal narrowband prominence of a target frequency vs its ±10..24 Hz local floor
  const measureToneProminence = (targetHz) => {
    const centerBin = Math.round(targetHz / binHz);
    if (centerBin < 3 || centerBin >= halfBins - 10) {
      return { prominenceDb: 0, peakDb: -120, peakHz: targetHz };
    }

    // Peak within ±1 bin (handles slight mains frequency drift ±0.5 Hz)
    let peakPow = 0;
    let peakBin = centerBin;
    for (let b = centerBin - 1; b <= centerBin + 1; b++) {
      if (avgPsd[b] > peakPow) {
        peakPow = avgPsd[b];
        peakBin = b;
      }
    }

    // Parabolic interpolation on log-power around peakBin for sub-bin frequency accuracy,
    // clamped within ±0.8 Hz of the nominal mains harmonic (mains grids hold ±0.1 Hz).
    let refinedHz = peakBin * binHz;
    if (peakBin > 1 && peakBin < halfBins - 1) {
      const yL = Math.log(Math.max(avgPsd[peakBin - 1], 1e-18));
      const yC = Math.log(Math.max(avgPsd[peakBin], 1e-18));
      const yR = Math.log(Math.max(avgPsd[peakBin + 1], 1e-18));
      const denom = yL - 2 * yC + yR;
      if (Math.abs(denom) > 1e-12) {
        const delta = Math.max(-0.5, Math.min(0.5, (0.5 * (yL - yR)) / denom));
        refinedHz = (peakBin + delta) * binHz;
      }
    }
    const clampedHz = Math.max(targetHz - 0.8, Math.min(targetHz + 0.8, refinedHz));

    // Local background median from [4..9] bins (~11–26 Hz) away on each side
    const bgVals = [];
    for (let offset = 4; offset <= 9; offset++) {
      if (peakBin - offset >= 1) bgVals.push(avgPsd[peakBin - offset]);
      if (peakBin + offset < halfBins) bgVals.push(avgPsd[peakBin + offset]);
    }
    bgVals.sort((a, b) => a - b);
    const medianBg = bgVals.length > 0 ? bgVals[Math.floor(bgVals.length / 2)] : 1e-12;

    const peakDb = 10 * Math.log10(Math.max(peakPow, 1e-16));
    const bgDb = 10 * Math.log10(Math.max(medianBg, 1e-16));
    // Normalize peak amplitude relative to full-scale FFT energy
    const relPeakDbfs = linearToDb(Math.sqrt(peakPow) / (fftSize * 0.25));
    return {
      prominenceDb: peakDb - bgDb,
      peakDb: relPeakDbfs,
      peakHz: clampedHz,
    };
  };

  let bestCandidate = null;

  for (const mainsHz of mainsCandidates) {
    const harmonics = [];
    let score = 0;
    let strongCount = 0;
    let maxProm = 0;

    for (let h = 1; h <= numHarmonics; h++) {
      const targetHz = mainsHz * h;
      const m = measureToneProminence(targetHz);
      // Require both narrowband prominence AND audible level (> -74 dBFS)
      const isPresent = m.prominenceDb >= 6.5 && m.peakDb > -74;
      harmonics.push({ h, targetHz, ...m, isPresent });
      if (isPresent) {
        strongCount++;
        score += m.prominenceDb * (h === 1 ? 1.35 : 1.0);
        if (m.prominenceDb > maxProm) maxProm = m.prominenceDb;
      }
    }

    const fundamentalStrong = harmonics[0]?.prominenceDb >= 9.0 && harmonics[0]?.peakDb > -72;
    const detected = fundamentalStrong || strongCount >= 2;

    if (detected && (!bestCandidate || score > bestCandidate.score)) {
      bestCandidate = {
        mainsHz,
        score,
        maxProm,
        harmonics,
      };
    }
  }

  if (!bestCandidate) {
    return { channels, report: emptyReport };
  }

  // Notch out all detected harmonics (and fundamental if detected) with surgical high-Q filters
  const toNotch = bestCandidate.harmonics.filter(
    (item) => item.isPresent || (item.h === 1 && item.prominenceDb >= 4.5 && item.peakDb > -76)
  );

  if (toNotch.length === 0) {
    return { channels, report: emptyReport };
  }

  let out = channels.map((ch) => Float32Array.from(ch));
  const notchedFreqs = [];

  for (const item of toNotch) {
    // Tighter Q on higher harmonics keeps adjacent speech formants completely untouched
    const q = Math.min(28, 16 + (item.h - 1) * 3);
    const coeffs = makeNotchCoeffs(sampleRate, item.peakHz, q);
    out = out.map((ch) => applyBiquad(ch, coeffs));
    notchedFreqs.push(Math.round(item.targetHz));
  }

  return {
    channels: out,
    report: {
      detected: true,
      mainsHz: bestCandidate.mainsHz,
      harmonicsNotched: notchedFreqs,
      prominenceDb: Number(bestCandidate.maxProm.toFixed(1)),
    },
  };
}
