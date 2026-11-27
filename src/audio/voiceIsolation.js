/**
 * voiceIsolation.js — Pure, Web-Worker-Safe Foreground Voice Isolation Engine
 *
 * Isolates human speech from stereo music beds, off-axis chatter, room ambience,
 * electrical mains hum, and broadband/non-stationary background noise without
 * hollowing out vocal warmth or breaking mid-word syllables.
 *
 * Processing pipeline:
 *   1. Vocal Band & Hum Pre-Clean:
 *      - Automatic 50/60 Hz mains hum & 4-harmonic notch removal
 *      - 4th-order 85 Hz Butterworth high-pass (sub-bass / kick / rumble cut)
 *      - 2nd-order 11.5 kHz Butterworth low-pass (ultrasonic hiss / cymbal trim)
 *   2. Optional Neural Pre-Isolation (DeepFilterNet3 48 kHz WASM inside Worker)
 *   3. Chunked 2048-pt 75%-Overlap Hann STFT Voice Isolation:
 *      a) Stereo Center-Voice Extraction:
 *         Uses Mid/Side dominance + L/R phase coherence to keep center-panned
 *         vocal energy intact while attenuating panned stereo instruments and
 *         diffuse room ambience. Auto-skips cleanly on mono/dual-mono files.
 *      b) Bidirectional Speech-Activity Detection (VAD):
 *         Tracks vocal-band energy (180–3800 Hz), spectral flatness, and harmonic
 *         peakiness with 25 ms lookahead and 220 ms hangover so word starts and
 *         trailing consonants never clip or break.
 *      c) Adaptive Background Tracker + Harmonic Speech Soft-Mask:
 *         Continuously tracks changing background noise spectra, preserves vocal
 *         harmonics & active consonants, and smoothly attenuates non-vocal bins
 *         and pauses down to the configured background floor with temporal +
 *         frequency smoothing (zero watery musical noise).
 *   4. Natural-Voice Blend + Safety True-Peak Limiter (-1.5 dBTP)
 */

import {
  applyBiquad,
  dbToLinear,
  estimateTruePeak,
  linearToDb,
  measureBS1770Loudness,
} from './analyzer.js';
import { fftRadix2, ifftRadix2 } from './fft.js';
import { VOICE_TARGET, detectAndRemoveMainsHum } from './voiceTarget.js';

export const DEFAULT_VOICE_ISOLATION_OPTIONS = {
  enabled: false,
  isolationStrength: 75, // 0–100% speech harmonic & spectral isolation strength
  centerFocus: 70, // 0–100% stereo center-voice extraction (auto-skips on mono)
  backgroundAttenuationDb: 24, // 10–36 dB max suppression during non-speech / background
  naturalBlend: 8, // 0–30% dry voice mixed back in for warmth
  vocalBandFocus: true, // 85 Hz HPF + 11.5 kHz LPF + 50/60 Hz hum notch
  useAiBoost: false, // Optional DeepFilterNet3 neural boost before STFT isolation
  keepOriginalLoudness: false, // Default false so mastering preset (-16 LUFS) can polish after
};

const STFT_SIZE = 2048;
const STFT_HOP = 512; // 75% overlap
const HALF_BINS = (STFT_SIZE >> 1) + 1;
const CHUNK_DURATION_SEC = 16;
const CHUNK_OVERLAP_SEC = 0.08;
const SILENT_PEAK_THRESHOLD_DB = -72;

function buildHannWindow(size) {
  const win = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / size));
  }
  return win;
}

const HANN_2048 = buildHannWindow(STFT_SIZE);
const HANN_75_NORM = 1 / 1.5;

function applyHighPass4thOrder(channels, sampleRate, cutoffHz = 85) {
  const f0 = Math.max(20, Math.min(cutoffHz, sampleRate * 0.4));
  const w0 = (2 * Math.PI * f0) / sampleRate;
  const cosW0 = Math.cos(w0);
  const sinW0 = Math.sin(w0);
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

function applyLowPass2ndOrder(channels, sampleRate, cutoffHz = 11500) {
  const f0 = Math.max(2000, Math.min(cutoffHz, sampleRate * 0.44));
  const w0 = (2 * Math.PI * f0) / sampleRate;
  const cosW0 = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
  const a0 = 1 + alpha;
  const coeffs = {
    b0: ((1 - cosW0) / 2) / a0,
    b1: (1 - cosW0) / a0,
    b2: ((1 - cosW0) / 2) / a0,
    a1: (-2 * cosW0) / a0,
    a2: (1 - alpha) / a0,
  };
  return channels.map((ch) => applyBiquad(ch, coeffs));
}

function inspectLevels(channels, sampleRate) {
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
  const nonZero = [];
  const active = [];
  for (let f = 0; f < numFrames; f++) {
    const start = f * frameSamples;
    const end = Math.min(length, start + frameSamples);
    let sq = 0;
    let cnt = 0;
    for (let i = start; i < end; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        const s = channels[ch][i];
        sq += s * s;
        cnt++;
      }
    }
    const db = linearToDb(Math.sqrt(sq / Math.max(1, cnt)));
    if (db > -85) nonZero.push(db);
    if (db > -45) active.push(db);
  }
  nonZero.sort((a, b) => a - b);
  active.sort((a, b) => a - b);

  const noiseFloorDb = nonZero.length > 0 ? nonZero[Math.floor(nonZero.length * 0.1)] : -100;
  const speechRmsDb =
    active.length > 0
      ? active[Math.floor(active.length * 0.8)]
      : nonZero.length > 0
        ? nonZero[Math.floor(nonZero.length * 0.85)]
        : -100;

  return { peak, peakDb, noiseFloorDb, speechRmsDb, snrDb: speechRmsDb - noiseFloorDb };
}

/**
 * Checks whether a 2-channel recording actually contains stereo difference (Side)
 * content worth center-extracting, vs mono or dual-mono where L ≈ R.
 */
export function measureStereoSideRatio(channels) {
  if (!channels || channels.length < 2) return 0;
  const L = channels[0];
  const R = channels[1];
  const len = Math.min(L.length, R.length);
  if (len === 0) return 0;
  const stride = Math.max(1, Math.floor(len / 48000));
  let midSq = 0;
  let sideSq = 0;
  for (let i = 0; i < len; i += stride) {
    const m = 0.5 * (L[i] + R[i]);
    const s = 0.5 * (L[i] - R[i]);
    midSq += m * m;
    sideSq += s * s;
  }
  return sideSq / (midSq + sideSq + 1e-12);
}

/**
 * Estimates initial per-bin background PSD from the quietest ~12% of non-zero frames
 * across the full recording (or from a user-selected waveform noise sample region).
 */
function estimateBackgroundPsd(channels, sampleRate, noiseSampleRegion) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const real = new Float32Array(STFT_SIZE);
  const imag = new Float32Array(STFT_SIZE);
  const bgPsd = new Float32Array(HALF_BINS);

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
    const regionFrames = Math.max(
      1,
      Math.floor((endSample - startSample - STFT_SIZE) / STFT_HOP) + 1
    );
    let count = 0;
    for (let f = 0; f < regionFrames; f++) {
      const base = startSample + f * STFT_HOP;
      for (let ch = 0; ch < numChannels; ch++) {
        const src = channels[ch];
        for (let i = 0; i < STFT_SIZE; i++) {
          const idx = base + i;
          real[i] = idx < endSample ? src[idx] * HANN_2048[i] : 0;
          imag[i] = 0;
        }
        fftRadix2(real, imag);
        for (let b = 0; b < HALF_BINS; b++) {
          bgPsd[b] += real[b] * real[b] + imag[b] * imag[b];
        }
        count++;
      }
    }
    if (count > 0) {
      for (let b = 0; b < HALF_BINS; b++) bgPsd[b] /= count;
      return { bgPsd, source: 'selected-region' };
    }
  }

  const totalFrames = Math.max(1, Math.floor((length - STFT_SIZE) / STFT_HOP) + 1);
  const scanStride = Math.max(1, Math.floor(totalFrames / 1000));
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
    candidates.push({ f, meanSq: cnt > 0 ? sq / cnt : 0 });
  }

  const minSq = Math.pow(10, -85 / 10);
  const quietDigitalSq = Math.pow(10, -75 / 10);
  const digitalSilentCount = candidates.filter((c) => c.meanSq <= quietDigitalSq).length;
  const nonZero = candidates.filter((c) => c.meanSq > minSq);
  const pool =
    digitalSilentCount >= Math.max(3, Math.ceil(candidates.length * 0.12))
      ? candidates
      : nonZero.length >= 4
        ? nonZero
        : candidates;
  pool.sort((a, b) => a.meanSq - b.meanSq);

  const quietCount = Math.max(2, Math.min(pool.length, Math.ceil(pool.length * 0.12)));
  let accum = 0;

  for (let q = 0; q < quietCount; q++) {
    const start = pool[q].f * STFT_HOP;
    for (let ch = 0; ch < numChannels; ch++) {
      const src = channels[ch];
      for (let i = 0; i < STFT_SIZE; i++) {
        const idx = start + i;
        real[i] = idx < length ? src[idx] * HANN_2048[i] : 0;
        imag[i] = 0;
      }
      fftRadix2(real, imag);
      for (let b = 0; b < HALF_BINS; b++) {
        bgPsd[b] += real[b] * real[b] + imag[b] * imag[b];
      }
      accum++;
    }
  }

  for (let b = 0; b < HALF_BINS; b++) {
    bgPsd[b] /= Math.max(1, accum);
  }
  return { bgPsd, source: 'auto' };
}

/**
 * Smoothstep helper mapping x in [edge0, edge1] smoothly to [0, 1].
 */
function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(1e-9, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * Core 2048-pt 75%-overlap Hann STFT Voice Isolation chunk processor.
 *
 * Pass 1: Analyze all frames in chunk (with stereo center-voice extraction applied
 *         in-place to the spectrum when stereo), extracting vocal-band energy,
 *         spectral flatness, and harmonic prominence to build a lookahead +
 *         hangover Voice Activity Probability curve `speechProb[f]`.
 * Pass 2: Apply adaptive background Wiener suppression + harmonic speech soft-mask
 *         + frequency/temporal smoothing and COLA overlap-add reconstruction.
 */
function processVoiceIsolationChunk(
  chunkChannels,
  sampleRate,
  {
    initialBgPsd,
    isolationStrengthPct,
    applyStereoCenter,
    centerFocusPct,
    backgroundAttenuationDb,
  }
) {
  const numChannels = chunkChannels.length;
  const chunkLen = chunkChannels[0].length;
  const pad = STFT_SIZE - STFT_HOP;
  const paddedLen = chunkLen + 2 * pad;
  const numFrames = Math.max(1, Math.ceil((paddedLen - STFT_SIZE) / STFT_HOP) + 1);

  const isoStrength = Math.max(0, Math.min(1, isolationStrengthPct / 100));
  const centerFocus = applyStereoCenter ? Math.max(0, Math.min(1, centerFocusPct / 100)) : 0;
  const bgFloorLinear = dbToLinear(-Math.max(6, Math.min(38, backgroundAttenuationDb)));
  // Active-speech floor stays gentle (-6 dB to -16 dB) so voice never turns metallic/thin
  const speechFloorLinear = dbToLinear(-(6 + isoStrength * 10));
  const overSub = 1.1 + isoStrength * 1.15;
  const alphaDD = 0.93;

  const binHz = sampleRate / STFT_SIZE;
  const vocalMinBin = Math.max(2, Math.floor(170 / binHz));
  const vocalMaxBin = Math.min(HALF_BINS - 1, Math.ceil(3800 / binHz));
  const consMinBin = Math.min(HALF_BINS - 1, Math.floor(2600 / binHz));
  const consMaxBin = Math.min(HALF_BINS - 1, Math.ceil(8500 / binHz));

  // Frequency-band vocal prior: full pass 110–7800 Hz, gentle rolloff outside vocal range
  const bandPrior = new Float32Array(HALF_BINS);
  for (let b = 0; b < HALF_BINS; b++) {
    const f = b * binHz;
    if (f < 85) {
      bandPrior[b] = 1 - 0.65 * isoStrength * (1 - smoothstep(30, 85, f));
    } else if (f <= 7800) {
      bandPrior[b] = 1.0;
    } else {
      bandPrior[b] = 1 - 0.55 * isoStrength * smoothstep(7800, 13000, f);
    }
  }

  // Per-frame spectral buffers for Pass 1 (VAD) -> Pass 2 (masking)
  const frameVocalDb = new Float32Array(numFrames);
  const frameHarmonicity = new Float32Array(numFrames);
  const frameVocalRatio = new Float32Array(numFrames);

  const reals = Array.from({ length: numChannels }, () => new Float32Array(STFT_SIZE));
  const imags = Array.from({ length: numChannels }, () => new Float32Array(STFT_SIZE));

  // Helper: applies stereo center-voice extraction to reals/imags in-place for one frame
  const applyCenterMaskInPlace = () => {
    if (centerFocus <= 0.005 || numChannels !== 2) return;
    const lRe = reals[0];
    const lIm = imags[0];
    const rRe = reals[1];
    const rIm = imags[1];
    const spatialFloor = 1 - 0.88 * centerFocus;
    const sideKeep = 1 - 0.92 * centerFocus;

    for (let b = 0; b < HALF_BINS; b++) {
      const lr = lRe[b];
      const li = lIm[b];
      const rr = rRe[b];
      const ri = rIm[b];

      const mr = 0.5 * (lr + rr);
      const mi = 0.5 * (li + ri);
      const sr = 0.5 * (lr - rr);
      const si = 0.5 * (li - ri);

      const pM = mr * mr + mi * mi;
      const pS = sr * sr + si * si;
      const pAvg = 0.5 * (lr * lr + li * li + rr * rr + ri * ri);
      const cross = lr * rr + li * ri; // Positive when L & R are in-phase centered voice

      const coherence = Math.max(0, cross) / (pAvg + 1e-12);
      const midDom = pM / (pM + pS + 1e-12);
      const centerScore = Math.sqrt(coherence * midDom);
      // Smooth spatial curve: preserves center voice (>0.72), suppresses panned/diffuse (<0.45)
      const gSpatial = spatialFloor + (1 - spatialFloor) * smoothstep(0.32, 0.78, centerScore);

      const outMr = mr * gSpatial;
      const outMi = mi * gSpatial;
      const outSr = sr * sideKeep * gSpatial;
      const outSi = si * sideKeep * gSpatial;

      const nlRe = outMr + outSr;
      const nlIm = outMi + outSi;
      const nrRe = outMr - outSr;
      const nrIm = outMi - outSi;

      lRe[b] = nlRe;
      lIm[b] = nlIm;
      rRe[b] = nrRe;
      rIm[b] = nrIm;

      if (b > 0 && b < STFT_SIZE >> 1) {
        const conjIdx = STFT_SIZE - b;
        lRe[conjIdx] = nlRe;
        lIm[conjIdx] = -nlIm;
        rRe[conjIdx] = nrRe;
        rIm[conjIdx] = -nrIm;
      }
    }
  };

  // PASS 1: Fast frame-level vocal-activity & harmonicity scan across chunk
  const invCh = 1 / numChannels;
  const vocalBandPowers = new Float32Array(Math.max(1, vocalMaxBin - vocalMinBin + 1));

  for (let f = 0; f < numFrames; f++) {
    const frameStart = f * STFT_HOP;
    for (let ch = 0; ch < numChannels; ch++) {
      const src = chunkChannels[ch];
      const re = reals[ch];
      const im = imags[ch];
      for (let i = 0; i < STFT_SIZE; i++) {
        const srcIdx = frameStart + i - pad;
        re[i] = srcIdx >= 0 && srcIdx < chunkLen ? src[srcIdx] * HANN_2048[i] : 0;
        im[i] = 0;
      }
      fftRadix2(re, im);
    }

    if (centerFocus > 0.005 && numChannels === 2) {
      applyCenterMaskInPlace();
    }

    let vocalPow = 0;
    let totalPow = 0;
    let logSum = 0;
    let vCount = 0;

    for (let b = 1; b < HALF_BINS; b++) {
      let p = 0;
      for (let ch = 0; ch < numChannels; ch++) {
        const r = reals[ch][b];
        const m = imags[ch][b];
        p += r * r + m * m;
      }
      p *= invCh;
      totalPow += p;

      if (b >= vocalMinBin && b <= vocalMaxBin) {
        vocalPow += p;
        vocalBandPowers[vCount] = p;
        logSum += Math.log(p + 1e-14);
        vCount++;
      }
    }

    const meanVocalPow = vocalPow / Math.max(1, vCount);
    // Spectral flatness: 1 = flat broadband noise, near 0 = harmonic tonal voice
    const geoMean = Math.exp(logSum / Math.max(1, vCount));
    const flatness = Math.min(1, geoMean / (meanVocalPow + 1e-14));

    frameVocalDb[f] = 10 * Math.log10(meanVocalPow + 1e-14);
    frameHarmonicity[f] = Math.max(0, 1 - flatness * 1.4);
    frameVocalRatio[f] = vocalPow / (totalPow + 1e-14);
  }

  // Compute chunk vocal floor & active peak to calibrate smooth VAD probability
  const sortedVocal = Float32Array.from(frameVocalDb).sort();
  const vocalFloorDb = sortedVocal[Math.floor(sortedVocal.length * 0.12)];
  const vocalPeakDb = sortedVocal[Math.floor(sortedVocal.length * 0.92)];
  const dynamicSpan = Math.max(8, vocalPeakDb - vocalFloorDb);

  const rawSpeechProb = new Float32Array(numFrames);
  for (let f = 0; f < numFrames; f++) {
    const aboveFloorDb = frameVocalDb[f] - vocalFloorDb;
    const energyCue = smoothstep(3.5, Math.min(16, dynamicSpan * 0.45 + 4), aboveFloorDb);
    const tonalCue = smoothstep(0.18, 0.58, frameHarmonicity[f]);
    const bandCue = smoothstep(0.22, 0.65, frameVocalRatio[f]);
    // Energy is primary; harmonicity & vocal-band concentration boost confidence and reject hiss/thumps
    rawSpeechProb[f] = Math.max(
      0,
      Math.min(1, energyCue * (0.55 + 0.25 * tonalCue + 0.2 * bandCue))
    );
  }

  // Bidirectional VAD smoothing: 25 ms lookahead + 220 ms smooth hangover (never breaks syllables!)
  const hopSec = STFT_HOP / sampleRate;
  const lookaheadFrames = Math.max(1, Math.round(0.025 / hopSec));
  const hangoverFrames = Math.max(4, Math.round(0.22 / hopSec));
  const releaseDecay = Math.exp(-hopSec / 0.09);

  const speechProb = new Float32Array(numFrames);
  let hold = 0;
  let env = 0;

  for (let f = 0; f < numFrames; f++) {
    // Look ahead by 25 ms so consonants at word starts open BEFORE the word arrives
    let aheadMax = rawSpeechProb[f];
    const maxIdx = Math.min(numFrames - 1, f + lookaheadFrames);
    for (let k = f + 1; k <= maxIdx; k++) {
      if (rawSpeechProb[k] > aheadMax) aheadMax = rawSpeechProb[k];
    }

    if (aheadMax >= 0.45) {
      hold = hangoverFrames;
      env = Math.max(env, aheadMax);
    } else if (hold > 0) {
      hold--;
      env = Math.max(aheadMax, env * 0.975);
    } else {
      env = Math.max(aheadMax, env * releaseDecay);
    }
    speechProb[f] = Math.max(0, Math.min(1, env));
  }

  // Count active speech frames for diagnostics
  let activeSpeechFrameCount = 0;
  for (let f = 0; f < numFrames; f++) {
    if (speechProb[f] >= 0.4) activeSpeechFrameCount++;
  }

  // PASS 2: Adaptive Spectral Background Suppression + Harmonic Vocal Soft-Mask + COLA Reconstruction
  const paddedOuts = Array.from({ length: numChannels }, () => new Float32Array(paddedLen));
  const bgPsd = Float32Array.from(initialBgPsd);
  const framePower = new Float32Array(HALF_BINS);
  const prevPostSnr = new Float32Array(HALF_BINS).fill(1);
  const prevWienerGain = new Float32Array(HALF_BINS).fill(1);
  const prevTotalGain = new Float32Array(HALF_BINS).fill(1);
  const rawGain = new Float32Array(HALF_BINS);
  const smoothGain = new Float32Array(HALF_BINS);

  for (let f = 0; f < numFrames; f++) {
    const frameStart = f * STFT_HOP;
    for (let ch = 0; ch < numChannels; ch++) {
      const src = chunkChannels[ch];
      const re = reals[ch];
      const im = imags[ch];
      for (let i = 0; i < STFT_SIZE; i++) {
        const srcIdx = frameStart + i - pad;
        re[i] = srcIdx >= 0 && srcIdx < chunkLen ? src[srcIdx] * HANN_2048[i] : 0;
        im[i] = 0;
      }
      fftRadix2(re, im);
    }

    if (centerFocus > 0.005 && numChannels === 2) {
      applyCenterMaskInPlace();
    }

    for (let b = 0; b < HALF_BINS; b++) {
      let p = 0;
      for (let ch = 0; ch < numChannels; ch++) {
        const r = reals[ch][b];
        const m = imags[ch][b];
        p += r * r + m * m;
      }
      framePower[b] = p * invCh;
    }

    const sp = speechProb[f];

    // Adapt background spectrum gently during non-speech pauses so varying noise tracks cleanly
    if (sp < 0.25) {
      const adaptRate = 0.04 * (1 - sp * 4);
      for (let b = 0; b < HALF_BINS; b++) {
        // Never allow loud transients to spike the background tracker upward abruptly
        const target = Math.min(framePower[b], bgPsd[b] * 4 + 1e-10);
        bgPsd[b] = (1 - adaptRate) * bgPsd[b] + adaptRate * target;
      }
    }

    // Dynamic gain floor interpolates smoothly between active-speech floor & pause floor
    const frameFloor = bgFloorLinear + (speechFloorLinear - bgFloorLinear) * sp;

    for (let b = 0; b < HALF_BINS; b++) {
      const power = framePower[b];
      const nPow = Math.max(bgPsd[b] * overSub, 1e-12);
      const postSnr = power / nPow;
      const priorSnr =
        alphaDD * (prevWienerGain[b] * prevWienerGain[b] * prevPostSnr[b]) +
        (1 - alphaDD) * Math.max(postSnr - 1, 0);

      // Decision-directed Wiener gain
      let wiener = priorSnr / (priorSnr + 1);
      prevPostSnr[b] = Math.min(postSnr, 50);
      prevWienerGain[b] = Math.max(speechFloorLinear, Math.min(1, wiener));

      // Local harmonic prominence boost during speech: keeps vocal formants crystal clear
      if (sp > 0.35 && b >= vocalMinBin && b <= consMaxBin) {
        const pL = framePower[Math.max(1, b - 2)];
        const pR = framePower[Math.min(HALF_BINS - 1, b + 2)];
        const localAvg = 0.5 * (pL + pR) + 1e-12;
        const harmonicBoost = power > localAvg * 1.4 ? 1.12 : 1.0;
        wiener = Math.min(1, wiener * harmonicBoost);

        // Protect unvoiced consonants (2.6–8.5 kHz) during active speech
        if (b >= consMinBin && b <= consMaxBin) {
          wiener = Math.max(wiener, 0.35 * sp);
        }
      }

      // Blend spectral mask with speech-presence envelope and vocal band prior
      const pauseAtten = bgFloorLinear + (1 - bgFloorLinear) * sp;
      const isolatedG = Math.min(wiener, pauseAtten) * bandPrior[b];
      // Respect isolationStrength slider cleanly (at 0% isolation -> unity gain 1.0)
      const blendedG = 1 - isoStrength * (1 - isolatedG);

      rawGain[b] = Math.max(frameFloor, Math.min(1, blendedG));
    }

    // 3-bin frequency smoothing + asymmetric temporal smoothing (prevents musical noise)
    for (let b = 0; b < HALF_BINS; b++) {
      const gPrev = rawGain[Math.max(0, b - 1)];
      const gCurr = rawGain[b];
      const gNext = rawGain[Math.min(HALF_BINS - 1, b + 1)];
      const freqSmooth = 0.2 * gPrev + 0.6 * gCurr + 0.2 * gNext;
      const alphaTime = freqSmooth > prevTotalGain[b] ? 0.3 : 0.76;
      const finalG = alphaTime * prevTotalGain[b] + (1 - alphaTime) * freqSmooth;
      smoothGain[b] = finalG;
      prevTotalGain[b] = finalG;
    }

    // Apply identical real-valued gain across all channels (exact phase coherence)
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

  return {
    channels: paddedOuts.map((ch) => ch.subarray(pad, pad + chunkLen)),
    activeSpeechFrameCount,
    totalFrames: numFrames,
  };
}

/**
 * Simple 4x oversampled lookahead true-peak safety limiter for standalone Voice Isolation.
 */
function applySafetyTruePeakLimiter(channels, sampleRate, ceilingDbtp = -1.5) {
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
 * Runs the complete Web-Worker-safe Voice Isolation pipeline.
 *
 * @param {Float32Array[]} inputChannels - Unmodified input PCM channels
 * @param {number} sampleRate
 * @param {object} options - Voice Isolation options
 * @param {object} [hooks] - `{ onProgress, aiDenoiseFn, shouldAbort }`
 * @returns {Promise<{ channels: Float32Array[], report: object }>}
 */
export async function runVoiceIsolationPipeline(
  inputChannels,
  sampleRate,
  options = {},
  hooks = {}
) {
  const { onProgress, aiDenoiseFn, shouldAbort } = hooks;
  const checkAbort = () => {
    if (shouldAbort?.()) {
      const err = new Error('Voice Isolation cancelled by user.');
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
        mode: 'voiceIsolation',
        skipped: true,
        warnings: ['Empty audio buffer — nothing to isolate.'],
      },
    };
  }

  onProgress?.({ percent: 4, stage: 'Voice Isolation: Analyzing vocal band & stereo field...' });
  checkAbort();

  const originalCopy = inputChannels.map((ch) => Float32Array.from(ch));
  const inputStats = inspectLevels(originalCopy, sampleRate);
  const inputLufs = measureBS1770Loudness(originalCopy, sampleRate).integratedLufs;
  const inputPeakDbtp = linearToDb(estimateTruePeak(originalCopy));

  if (inputStats.peakDb < SILENT_PEAK_THRESHOLD_DB) {
    warnings.push('Input audio is silent (peak below -72 dBFS) — Voice Isolation skipped safely.');
    onProgress?.({ percent: 100, stage: 'Voice Isolation: Silent input skipped.' });
    return {
      channels: originalCopy,
      report: {
        mode: 'voiceIsolation',
        skipped: true,
        inputLufs,
        outputLufs: inputLufs,
        inputPeakDbtp,
        outputPeakDbtp: inputPeakDbtp,
        estimatedNoiseFloorDb: inputStats.noiseFloorDb,
        outputNoiseFloorDb: inputStats.noiseFloorDb,
        speechActivityPct: 0,
        stereoCenterApplied: false,
        stereoCenterSkipped: true,
        humReport: null,
        usedAiBoost: false,
        warnings,
      },
    };
  }

  const isolationStrength = Math.max(0, Math.min(100, options.isolationStrength ?? 75));
  const centerFocus = Math.max(0, Math.min(100, options.centerFocus ?? 70));
  const backgroundAttenuationDb = Math.max(
    10,
    Math.min(36, options.backgroundAttenuationDb ?? 24)
  );
  const naturalBlendFrac = Math.max(0, Math.min(0.4, (options.naturalBlend ?? 8) / 100));
  const vocalBandFocus = options.vocalBandFocus !== false;

  // Step 1: Automatic 50/60 Hz mains hum notch + Vocal Bandpass (85 Hz HPF / 11.5 kHz LPF)
  let preClean = originalCopy;
  let humReport = null;
  if (vocalBandFocus) {
    onProgress?.({
      percent: 10,
      stage: 'Voice Isolation (1/4): Scanning 50/60 Hz hum & focusing vocal band (85 Hz–11.5 kHz)...',
    });
    checkAbort();
    const humRes = detectAndRemoveMainsHum(preClean, sampleRate, VOICE_TARGET.hum);
    preClean = humRes.channels;
    humReport = humRes.report;
    preClean = applyHighPass4thOrder(preClean, sampleRate, 85);
    preClean = applyLowPass2ndOrder(preClean, sampleRate, 11500);
  }

  const dryReference = preClean.map((ch) => Float32Array.from(ch));
  let working = preClean.map((ch) => Float32Array.from(ch));

  // Step 2: Optional DeepFilterNet3 Neural Pre-Isolation inside Web Worker
  let usedAiBoost = false;
  if ((options.useAiBoost || options.useAiDenoise) && typeof aiDenoiseFn === 'function') {
    try {
      onProgress?.({
        percent: 18,
        stage: 'Voice Isolation (2/4): Running DeepFilterNet3 neural speech extraction...',
      });
      checkAbort();
      working = await aiDenoiseFn(working, sampleRate, {
        noiseReduction: Math.max(70, isolationStrength),
        onProgress: (p) => {
          const pct = Math.round(18 + (p.percent || 0) * 0.36);
          onProgress?.({
            percent: pct,
            stage: p.stage || 'Voice Isolation: Neural speech extraction...',
          });
        },
        shouldAbort,
      });
      usedAiBoost = true;
    } catch (aiErr) {
      if (aiErr?.name === 'AbortError') throw aiErr;
      warnings.push(
        `Neural boost unavailable (${aiErr?.message || 'WASM error'}) — completed using pure-DSP Stereo Center + Harmonic Voice Isolation.`
      );
      usedAiBoost = false;
    }
  }

  // Step 3: Stereo Center-Voice + Harmonic Speech-Presence STFT Isolation (chunked)
  const sideRatio = numChannels === 2 ? measureStereoSideRatio(working) : 0;
  const applyStereoCenter = numChannels === 2 && centerFocus > 0 && sideRatio >= 0.008;
  const stereoCenterSkipped = !applyStereoCenter;
  if (centerFocus > 0 && stereoCenterSkipped) {
    warnings.push(
      numChannels === 1
        ? 'Mono input detected — stereo center-voice extraction skipped automatically.'
        : 'Dual-mono recording detected (L ≈ R) — stereo center extraction skipped automatically.'
    );
  }

  const { bgPsd, source: noiseSource } = estimateBackgroundPsd(
    working,
    sampleRate,
    options.noiseSampleRegion
  );

  const chunkSamples = Math.max(STFT_SIZE * 4, Math.round(CHUNK_DURATION_SEC * sampleRate));
  const overlapSamples = Math.min(
    Math.round(CHUNK_OVERLAP_SEC * sampleRate),
    Math.floor(chunkSamples / 4)
  );
  const stepSamples = Math.max(1, chunkSamples - overlapSamples);
  const numChunks = Math.max(1, Math.ceil(Math.max(0, length - overlapSamples) / stepSamples));

  const assembled = Array.from({ length: numChannels }, () => new Float32Array(length));
  const basePct = usedAiBoost ? 56 : 20;
  const spanPct = usedAiBoost ? 30 : 66;
  let totalActiveFrames = 0;
  let totalProcessedFrames = 0;

  for (let c = 0; c < numChunks; c++) {
    checkAbort();
    const start = c * stepSamples;
    const end = Math.min(length, start + chunkSamples);
    const chunkIn = working.map((ch) => ch.subarray(start, end));

    onProgress?.({
      percent: Math.round(basePct + (c / numChunks) * spanPct),
      stage: `Voice Isolation (3/4): ${
        applyStereoCenter ? 'Center-voice + harmonic speech mask' : 'Harmonic speech mask'
      }${numChunks > 1 ? ` (chunk ${c + 1}/${numChunks})` : ''}...`,
    });

    const chunkRes = processVoiceIsolationChunk(chunkIn, sampleRate, {
      initialBgPsd: bgPsd,
      isolationStrengthPct: isolationStrength,
      applyStereoCenter,
      centerFocusPct: centerFocus,
      backgroundAttenuationDb,
    });

    totalActiveFrames += chunkRes.activeSpeechFrameCount;
    totalProcessedFrames += chunkRes.totalFrames;

    const len = end - start;
    const fadeInLen = c > 0 ? Math.min(overlapSamples, len) : 0;

    for (let ch = 0; ch < numChannels; ch++) {
      const dst = assembled[ch];
      const src = chunkRes.channels[ch];
      for (let i = 0; i < fadeInLen; i++) {
        const t = i / fadeInLen;
        const wNew = 0.5 * (1 - Math.cos(Math.PI * t));
        dst[start + i] = dst[start + i] * (1 - wNew) + src[i] * wNew;
      }
      for (let i = fadeInLen; i < len; i++) {
        dst[start + i] = src[i];
      }
    }
  }

  working = assembled;
  checkAbort();

  // Step 4: Natural-Voice Blend + Safety True-Peak Limiter
  if (naturalBlendFrac > 0.001) {
    onProgress?.({
      percent: 90,
      stage: `Voice Isolation (4/4): Blending ${Math.round(naturalBlendFrac * 100)}% natural voice...`,
    });
    const wetWeight = 1 - naturalBlendFrac;
    for (let ch = 0; ch < numChannels; ch++) {
      const w = working[ch];
      const d = dryReference[ch];
      for (let i = 0; i < length; i++) {
        w[i] = w[i] * wetWeight + d[i] * naturalBlendFrac;
      }
    }
  }

  // Optional loudness match if requested (e.g. Simple mode with Keep Original Loudness ON)
  const safetyCeiling =
    typeof options.limiterCeilingDbtp === 'number' ? options.limiterCeilingDbtp : -1.5;
  let limiterMaxGrDb = 0;

  if (options.keepOriginalLoudness && inputLufs > -68) {
    const curLufs = measureBS1770Loudness(working, sampleRate).integratedLufs;
    if (curLufs > -68) {
      const gainDb = Math.max(-18, Math.min(18, inputLufs - curLufs));
      const lin = dbToLinear(gainDb);
      for (let ch = 0; ch < numChannels; ch++) {
        const w = working[ch];
        for (let i = 0; i < length; i++) w[i] *= lin;
      }
    }
  }

  const limited = applySafetyTruePeakLimiter(working, sampleRate, safetyCeiling);
  working = limited.channels;
  limiterMaxGrDb = limited.maxGrDb;

  const outputLufs = measureBS1770Loudness(working, sampleRate).integratedLufs;
  const outputPeakDbtp = linearToDb(estimateTruePeak(working));
  const outputStats = inspectLevels(working, sampleRate);
  const speechActivityPct =
    totalProcessedFrames > 0
      ? Number(((totalActiveFrames / totalProcessedFrames) * 100).toFixed(1))
      : 0;

  onProgress?.({ percent: 100, stage: 'Voice Isolation complete.' });

  return {
    channels: working,
    report: {
      mode: 'voiceIsolation',
      skipped: false,
      inputLufs,
      outputLufs,
      loudnessDeltaLu: outputLufs - inputLufs,
      inputPeakDbtp,
      outputPeakDbtp,
      estimatedNoiseFloorDb: inputStats.noiseFloorDb,
      outputNoiseFloorDb: outputStats.noiseFloorDb,
      noiseFloorDropDb: inputStats.noiseFloorDb - outputStats.noiseFloorDb,
      measuredSnrDb: inputStats.snrDb,
      isolationStrength,
      centerFocus,
      backgroundAttenuationDb,
      speechActivityPct,
      stereoCenterApplied: applyStereoCenter,
      stereoCenterSkipped,
      humReport,
      noiseSource,
      usedAiBoost,
      usedAiDenoise: usedAiBoost,
      limiterMaxGrDb,
      warnings,
    },
  };
}
