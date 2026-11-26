/**
 * analyzer.js — Audio Metering & ITU-R BS.1770-4 Loudness Analyzer
 *
 * Plain-language overview:
 * Before changing an audio file, mastering software measures it across
 * several dimensions:
 *   1. Peak (dBFS) & True Peak (dBTP): loudest instantaneous sample and
 *      reconstructed inter-sample peak (using 4x cubic interpolation).
 *   2. RMS (dBFS): average electrical energy across the entire file.
 *   3. Integrated Loudness (LUFS): human-perceived loudness per ITU-R BS.1770-4.
 *      It runs the audio through a two-stage "K-weighting" filter (a high shelf
 *      modeling head acoustics + a low cut ignoring sub-bass rumble), splits it
 *      into 400 ms overlapping blocks, and discards quiet pauses using two-stage
 *      gating (-70 LUFS absolute gate, then -10 LU relative gate).
 *   4. Silence %: share of 50 ms frames quieter than -45 dBFS.
 *   5. Clipping count: samples hitting or brushing full digital scale (>= 0.999).
 */

import { computeSpectrogram } from './fft.js';

export const SILENCE_THRESHOLD_DB = -45;
export const CLIP_SAMPLE_THRESHOLD = 0.999;

/**
 * Converts a linear amplitude ratio to decibels full scale (dBFS).
 */
export function linearToDb(value) {
  if (!Number.isFinite(value) || value <= 1e-10) return -100;
  return 20 * Math.log10(value);
}

/**
 * Converts decibels to a linear amplitude multiplier.
 */
export function dbToLinear(db) {
  return Math.pow(10, db / 20);
}

/**
 * Computes sample-rate-accurate ITU-R BS.1770-4 K-weighting biquad coefficients.
 * Uses exact bilinear-transformed analog prototypes so 44.1 kHz, 48 kHz, and
 * 96 kHz files all produce calibrated LUFS readings.
 */
export function getKWeightingCoeffs(sampleRate) {
  // Stage 1: Acoustic pre-filter (+4 dB high-shelf above ~1682 Hz)
  const f1 = 1681.974450955533;
  const g1 = 3.999843853973347;
  const q1 = 0.7071752369554196;
  const k1 = Math.tan((Math.PI * f1) / sampleRate);
  const vh = Math.pow(10, g1 / 20);
  const vb = Math.pow(vh, 0.4996667741545416);
  const a0_1 = 1 + k1 / q1 + k1 * k1;

  const pre = {
    b0: (vh + (vb * k1) / q1 + k1 * k1) / a0_1,
    b1: (2 * (k1 * k1 - vh)) / a0_1,
    b2: (vh - (vb * k1) / q1 + k1 * k1) / a0_1,
    a1: (2 * (k1 * k1 - 1)) / a0_1,
    a2: (1 - k1 / q1 + k1 * k1) / a0_1,
  };

  // Stage 2: Revised Low-Frequency B-curve (RLB) high-pass filter (~38 Hz)
  const f2 = 38.13547087602444;
  const q2 = 0.5003270373238773;
  const k2 = Math.tan((Math.PI * f2) / sampleRate);
  const a0_2 = 1 + k2 / q2 + k2 * k2;

  const rlb = {
    b0: 1 / a0_2,
    b1: -2 / a0_2,
    b2: 1 / a0_2,
    a1: (2 * (k2 * k2 - 1)) / a0_2,
    a2: (1 - k2 / q2 + k2 * k2) / a0_2,
  };

  return { pre, rlb };
}

/**
 * Applies a Direct Form II Transposed biquad filter to a single channel array.
 */
export function applyBiquad(input, coeffs) {
  const { b0, b1, b2, a1, a2 } = coeffs;
  const out = new Float32Array(input.length);
  let z1 = 0;
  let z2 = 0;
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    const y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    out[i] = y;
  }
  return out;
}

/**
 * Computes ITU-R BS.1770-4 gated integrated LUFS AND a short-term LUFS curve.
 *
 * @param {Float32Array[]} channels - Array of per-channel PCM Float32Arrays
 * @param {number} sampleRate
 * @returns {{
 *   integratedLufs: number,
 *   shortTermSeries: Array<{ time: number, lufs: number }>,
 *   shortTermWindowSec: number
 * }}
 */
export function measureBS1770Loudness(channels, sampleRate) {
  const numChannels = channels.length;
  const length = channels[0]?.length || 0;
  if (length === 0) {
    return { integratedLufs: -70, shortTermSeries: [], shortTermWindowSec: 3 };
  }

  const { pre, rlb } = getKWeightingCoeffs(sampleRate);

  // Filter each channel through K-weighting (Stage 1 -> Stage 2).
  // Channel weights: L/R/C = 1.0, surround channels (index 3, 4) = 1.41.
  const weightedSqSum = new Float64Array(length);
  for (let ch = 0; ch < numChannels; ch++) {
    const stage1 = applyBiquad(channels[ch], pre);
    const kWeighted = applyBiquad(stage1, rlb);
    const weight = ch === 3 || ch === 4 ? 1.41 : 1.0;
    for (let i = 0; i < length; i++) {
      weightedSqSum[i] += weight * kWeighted[i] * kWeighted[i];
    }
  }

  // Prefix-sum (cumulative energy) lets us compute any block's mean square in O(1).
  const prefix = new Float64Array(length + 1);
  for (let i = 0; i < length; i++) {
    prefix[i + 1] = prefix[i] + weightedSqSum[i];
  }

  // 1) Integrated LUFS: 400 ms gating blocks with 75% overlap (100 ms hop).
  const blockSamples = Math.max(1, Math.round(0.4 * sampleRate));
  const hopSamples = Math.max(1, Math.round(0.1 * sampleRate));

  const blockPowers = [];
  if (length < blockSamples) {
    const meanSq = prefix[length] / Math.max(1, length);
    blockPowers.push(meanSq);
  } else {
    for (let start = 0; start + blockSamples <= length; start += hopSamples) {
      const meanSq = (prefix[start + blockSamples] - prefix[start]) / blockSamples;
      blockPowers.push(meanSq);
    }
  }

  // Absolute gate: discard blocks quieter than -70 LUFS.
  const absThresholdPower = Math.pow(10, (-70 + 0.691) / 10);
  const aboveAbs = blockPowers.filter((p) => p > absThresholdPower);

  let integratedLufs = -70;
  if (aboveAbs.length > 0) {
    let absSum = 0;
    for (let i = 0; i < aboveAbs.length; i++) absSum += aboveAbs[i];
    const absMeanPower = absSum / aboveAbs.length;
    const ungatedLufs = -0.691 + 10 * Math.log10(absMeanPower);

    // Relative gate: discard blocks more than 10 LU quieter than ungated mean.
    const relThresholdPower = Math.pow(10, (ungatedLufs - 10 + 0.691) / 10);
    const aboveRel = aboveAbs.filter((p) => p > relThresholdPower);

    if (aboveRel.length > 0) {
      let relSum = 0;
      for (let i = 0; i < aboveRel.length; i++) relSum += aboveRel[i];
      const relMeanPower = relSum / aboveRel.length;
      integratedLufs = -0.691 + 10 * Math.log10(relMeanPower);
    } else {
      integratedLufs = ungatedLufs;
    }
  } else {
    // Fallback if entire clip is quieter than -70 LUFS
    const totalMeanSq = prefix[length] / Math.max(1, length);
    integratedLufs =
      totalMeanSq > 1e-12 ? -0.691 + 10 * Math.log10(totalMeanSq) : -70;
  }

  // 2) Short-term loudness over time: 3.0 s window (per BS.1770) for clips >= 5s,
  //    or 1.0 s window for shorter clips so the curve remains informative.
  const durationSec = length / sampleRate;
  const shortTermWindowSec = durationSec >= 5 ? 3.0 : 1.0;
  const stWinSamples = Math.max(1, Math.round(shortTermWindowSec * sampleRate));
  const stHopSamples = Math.max(1, Math.round(0.1 * sampleRate));
  const shortTermSeries = [];

  for (let pos = 0; pos <= length; pos += stHopSamples) {
    const winStart = Math.max(0, pos - stWinSamples);
    const winEnd = Math.min(length, Math.max(winStart + 1, pos));
    // For the very beginning of the file, use a forward-looking window so t=0 has data
    const effStart = pos < stWinSamples * 0.25 ? 0 : winStart;
    const effEnd =
      pos < stWinSamples * 0.25
        ? Math.min(length, Math.max(1, Math.round(stWinSamples * 0.5)))
        : winEnd;

    const meanSq =
      (prefix[effEnd] - prefix[effStart]) / Math.max(1, effEnd - effStart);
    const lufs =
      meanSq > 1e-9
        ? Math.max(-70, -0.691 + 10 * Math.log10(meanSq))
        : -70;

    shortTermSeries.push({
      time: Math.min(durationSec, pos / sampleRate),
      lufs,
    });
  }

  return {
    integratedLufs: Math.max(-70, integratedLufs),
    shortTermSeries,
    shortTermWindowSec,
  };
}

/**
 * Estimates True Peak (dBTP) using 4x Catmull-Rom cubic interpolation between samples.
 * Inter-sample peaks can exceed 0 dBFS during D/A conversion even if discrete
 * samples stay below 1.0.
 */
export function estimateTruePeak(channels) {
  let maxTruePeak = 0;
  const fractions = [0.25, 0.5, 0.75];

  for (let ch = 0; ch < channels.length; ch++) {
    const data = channels[ch];
    const len = data.length;
    for (let i = 0; i < len; i++) {
      const absVal = Math.abs(data[i]);
      if (absVal > maxTruePeak) maxTruePeak = absVal;

      // Only run cubic oversampling near high-amplitude regions for speed
      if (absVal > 0.45 && i >= 1 && i + 2 < len) {
        const y0 = data[i - 1];
        const y1 = data[i];
        const y2 = data[i + 1];
        const y3 = data[i + 2];

        for (let f = 0; f < 3; f++) {
          const t = fractions[f];
          const t2 = t * t;
          const t3 = t2 * t;
          const interp =
            0.5 *
            (2 * y1 +
              (-y0 + y2) * t +
              (2 * y0 - 5 * y1 + 4 * y2 - y3) * t2 +
              (-y0 + 3 * y1 - 3 * y2 + y3) * t3);
          const absInterp = Math.abs(interp);
          if (absInterp > maxTruePeak) maxTruePeak = absInterp;
        }
      }
    }
  }
  return maxTruePeak;
}

/**
 * Builds downsampled min/max/rms waveform bins for crisp canvas plotting.
 */
export function buildWaveformPeaks(audioBuffer, binCount = 600) {
  const length = audioBuffer.length;
  const numChannels = audioBuffer.numberOfChannels;
  const channels = [];
  for (let ch = 0; ch < numChannels; ch++) {
    channels.push(audioBuffer.getChannelData(ch));
  }

  const bins = Math.min(binCount, Math.max(1, length));
  const samplesPerBin = length / bins;
  const minArr = new Float32Array(bins);
  const maxArr = new Float32Array(bins);
  const rmsArr = new Float32Array(bins);

  for (let b = 0; b < bins; b++) {
    const start = Math.floor(b * samplesPerBin);
    const end = Math.min(length, Math.max(start + 1, Math.floor((b + 1) * samplesPerBin)));
    let min = 0;
    let max = 0;
    let sumSq = 0;
    let count = 0;

    // Stride if a bin contains many thousands of samples to keep UI instant
    const stride = Math.max(1, Math.floor((end - start) / 400));
    for (let i = start; i < end; i += stride) {
      let s = 0;
      for (let ch = 0; ch < numChannels; ch++) {
        s += channels[ch][i];
      }
      s /= numChannels;
      if (s < min) min = s;
      if (s > max) max = s;
      sumSq += s * s;
      count++;
    }

    minArr[b] = min;
    maxArr[b] = max;
    rmsArr[b] = Math.sqrt(sumSq / Math.max(1, count));
  }

  return { bins, min: minArr, max: maxArr, rms: rmsArr };
}

/**
 * Runs a complete analysis pass on a Web Audio `AudioBuffer`.
 *
 * @param {AudioBuffer} audioBuffer
 * @returns {object} Comprehensive metrics + waveform bins + spectrogram + LUFS curve
 */
export function analyzeAudioBuffer(audioBuffer) {
  const sampleRate = audioBuffer.sampleRate;
  const numChannels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;
  const duration = audioBuffer.duration;

  const channels = [];
  for (let ch = 0; ch < numChannels; ch++) {
    channels.push(audioBuffer.getChannelData(ch));
  }

  // 1. Sample Peak, RMS, and Clipping Count
  let peakLinear = 0;
  let totalSumSq = 0;
  let clippedSamples = 0;

  for (let ch = 0; ch < numChannels; ch++) {
    const data = channels[ch];
    for (let i = 0; i < length; i++) {
      const v = data[i];
      const absV = Math.abs(v);
      if (absV > peakLinear) peakLinear = absV;
      totalSumSq += v * v;
      if (absV >= CLIP_SAMPLE_THRESHOLD) {
        clippedSamples++;
      }
    }
  }

  const rmsLinear = Math.sqrt(totalSumSq / Math.max(1, length * numChannels));
  const peakDb = linearToDb(peakLinear);
  const rmsDb = linearToDb(rmsLinear);
  const truePeakLinear = estimateTruePeak(channels);
  const truePeakDbtp = linearToDb(truePeakLinear);

  // 2. Silence Percentage & Noise Floor Estimate (50 ms frames)
  const frameSamples = Math.max(1, Math.round(0.05 * sampleRate));
  const numFrames = Math.max(1, Math.ceil(length / frameSamples));
  const frameRmsDbs = new Float32Array(numFrames);
  let silentFrames = 0;

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
    const fRmsDb = linearToDb(Math.sqrt(sq / Math.max(1, cnt)));
    frameRmsDbs[f] = fRmsDb;
    if (fRmsDb < SILENCE_THRESHOLD_DB) {
      silentFrames++;
    }
  }

  const silencePercent = (silentFrames / numFrames) * 100;

  // Estimate noise floor as the 10th percentile of frame RMS levels
  const sortedFrames = Float32Array.from(frameRmsDbs).sort();
  const p10Index = Math.min(
    sortedFrames.length - 1,
    Math.floor(sortedFrames.length * 0.1)
  );
  const estimatedNoiseFloorDb = sortedFrames[p10Index];

  // 3. ITU-R BS.1770-4 Loudness (Integrated LUFS + Short-Term Series)
  const { integratedLufs, shortTermSeries, shortTermWindowSec } =
    measureBS1770Loudness(channels, sampleRate);

  // 4. Visual data (Waveform + Spectrogram)
  const waveform = buildWaveformPeaks(audioBuffer, 640);
  const spectrogram = computeSpectrogram(audioBuffer, {
    fftSize: 1024,
    maxFrames: 280,
    maxFreqHz: 12000,
  });

  return {
    duration,
    sampleRate,
    channels: numChannels,
    length,
    peakDb,
    truePeakDbtp,
    rmsDb,
    integratedLufs,
    silencePercent,
    clippedSamples,
    estimatedNoiseFloorDb,
    shortTermSeries,
    shortTermWindowSec,
    waveform,
    spectrogram,
  };
}
