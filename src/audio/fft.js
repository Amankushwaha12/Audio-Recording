/**
 * fft.js — Fast Fourier Transform & Spectrogram Engine
 *
 * Plain-language overview:
 * A waveform shows amplitude over time, which hides which frequencies
 * (low rumble vs. voice formants vs. high-frequency tape hiss) are present.
 * The Short-Time Fourier Transform (STFT) slices the audio into tiny
 * overlapping windows, multiplies each window by a smooth bell curve
 * (a Hann window) to prevent edge artifacts, and runs a Radix-2
 * Cooley-Tukey FFT to measure energy at each frequency bin.
 */

/**
 * In-place iterative Radix-2 Cooley-Tukey FFT.
 * Requires `real.length === imag.length` to be a power of two (e.g., 1024).
 *
 * @param {Float32Array} real - Real component array (modified in place)
 * @param {Float32Array} imag - Imaginary component array (modified in place)
 */
export function fftRadix2(real, imag) {
  const n = real.length;
  if ((n & (n - 1)) !== 0) {
    throw new Error(`FFT size must be a power of 2, received ${n}`);
  }

  // Step 1: Bit-reversal permutation reorders samples so butterfly stages
  // can operate sequentially in memory without recursion overhead.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      const tr = real[i];
      real[i] = real[j];
      real[j] = tr;
      const ti = imag[i];
      imag[i] = imag[j];
      imag[j] = ti;
    }
  }

  // Step 2: Combine pairs of frequency bins using complex "twiddle" rotations.
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wLenReal = Math.cos(angle);
    const wLenImag = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let wReal = 1.0;
      let wImag = 0.0;
      for (let j = 0; j < halfLen; j++) {
        const uReal = real[i + j];
        const uImag = imag[i + j];
        const vReal =
          real[i + j + halfLen] * wReal - imag[i + j + halfLen] * wImag;
        const vImag =
          real[i + j + halfLen] * wImag + imag[i + j + halfLen] * wReal;

        real[i + j] = uReal + vReal;
        imag[i + j] = uImag + vImag;
        real[i + j + halfLen] = uReal - vReal;
        imag[i + j + halfLen] = uImag - vImag;

        const nextWReal = wReal * wLenReal - wImag * wLenImag;
        wImag = wReal * wLenImag + wImag * wLenReal;
        wReal = nextWReal;
      }
    }
  }
}

/**
 * In-place Inverse Radix-2 FFT computed via conjugation identity:
 * IFFT(X) = conj(FFT(conj(X))) / N
 */
export function ifftRadix2(real, imag) {
  const n = real.length;
  for (let i = 0; i < n; i++) {
    imag[i] = -imag[i];
  }
  fftRadix2(real, imag);
  const invN = 1 / n;
  for (let i = 0; i < n; i++) {
    real[i] *= invN;
    imag[i] = -imag[i] * invN;
  }
}

/**
 * Computes a compact 2D STFT spectrogram matrix for canvas rendering.
 *
 * To keep rendering snappy even on multi-minute recordings, we sample at most
 * `maxFrames` evenly spaced time slices across the audio duration, computing
 * a 1024-point Hann-windowed FFT at each slice.
 *
 * @param {AudioBuffer} audioBuffer - Source Web Audio buffer
 * @param {object} [options]
 * @param {number} [options.fftSize=1024] - FFT window size (power of 2)
 * @param {number} [options.maxFrames=320] - Horizontal time resolution
 * @param {number} [options.maxFreqHz=12000] - Upper display frequency cutoff
 * @returns {{
 *   frames: number,
 *   bins: number,
 *   data: Float32Array,
 *   minDb: number,
 *   maxDb: number,
 *   maxFreqHz: number,
 *   duration: number
 * }}
 */
export function computeSpectrogram(audioBuffer, options = {}) {
  const fftSize = options.fftSize || 1024;
  const maxFrames = options.maxFrames || 320;
  const sampleRate = audioBuffer.sampleRate;
  const nyquist = sampleRate / 2;
  const maxFreqHz = Math.min(options.maxFreqHz || 12000, nyquist);
  const length = audioBuffer.length;
  const numChannels = audioBuffer.numberOfChannels;

  // Mix channels to mono for unified spectral inspection.
  const mono = new Float32Array(length);
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = audioBuffer.getChannelData(ch);
    const scale = 1 / numChannels;
    for (let i = 0; i < length; i++) {
      mono[i] += channelData[i] * scale;
    }
  }

  // Precompute Hann window to taper frame edges smoothly to zero.
  const hann = new Float32Array(fftSize);
  let windowSum = 0;
  for (let i = 0; i < fftSize; i++) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
    hann[i] = w;
    windowSum += w;
  }
  // Coherent gain normalization keeps sine-wave magnitudes calibrated to dBFS.
  const magScale = 2 / windowSum;

  const halfBins = fftSize >> 1;
  const maxBin = Math.min(
    halfBins,
    Math.max(8, Math.ceil((maxFreqHz / nyquist) * halfBins))
  );

  const usableLength = Math.max(1, length - fftSize);
  const frames = Math.min(
    maxFrames,
    Math.max(1, Math.floor(length / (fftSize / 4)))
  );
  const hop = frames > 1 ? usableLength / (frames - 1) : 0;

  const minDb = -90;
  const maxDb = -6;
  const data = new Float32Array(frames * maxBin);
  const real = new Float32Array(fftSize);
  const imag = new Float32Array(fftSize);

  for (let f = 0; f < frames; f++) {
    const start = Math.min(
      Math.max(0, Math.floor(f * hop)),
      Math.max(0, length - 1)
    );

    for (let i = 0; i < fftSize; i++) {
      const idx = start + i;
      real[i] = idx < length ? mono[idx] * hann[i] : 0;
      imag[i] = 0;
    }

    fftRadix2(real, imag);

    const frameOffset = f * maxBin;
    for (let b = 0; b < maxBin; b++) {
      const mag = Math.hypot(real[b], imag[b]) * magScale;
      const db = 20 * Math.log10(Math.max(mag, 1e-6));
      data[frameOffset + b] = Math.max(minDb, Math.min(maxDb, db));
    }
  }

  return {
    frames,
    bins: maxBin,
    data,
    minDb,
    maxDb,
    maxFreqHz: (maxBin / halfBins) * nyquist,
    duration: audioBuffer.duration,
  };
}
