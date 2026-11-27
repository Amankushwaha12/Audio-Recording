/**
 * Automated DSP verification tests for Voice Clean (`src/audio/voiceCleanCore.js`).
 *
 * Run with:
 *   node tests/voiceClean.test.mjs
 *
 * Verifies:
 *   (a) Output integrated LUFS is within 0.5 LU of input when Loudness Match is ON.
 *   (b) Zero clipping (0 samples >= 0.999 and True Peak <= -1.4 dBTP).
 *   (c) Exact duration (sample length) and sample rate preserved, original buffer untouched,
 *       and stereo inter-channel phase consistency preserved.
 *   (d) Silent input does not crash and reports a warning cleanly.
 *   (e) Clean dry speech file is not made worse (auto-skips reverb, stays clean & matched).
 *   (f) Real-world user recording (if present) and long chunked (>18 s) audio stay artifact-free.
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {
  DEFAULT_VOICE_CLEAN_OPTIONS,
  VOICE_CLEAN_PRESETS,
  runVoiceCleanPipeline,
} from '../src/audio/voiceCleanCore.js';
import {
  estimateTruePeak,
  linearToDb,
  measureBS1770Loudness,
} from '../src/audio/analyzer.js';

/** Deterministic PRNG (Mulberry32) so tests are 100% reproducible. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Synthesize speech-like harmonic formants with natural pauses. */
function makeSpeechSignal(sampleRate, durationSec, { noiseLevel = 0, reverbDecaySec = 0, seed = 42 } = {}) {
  const rand = mulberry32(seed);
  const length = Math.round(sampleRate * durationSec);
  const dry = new Float32Array(length);

  // Syllabic speech bursts separated by pauses (pause at 0.0-0.6s, bursts in between)
  for (let i = 0; i < length; i++) {
    const t = i / sampleRate;
    const inBurst =
      (t >= 0.65 && t <= 1.95) ||
      (t >= 2.55 && t <= 3.85) ||
      (t >= 4.45 && t <= Math.min(durationSec - 0.35, 5.85)) ||
      (t >= 6.5 && t <= durationSec - 0.5);

    if (inBurst) {
      const syllMod = 0.55 + 0.45 * Math.sin(2 * Math.PI * 3.8 * t);
      const f0 = 135 + 22 * Math.sin(2 * Math.PI * 1.1 * t);
      const voice =
        0.48 * Math.sin(2 * Math.PI * f0 * t) +
        0.30 * Math.sin(2 * Math.PI * f0 * 2.0 * t + 0.4) +
        0.18 * Math.sin(2 * Math.PI * f0 * 3.0 * t + 0.9) +
        0.10 * Math.sin(2 * Math.PI * 1450 * t) +
        0.05 * Math.sin(2 * Math.PI * 3100 * t);
      dry[i] = voice * syllMod * 0.32;
    }
  }

  let sig = Float32Array.from(dry);

  // Optional synthetic exponential room reverb tail
  if (reverbDecaySec > 0.05) {
    const irLen = Math.min(sampleRate * 2, Math.round(sampleRate * reverbDecaySec * 1.15));
    const rev = new Float32Array(length);
    // Tap-based diffuse reverberator (fast & realistic decay slope)
    const numTaps = 36;
    const taps = [];
    for (let k = 0; k < numTaps; k++) {
      const delaySec = 0.03 + (k / numTaps) * reverbDecaySec * 0.95 + rand() * 0.012;
      const delaySamples = Math.round(delaySec * sampleRate);
      if (delaySamples < irLen) {
        const gain = Math.pow(10, (-3 * delaySec) / reverbDecaySec) * (rand() > 0.5 ? 1 : -1) * 0.14;
        taps.push({ d: delaySamples, g: gain });
      }
    }
    for (let i = 0; i < length; i++) {
      let sum = sig[i];
      for (let k = 0; k < taps.length; k++) {
        const idx = i - taps[k].d;
        if (idx >= 0) sum += sig[idx] * taps[k].g;
      }
      rev[i] = sum;
    }
    sig = rev;
  }

  // Optional broadband stationary noise + 55 Hz hum
  if (noiseLevel > 0) {
    for (let i = 0; i < length; i++) {
      const t = i / sampleRate;
      const white = (rand() * 2 - 1) * noiseLevel;
      const hum = Math.sin(2 * Math.PI * 55 * t) * noiseLevel * 0.9;
      sig[i] += white + hum;
    }
  }

  return sig;
}

function countClipped(channels) {
  let count = 0;
  for (const ch of channels) {
    for (let i = 0; i < ch.length; i++) {
      if (!Number.isFinite(ch[i])) throw new Error(`Non-finite sample at index ${i}`);
      if (Math.abs(ch[i]) >= 0.999) count++;
    }
  }
  return count;
}

function rmsSlice(arr, start, end) {
  let sum = 0;
  const s = Math.max(0, start);
  const e = Math.min(arr.length, end);
  for (let i = s; i < e; i++) sum += arr[i] * arr[i];
  return Math.sqrt(sum / Math.max(1, e - s));
}

function correlation(a, b) {
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < a.length; i++) {
    num += a[i] * b[i];
    da += a[i] * a[i];
    db += b[i] * b[i];
  }
  return num / Math.max(1e-12, Math.sqrt(da * db));
}

function decodePcm16Wav(filePath) {
  const buf = fs.readFileSync(filePath);
  const numChannels = buf.readUInt16LE(22);
  const sampleRate = buf.readUInt32LE(24);
  const bitsPerSample = buf.readUInt16LE(34);
  let offset = 12;
  let dataOffset = -1;
  let dataSize = 0;
  while (offset + 8 <= buf.length) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    if (id === 'data') {
      dataOffset = offset + 8;
      dataSize = size;
      break;
    }
    offset += 8 + size + (size % 2);
  }
  assert(dataOffset > 0 && bitsPerSample === 16, 'Expected 16-bit PCM WAV');
  const frameCount = Math.floor(dataSize / (numChannels * 2));
  const channels = Array.from({ length: numChannels }, () => new Float32Array(frameCount));
  for (let i = 0; i < frameCount; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const s = buf.readInt16LE(dataOffset + (i * numChannels + ch) * 2);
      channels[ch][i] = s / 32768;
    }
  }
  return { channels, sampleRate };
}

async function runTests() {
  let passed = 0;
  const runCase = async (name, fn) => {
    await fn();
    passed++;
    console.log(`PASS  ${name}`);
  };

  // (a) + (b) + (c): Noisy + reverberant stereo speech at 48 kHz
  await runCase('(a,b,c) Noisy stereo 48 kHz: LUFS within 0.5 LU, 0 clipping, exact duration & non-mutated input', async () => {
    const sr = 48000;
    const mono = makeSpeechSignal(sr, 6.0, { noiseLevel: 0.025, reverbDecaySec: 0.65, seed: 11 });
    const left = Float32Array.from(mono);
    const right = Float32Array.from(mono);
    const leftSnap = Float32Array.from(left);

    const inputLufs = measureBS1770Loudness([left, right], sr).integratedLufs;
    const res = await runVoiceCleanPipeline([left, right], sr, {
      ...DEFAULT_VOICE_CLEAN_OPTIONS,
      noiseSampleRegion: { startSec: 0.05, endSec: 0.55 },
    });

    // Original buffer untouched
    assert.deepEqual(left, leftSnap, 'Input buffer must never be mutated');
    // Duration & channels preserved
    assert.equal(res.channels.length, 2);
    assert.equal(res.channels[0].length, left.length);
    assert.equal(res.channels[1].length, right.length);
    // Stereo L/R consistency (identical channels must stay identical — zero phase drift)
    assert.ok(correlation(res.channels[0], res.channels[1]) > 0.9999, 'Stereo channels must stay phase-aligned');
    // Output LUFS within 0.5 LU of input
    const outLufs = measureBS1770Loudness(res.channels, sr).integratedLufs;
    const lufsDiff = Math.abs(outLufs - inputLufs);
    assert.ok(lufsDiff <= 0.5, `Expected |outLufs - inputLufs| <= 0.5 LU, got ${lufsDiff.toFixed(2)} LU`);
    // Zero clipping & true-peak ceiling respected
    assert.equal(countClipped(res.channels), 0, 'Output must have 0 clipped samples');
    const tpDbtp = linearToDb(estimateTruePeak(res.channels));
    assert.ok(tpDbtp <= -1.35, `Expected True Peak <= -1.35 dBTP, got ${tpDbtp.toFixed(2)} dBTP`);
    // Noise floor during pause reduced meaningfully
    const inPauseRms = rmsSlice(left, Math.round(0.1 * sr), Math.round(0.5 * sr));
    const outPauseRms = rmsSlice(res.channels[0], Math.round(0.1 * sr), Math.round(0.5 * sr));
    const noiseDropDb = linearToDb(inPauseRms) - linearToDb(outPauseRms);
    assert.ok(noiseDropDb >= 6.0, `Expected >= 6 dB pause noise reduction, got ${noiseDropDb.toFixed(1)} dB`);
  });

  // (d) Silent input safety
  await runCase('(d) Silent input does not crash and emits warning', async () => {
    const sr = 44100;
    const silent = new Float32Array(sr * 2);
    const res = await runVoiceCleanPipeline([silent], sr, DEFAULT_VOICE_CLEAN_OPTIONS);
    assert.equal(res.channels.length, 1);
    assert.equal(res.channels[0].length, silent.length);
    assert.equal(countClipped(res.channels), 0);
    assert.equal(res.report.skipped, true);
    assert.ok(res.report.warnings.length > 0, 'Silent file should emit a warning');
  });

  // (e) Clean dry speech file is not made worse
  await runCase('(e) Clean dry 44.1 kHz mono speech: auto-skips reverb, stays matched & intact', async () => {
    const sr = 44100;
    const clean = makeSpeechSignal(sr, 5.0, { noiseLevel: 0, reverbDecaySec: 0, seed: 7 });
    const inLufs = measureBS1770Loudness([clean], sr).integratedLufs;
    const res = await runVoiceCleanPipeline([clean], sr, {
      ...DEFAULT_VOICE_CLEAN_OPTIONS,
      ...VOICE_CLEAN_PRESETS.clear,
      keepOriginalLoudness: true,
    });

    assert.equal(res.channels[0].length, clean.length);
    assert.equal(countClipped(res.channels), 0);
    assert.equal(res.report.reverbSkipped, true, 'Dry speech must auto-skip reverb reduction');
    const outLufs = measureBS1770Loudness(res.channels, sr).integratedLufs;
    assert.ok(
      Math.abs(outLufs - inLufs) <= 0.5,
      `Expected clean file LUFS within 0.5 LU, got ${Math.abs(outLufs - inLufs).toFixed(2)} LU`
    );
    // Clean pauses must stay clean (no watery artifacts or added noise)
    const outPauseRms = rmsSlice(res.channels[0], Math.round(0.1 * sr), Math.round(0.5 * sr));
    assert.ok(
      linearToDb(outPauseRms) < -65,
      `Expected clean pause to remain < -65 dBFS, got ${linearToDb(outPauseRms).toFixed(1)} dBFS`
    );
    // Active speech energy & waveform integrity preserved (allowing for 75 Hz IIR HPF phase shift)
    const s0 = Math.round(0.75 * sr);
    const s1 = Math.round(1.85 * sr);
    const inSpeechDb = linearToDb(rmsSlice(clean, s0, s1));
    const outSpeechDb = linearToDb(rmsSlice(res.channels[0], s0, s1));
    assert.ok(
      Math.abs(outSpeechDb - inSpeechDb) <= 1.5,
      `Expected active speech RMS within 1.5 dB, got ${(outSpeechDb - inSpeechDb).toFixed(2)} dB`
    );
    const corr = correlation(clean.subarray(s0, s1), res.channels[0].subarray(s0, s1));
    assert.ok(corr > 0.70, `Expected strong speech waveform integrity on clean file, got ${corr.toFixed(3)}`);
  });

  // (f) Chunked long audio (>18 s) seamless crossfade test
  await runCase('(f) Chunked 22 s audio preserves length, LUFS match, and zero clipping', async () => {
    const sr = 44100;
    const longSig = makeSpeechSignal(sr, 22.0, { noiseLevel: 0.015, reverbDecaySec: 0, seed: 99 });
    const inLufs = measureBS1770Loudness([longSig], sr).integratedLufs;
    const res = await runVoiceCleanPipeline([longSig], sr, DEFAULT_VOICE_CLEAN_OPTIONS);
    assert.equal(res.channels[0].length, longSig.length);
    assert.equal(countClipped(res.channels), 0);
    const outLufs = measureBS1770Loudness(res.channels, sr).integratedLufs;
    assert.ok(Math.abs(outLufs - inLufs) <= 0.5, `Chunked LUFS diff ${Math.abs(outLufs - inLufs).toFixed(2)} LU`);
  });

  // (g) Real recording integration check (if present in workspace)
  const sampleWav = path.resolve('public/assets/aistudio/orignal_bd729d02.wav');
  if (fs.existsSync(sampleWav)) {
    await runCase('(g) Real recording orignal_bd729d02.wav: LUFS <= 0.5 LU, 0 clipping, length matched', async () => {
      const { channels, sampleRate } = decodePcm16Wav(sampleWav);
      const inLufs = measureBS1770Loudness(channels, sampleRate).integratedLufs;
      const res = await runVoiceCleanPipeline(channels, sampleRate, DEFAULT_VOICE_CLEAN_OPTIONS);
      assert.equal(res.channels.length, channels.length);
      assert.equal(res.channels[0].length, channels[0].length);
      assert.equal(countClipped(res.channels), 0);
      const outLufs = measureBS1770Loudness(res.channels, sampleRate).integratedLufs;
      const diff = Math.abs(outLufs - inLufs);
      assert.ok(diff <= 0.5, `Real WAV LUFS diff ${diff.toFixed(2)} LU exceeded 0.5 LU`);
      const tp = linearToDb(estimateTruePeak(res.channels));
      assert.ok(tp <= -1.35, `Real WAV true peak ${tp.toFixed(2)} dBTP exceeded ceiling`);
    });
  }

  console.log(`\nAll ${passed} Voice Clean tests passed.`);
}

runTests().catch((err) => {
  console.error('TEST FAILED:', err);
  process.exit(1);
});
