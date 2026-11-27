import React from 'react';
import { ArrowLeft, SlidersHorizontal, Sparkles } from 'lucide-react';

const VOICE_CLEAN_STAGES = [
  {
    index: '01.',
    name: 'Pre-Clean High-Pass (75 Hz)',
    description:
      'A gentle 12 dB/oct Butterworth high-pass removes sub-bass rumble, AC hum, and mic bumps while keeping the natural chest warmth of human speech.',
  },
  {
    index: '02.',
    name: 'Spectral STFT or DeepFilterNet3',
    description:
      'Uses a 2048-point, 75%-overlap Hann STFT with stereo-linked Wiener smoothing (auto quietest-10% noise profile or your selected waveform region) and a strict gain floor (6–18 dB max attenuation) to eliminate watery artifacts. Optionally swaps in lazy-loaded 48 kHz DeepFilterNet3 WASM.',
  },
  {
    index: '03.',
    name: 'Late-Reverberation Suppression',
    description:
      'Estimates room RT60 decay slopes in the STFT domain and subtracts predicted late-reverb tails with a safe gain floor. Automatically skips and notifies you when your audio is already dry.',
  },
  {
    index: '04.',
    name: 'Natural-Voice Blend',
    description:
      'Mixes a controllable percentage (default 15%) of the time-aligned pre-cleaned dry voice back in to preserve micro-transients, breath, and human timbre.',
  },
  {
    index: '05.',
    name: 'Gentle Dynamic De-Esser (5–9 kHz)',
    description:
      'Softens sharp "s / sh / t" spikes dynamically only when sibilant energy exceeds threshold, leaving normal consonant clarity untouched.',
  },
  {
    index: '06.',
    name: 'Gentle Tone Shaping (±3 dB Cap)',
    description:
      'Applies -2.0 dB at 250 Hz to clear boxy room mud and +1.5 dB at 3.5 kHz for clean vocal presence, never exceeding ±3 dB so speech never turns harsh.',
  },
  {
    index: '07.',
    name: 'Loudness Match (±0.5 LU) & -1.5 dBTP Limiter',
    description:
      'Measures ITU-R BS.1770-4 integrated LUFS before and after cleaning so output matches your input within ±0.5 LU (unless you choose a fixed LUFS preset), then enforces a 4x oversampled true-peak ceiling at -1.5 dBTP.',
  },
];

const MASTERING_STAGES = [
  {
    index: 'a.',
    name: 'High-Pass Filter (24 dB/oct)',
    description:
      'Steeper optional rumble cutoff (40–200 Hz) for aggressive wind or outdoor low-end cleanup in Advanced mode.',
  },
  {
    index: 'b.',
    name: 'Spectral Wiener / Simple Gate',
    description:
      'Available directly when Voice Clean is bypassed, offering classic STFT noise suppression or pause gating.',
  },
  {
    index: 'c.',
    name: 'Voice Clarity EQ, Smooth Tone & Compressor',
    description:
      'Adaptive tonal tilt balancing, 2.5–5 kHz harshness taming (Smooth Tone), and stereo-linked soft-knee dynamics compression.',
  },
  {
    index: 'd.',
    name: 'Speech-Aware Auto-Leveler',
    description:
      'Rides gain smoothly across 0.5 s speech windows without pulling up silence or background noise between sentences.',
  },
  {
    index: 'e.',
    name: 'Loudness Norm & True-Peak Limiter',
    description:
      'One-click broadcast standards (Podcast -16 LUFS, YouTube -14 LUFS, Broadcast -23 LUFS, Audiobook -20 dBFS RMS) protected by a 4x oversampled lookahead limiter.',
  },
  {
    index: 'f.',
    name: 'Crossfaded Silence Trimming',
    description:
      'Shortens long dead-air pauses cleanly using 25 ms raised-cosine crossfades with zero boundary clicks.',
  },
];

const TUNING_TIPS = [
  {
    symptom: 'Voice sounds slightly robotic or watery',
    fix: 'Raise "Keep natural voice" to 20–30%, or lower "Noise reduction" by 10–20%.',
  },
  {
    symptom: 'Background hiss or fan noise is still audible',
    fix: 'Raise "Noise reduction", Shift+Drag a quiet pause on the Original waveform to set a manual noise profile, or enable "AI denoise (DeepFilterNet3)".',
  },
  {
    symptom: 'Recording still sounds echoey or roomy',
    fix: 'Raise "Reverb reduction" to 50–75% (or pick the Strong Voice Clean preset).',
  },
  {
    symptom: 'Upper mids or "S" sounds feel too sharp',
    fix: 'Enable "Smooth Tone" to tame harsh 2.5–5 kHz resonances without dulling articulation.',
  },
];

const ARCHITECTURE_NOTES = [
  'All decoding, Web Worker DSP, FFT spectrograms, IndexedDB Original/Processed storage, and 16-bit WAV encoding run 100% locally in your browser with zero server uploads or accounts.',
  'Long recordings are processed in 16-second chunks with 80 ms raised-cosine overlap crossfades so memory stays low and the UI never freezes.',
  'A/B playback compares Original (A) against Processed (B) at matched loudness by default (attenuating whichever track is louder), ensuring louder never tricks your ears into sounding better.',
  'For multi-hour files or server-side GPU acceleration, src/audio/aiDenoiser.js includes an ExternalFastApiDenoiser adapter stub for direct browser uploads to a Python FastAPI POST /clean service (avoiding Vercel serverless ~4.5 MB body limits).',
];

export default function HowItWorks({ onOpenSimple, onOpenAdvanced }) {
  return (
    <main
      id="how-it-works"
      aria-label="How Mini Audio Mastering Lab works"
      className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-10 space-y-12"
    >
      {/* Dedicated Page Header */}
      <section className="border-b border-[var(--border)] pb-8 flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div className="space-y-2.5 max-w-2xl">
          <p className="font-mono text-xs tracking-[0.2em] uppercase text-[var(--accent)]">
            Signal Architecture &amp; Guide
          </p>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-[var(--text)] [text-wrap:balance]">
            How the Local Audio DSP Pipeline Works
          </h1>
          <p className="text-sm text-[var(--text-muted)] leading-relaxed">
            Every stage runs locally in your browser on copied 32-bit float PCM buffers. Start in{' '}
            <span className="text-[var(--text)] font-medium">Simple</span> mode for one-click Voice
            Clean, or switch to{' '}
            <span className="text-[var(--text)] font-medium">Advanced</span> for full mastering
            controls, FFT spectrograms, and side-by-side IndexedDB comparison storage.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={onOpenSimple}
            className="inline-flex items-center gap-2 px-4 py-2.5 text-xs font-bold tracking-wider uppercase bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-[var(--accent-ink)] transition-colors whitespace-nowrap cursor-pointer"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Simple Mode
          </button>
          <button
            type="button"
            onClick={onOpenAdvanced}
            className="inline-flex items-center gap-2 px-4 py-2.5 text-xs font-semibold tracking-wider uppercase border border-[var(--border-hover)] bg-[var(--surface)] hover:bg-[var(--surface-hover)] text-[var(--text)] transition-colors whitespace-nowrap cursor-pointer"
          >
            <SlidersHorizontal className="w-3.5 h-3.5 text-[var(--accent)]" />
            Advanced Lab
          </button>
        </div>
      </section>

      {/* Section 01: Voice Clean Web Worker Pipeline */}
      <section className="space-y-5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 className="text-lg font-bold tracking-tight text-[var(--text)] flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-[var(--accent)]" />
              01. Voice Clean Stage (Web Worker)
            </h2>
            <p className="text-xs text-[var(--text-muted)] mt-1">
              Runs off the main thread before any dynamics or mastering stage. Never overwrites your
              original recording.
            </p>
          </div>
          <span className="font-mono text-xs text-[var(--text-subtle)]">
            7 deterministic stages · 2048-pt STFT / 48 kHz WASM
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 border-t border-l border-[var(--border)]">
          {VOICE_CLEAN_STAGES.map((stage) => (
            <div
              key={stage.index}
              className="p-5 bg-[var(--surface)] border-r border-b border-[var(--border)] space-y-2"
            >
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-xs font-semibold text-[var(--accent)]">
                  {stage.index}
                </span>
                <h3 className="text-sm font-semibold text-[var(--text)]">{stage.name}</h3>
              </div>
              <p className="text-xs leading-relaxed text-[var(--text-muted)]">
                {stage.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Section 02: Advanced Mastering Chain */}
      <section className="space-y-5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 className="text-lg font-bold tracking-tight text-[var(--text)]">
              02. Advanced Mastering Chain (Stages a–f)
            </h2>
            <p className="text-xs text-[var(--text-muted)] mt-1">
              Optional post-clean dynamics, tonal smoothing, platform loudness targets, and dead-air
              trimming.
            </p>
          </div>
          <span className="font-mono text-xs text-[var(--text-subtle)]">
            Available in Advanced mode
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 border-t border-l border-[var(--border)]">
          {MASTERING_STAGES.map((stage) => (
            <div
              key={stage.index}
              className="p-5 bg-[var(--surface)] border-r border-b border-[var(--border)] space-y-2"
            >
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-xs font-semibold text-[var(--accent)]">
                  {stage.index}
                </span>
                <h3 className="text-sm font-semibold text-[var(--text)]">{stage.name}</h3>
              </div>
              <p className="text-xs leading-relaxed text-[var(--text-muted)]">
                {stage.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Section 03: Quick Tuning Guide & Engineering Notes */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-8 pt-2">
        <div className="bg-[var(--surface)] border border-[var(--border)] p-6 space-y-4">
          <h2 className="text-base font-bold tracking-tight text-[var(--text)]">
            03. Quick Tuning Guide
          </h2>
          <div className="divide-y divide-[var(--border)]">
            {TUNING_TIPS.map((tip) => (
              <div key={tip.symptom} className="py-3.5 first:pt-0 last:pb-0 space-y-1">
                <div className="text-xs font-semibold text-[var(--text)]">{tip.symptom}</div>
                <p className="text-xs text-[var(--text-muted)] leading-relaxed">{tip.fix}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-[var(--surface)] border border-[var(--border)] p-6 space-y-4">
          <h2 className="text-base font-bold tracking-tight text-[var(--text)]">
            04. Privacy, Performance &amp; Architecture Notes
          </h2>
          <ul className="space-y-3 text-xs leading-relaxed text-[var(--text-muted)]">
            {ARCHITECTURE_NOTES.map((note, idx) => (
              <li key={idx} className="flex items-start gap-2.5">
                <span className="font-mono text-[var(--accent)] shrink-0 mt-0.5">
                  0{idx + 1}.
                </span>
                <span>{note}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>
    </main>
  );
}
