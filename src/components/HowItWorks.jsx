import React from 'react';

const STAGES = [
  {
    index: 'a.',
    name: 'High-Pass Filter',
    description:
      'A 2nd-order Butterworth filter gently rolls off frequencies below the cutoff (default 80 Hz) to clear out microphone handling thumps, wind noise, and air-conditioning hum that waste headroom without affecting speech clarity.',
  },
  {
    index: 'b.',
    name: 'Spectral Wiener, Gate, or DeepFilterNet3',
    description:
      'Choose Spectral Wiener (STFT overlap-add noise suppression that works in pure JS during speech), Simple Gate (silences pauses), or DeepFilterNet3 (48 kHz WASM neural denoising with smart pre-gain and delay alignment).',
  },
  {
    index: 'c.',
    name: 'Voice Clarity EQ, De-Esser & Compressor',
    description:
      'Removes 260 Hz room mud, boosts 3.2 kHz vocal clarity with a split-band de-esser for harsh "s" sounds, then applies soft-knee compression so soft and loud syllables sit evenly.',
  },
  {
    index: 'd.',
    name: 'Speech-Aware Auto-Leveler',
    description:
      'Analyzes ~0.5 s windows and rides a smooth virtual fader toward a consistent speech level while ignoring quiet pauses so background hiss is never pumped up between sentences.',
  },
  {
    index: 'e.',
    name: 'Loudness Normalization & True-Peak Limiter',
    description:
      'Measures integrated ITU-R BS.1770-4 LUFS (or RMS for audiobooks) to hit platform loudness standards, then runs a 4x oversampled lookahead limiter to catch inter-sample peaks before they clip.',
  },
  {
    index: 'f.',
    name: 'Silence Trimming',
    description:
      'Detects dead-air spans longer than your chosen limit and trims the middle down to a natural breath pause, joining the edges with 25 ms raised-cosine crossfades to prevent clicks.',
  },
];

const LIMITATIONS = [
  'Gate-based noise reduction only attenuates background noise during pauses; it cannot separate steady hiss or street traffic that overlaps directly with spoken words the way neural speech-enhancement models do.',
  'Processing runs synchronously inside browser memory on uncompressed 32-bit float PCM buffers, so very long recordings (>50 MB compressed or multi-hour podcasts) can hit tab memory limits.',
  'The single-band dynamics compressor and simple auto-leveler do not perform multiband tonal balancing (de-essing harsh "s" sounds or adaptive EQ matching across different microphones).',
];

const EXTENSIONS = [
  {
    title: 'Local Whisper Transcription & Text-Based Editing',
    detail:
      'Run Whisper in the browser via WebGPU/WASM to generate word-level timestamps, allowing editors to delete sentences or export SRT captions right alongside the mastered WAV.',
  },
  {
    title: 'Automatic Filler Word & Breath Removal',
    detail:
      'Combine forced alignment or lightweight acoustic classifiers with the crossfade splice engine to flag and soften repeated "um/uh" fillers and sharp inhalations.',
  },
  {
    title: 'Multitrack Speech Mixing & Auto-Ducking',
    detail:
      'Accept separate host, guest, and background music stems so the music bed automatically ducks by 12–18 dB whenever a speaker track crosses the voice-activity threshold.',
  },
];

export default function HowItWorks() {
  return (
    <section
      id="how-it-works"
      aria-label="How each audio processing stage works"
      className="border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0D1320]"
    >
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-10 space-y-10">
        {/* How each processing stage works */}
        <div className="space-y-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              How It Works
            </h2>
            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1 max-w-2xl">
              Every mastering run executes six deterministic digital signal processing stages
              locally in your browser using the Web Audio API and typed PCM sample arrays.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-6 pt-2">
            {STAGES.map((stage) => (
              <div
                key={stage.index}
                className="pt-3 border-t border-slate-200 dark:border-slate-800"
              >
                <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  <span className="font-mono text-xs text-sky-600 dark:text-cyan-400 mr-1.5">
                    {stage.index}
                  </span>
                  {stage.name}
                </h3>
                <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-400 mt-1.5">
                  {stage.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Limitations & Extension Ideas */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 pt-6 border-t border-slate-200 dark:border-slate-800">
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Known Limitations
            </h3>
            <ul className="space-y-2.5 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
              {LIMITATIONS.map((item, idx) => (
                <li key={idx} className="flex items-start gap-2.5">
                  <span className="font-mono text-slate-400 dark:text-slate-500 mt-0.5">
                    0{idx + 1}.
                  </span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Three Ideas for Extending This Lab
            </h3>
            <div className="space-y-3">
              {EXTENSIONS.map((ext, idx) => (
                <div key={ext.title} className="text-xs leading-relaxed">
                  <div className="font-medium text-slate-900 dark:text-slate-100">
                    <span className="font-mono text-sky-600 dark:text-cyan-400 mr-1.5">
                      0{idx + 1}.
                    </span>
                    {ext.title}
                  </div>
                  <p className="text-slate-600 dark:text-slate-400 mt-0.5 pl-6">
                    {ext.detail}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
