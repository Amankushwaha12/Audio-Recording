import React from 'react';
import { SILENCE_THRESHOLD_DB } from '../audio/analyzer.js';

function fmtDb(val) {
  if (val == null || !Number.isFinite(val)) return '—';
  if (val <= -99.5) return '-∞ dB';
  return `${val.toFixed(1)}`;
}

function fmtDelta(before, after, unit = '', decimals = 1) {
  if (before == null || after == null || !Number.isFinite(before) || !Number.isFinite(after)) {
    return '—';
  }
  const diff = after - before;
  if (Math.abs(diff) < Math.pow(10, -decimals) / 2) return `0.0${unit}`;
  const sign = diff > 0 ? '+' : '';
  return `${sign}${diff.toFixed(decimals)}${unit}`;
}

function formatDuration(sec) {
  if (sec == null || !Number.isFinite(sec)) return '—';
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(2).padStart(5, '0');
  return m > 0 ? `${m}:${s}` : `${sec.toFixed(2)} s`;
}

export default function MetricsComparison({
  originalAnalysis,
  processedAnalysis,
  options,
  processingReport,
  voiceCleanOptions,
}) {
  if (!originalAnalysis) {
    return (
      <section
        id="metrics"
        className="bg-white dark:bg-[#111827] border border-slate-200 dark:border-slate-800 rounded-lg p-5"
      >
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          Analysis Panel (Before vs After)
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          Upload an audio file or load the built-in demo clip to inspect duration, sample rate,
          peak/RMS levels, ITU-R BS.1770 LUFS loudness, estimated noise floor, reverb, silence
          percentage, and clipping samples.
        </p>
      </section>
    );
  }

  const orig = originalAnalysis;
  const proc = processedAnalysis;
  const vc = processingReport?.voiceClean || null;
  const isLoudnessMatchedMode = Boolean(
    voiceCleanOptions?.enabled && voiceCleanOptions?.keepOriginalLoudness
  );

  const lufsStatus = (() => {
    if (!proc) return null;
    if (isLoudnessMatchedMode && orig.integratedLufs > -65) {
      const diff = Math.abs(proc.integratedLufs - orig.integratedLufs);
      return diff <= 0.55
        ? { text: 'Matched (±0.5 LU)', tone: 'good' }
        : { text: `Offset ${diff.toFixed(1)} LU`, tone: 'warn' };
    }
    if (options.normalizeEnabled && options.normalizeMode === 'lufs') {
      return Math.abs(proc.integratedLufs - options.targetLoudness) <= 1.2
        ? { text: 'On target', tone: 'good' }
        : { text: 'Limited', tone: 'warn' };
    }
    return null;
  })();

  const effectiveCeilingDbtp =
    voiceCleanOptions?.enabled && !options.normalizeEnabled
      ? -1.5
      : options.limiterCeilingDbtp;

  const rows = [
    {
      label: 'Duration',
      note: 'Total track length',
      before: formatDuration(orig.duration),
      after: proc ? formatDuration(proc.duration) : 'Pending',
      delta: proc ? fmtDelta(orig.duration, proc.duration, ' s', 2) : '—',
      status: null,
    },
    {
      label: 'Sample rate & channels',
      note: 'PCM format',
      before: `${orig.sampleRate.toLocaleString()} Hz · ${orig.channels === 1 ? 'Mono' : `${orig.channels} ch`}`,
      after: proc
        ? `${proc.sampleRate.toLocaleString()} Hz · ${proc.channels === 1 ? 'Mono' : `${proc.channels} ch`}`
        : 'Pending',
      delta: proc ? 'Matched' : '—',
      status: null,
    },
    {
      label: 'Integrated loudness',
      note: isLoudnessMatchedMode
        ? 'ITU-R BS.1770-4 · Voice Clean loudness match (±0.5 LU)'
        : 'ITU-R BS.1770-4 K-weighted & gated',
      before: `${fmtDb(orig.integratedLufs)} LUFS`,
      after: proc ? `${fmtDb(proc.integratedLufs)} LUFS` : 'Pending',
      delta: proc ? fmtDelta(orig.integratedLufs, proc.integratedLufs, ' LU', 1) : '—',
      status: lufsStatus,
    },
    {
      label: 'RMS level',
      note: 'Average electrical energy',
      before: `${fmtDb(orig.rmsDb)} dBFS`,
      after: proc ? `${fmtDb(proc.rmsDb)} dBFS` : 'Pending',
      delta: proc ? fmtDelta(orig.rmsDb, proc.rmsDb, ' dB', 1) : '—',
      status:
        proc && !isLoudnessMatchedMode && options.normalizeEnabled && options.normalizeMode === 'rms'
          ? Math.abs(proc.rmsDb - options.targetLoudness) <= 1.2
            ? { text: 'On target', tone: 'good' }
            : { text: 'Limited', tone: 'warn' }
          : null,
    },
    {
      label: 'Peak / True peak',
      note: 'Sample peak (4x oversampled dBTP)',
      before: `${fmtDb(orig.peakDb)} dBFS (${fmtDb(orig.truePeakDbtp)} dBTP)`,
      after: proc
        ? `${fmtDb(proc.peakDb)} dBFS (${fmtDb(proc.truePeakDbtp)} dBTP)`
        : 'Pending',
      delta: proc ? fmtDelta(orig.truePeakDbtp, proc.truePeakDbtp, ' dB', 1) : '—',
      status: proc
        ? proc.truePeakDbtp <= effectiveCeilingDbtp + 0.15
          ? { text: 'Safe', tone: 'good' }
          : { text: 'Hot peak', tone: 'warn' }
        : orig.truePeakDbtp > -0.2
          ? { text: 'Near 0 dB', tone: 'warn' }
          : null,
    },
    {
      label: 'Estimated noise floor',
      note: 'Quietest ~10% active background frames',
      before: `${fmtDb(orig.estimatedNoiseFloorDb)} dBFS`,
      after: proc ? `${fmtDb(proc.estimatedNoiseFloorDb)} dBFS` : 'Pending',
      delta: proc
        ? fmtDelta(orig.estimatedNoiseFloorDb, proc.estimatedNoiseFloorDb, ' dB', 1)
        : '—',
      status: proc
        ? proc.estimatedNoiseFloorDb <= orig.estimatedNoiseFloorDb - 2.0
          ? { text: 'Cleaner', tone: 'good' }
          : null
        : null,
    },
    ...(vc
      ? [
          {
            label: 'Estimated reverb (RT60)',
            note: vc.reverbSkipped
              ? 'Dry recording detected — reverb step auto-skipped'
              : 'Late-decay room tail estimate',
            before:
              vc.estimatedRt60Sec != null
                ? `${vc.estimatedRt60Sec.toFixed(2)} s`
                : 'Dry / N/A',
            after: vc.reverbSkipped
              ? 'Skipped (already dry)'
              : vc.outputRt60Sec != null
                ? `${vc.outputRt60Sec.toFixed(2)} s`
                : 'Dry / N/A',
            delta:
              !vc.reverbSkipped && vc.estimatedRt60Sec != null && vc.outputRt60Sec != null
                ? fmtDelta(vc.estimatedRt60Sec, vc.outputRt60Sec, ' s', 2)
                : '—',
            status: vc.reverbSkipped
              ? { text: 'Already dry', tone: 'good' }
              : vc.estimatedRt60Sec != null &&
                  vc.outputRt60Sec != null &&
                  vc.outputRt60Sec < vc.estimatedRt60Sec - 0.03
                ? { text: 'Tighter', tone: 'good' }
                : null,
          },
        ]
      : []),
    {
      label: 'Silence percentage',
      note: `50 ms frames below ${SILENCE_THRESHOLD_DB} dBFS`,
      before: `${orig.silencePercent.toFixed(1)}%`,
      after: proc ? `${proc.silencePercent.toFixed(1)}%` : 'Pending',
      delta: proc ? fmtDelta(orig.silencePercent, proc.silencePercent, '%', 1) : '—',
      status: null,
    },
    {
      label: 'Clipping detection',
      note: 'Samples at or near full scale (>= 0.999)',
      before: `${orig.clippedSamples.toLocaleString()} samples`,
      after: proc ? `${proc.clippedSamples.toLocaleString()} samples` : 'Pending',
      delta: proc
        ? `${proc.clippedSamples - orig.clippedSamples > 0 ? '+' : ''}${(
            proc.clippedSamples - orig.clippedSamples
          ).toLocaleString()}`
        : '—',
      status: proc
        ? proc.clippedSamples === 0
          ? { text: 'Clean', tone: 'good' }
          : { text: 'Clipping', tone: 'bad' }
        : orig.clippedSamples > 0
          ? { text: 'Clipped', tone: 'bad' }
          : { text: 'Clean', tone: 'good' },
    },
  ];

  const toneClass = (tone) => {
    if (tone === 'good') return 'text-emerald-700 dark:text-emerald-400';
    if (tone === 'warn') return 'text-amber-700 dark:text-amber-400';
    return 'text-rose-700 dark:text-rose-400';
  };

  return (
    <section
      id="metrics"
      aria-label="Before and after audio metrics"
      className="bg-white dark:bg-[#111827] border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden"
    >
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          Analysis Panel (Before vs After)
        </h2>
        <span className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
          Estimated input noise floor: {orig.estimatedNoiseFloorDb.toFixed(1)} dBFS
          {proc ? ` → Output floor: ${proc.estimatedNoiseFloorDb.toFixed(1)} dBFS` : ''}
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0D1320] text-xs font-medium text-slate-500 dark:text-slate-400">
              <th className="py-2.5 px-4">Metric</th>
              <th className="py-2.5 px-4">Original (Before)</th>
              <th className="py-2.5 px-4">Processed (After)</th>
              <th className="py-2.5 px-4">Delta</th>
              <th className="py-2.5 px-4 text-right">State</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200/80 dark:divide-slate-800/80 text-xs">
            {rows.map((row) => (
              <tr
                key={row.label}
                className="hover:bg-slate-50/60 dark:hover:bg-slate-800/30 transition-colors"
              >
                <td className="py-2.5 px-4">
                  <div className="font-medium text-slate-900 dark:text-slate-100">
                    {row.label}
                  </div>
                  <div className="text-[11px] text-slate-500 dark:text-slate-400">
                    {row.note}
                  </div>
                </td>
                <td className="py-2.5 px-4 font-mono tabular-nums text-slate-700 dark:text-slate-300 whitespace-nowrap">
                  {row.before}
                </td>
                <td
                  className={`py-2.5 px-4 font-mono tabular-nums whitespace-nowrap ${
                    proc
                      ? 'text-slate-900 dark:text-slate-100 font-medium'
                      : 'text-slate-400 dark:text-slate-500'
                  }`}
                >
                  {row.after}
                </td>
                <td className="py-2.5 px-4 font-mono tabular-nums text-slate-500 dark:text-slate-400 whitespace-nowrap">
                  {row.delta}
                </td>
                <td className="py-2.5 px-4 text-right font-mono whitespace-nowrap">
                  {row.status ? (
                    <span className={`font-medium ${toneClass(row.status.tone)}`}>
                      {row.status.tone === 'good'
                        ? '✓ '
                        : row.status.tone === 'warn'
                          ? '▲ '
                          : '✕ '}
                      {row.status.text}
                    </span>
                  ) : (
                    <span className="text-slate-400 dark:text-slate-600">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Voice Clean Report & Warnings Footer */}
      {vc && (
        <div className="px-4 py-3 border-t border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0D1320] space-y-2 text-xs">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-semibold text-slate-800 dark:text-slate-200">
              Voice Clean Report
            </span>
            <span className="font-mono tabular-nums text-[11px] text-slate-500 dark:text-slate-400">
              Denoiser: {vc.usedAiDenoise ? 'AI (DeepFilterNet3 WASM)' : 'Spectral STFT (2048-pt)'}{' '}
              · Profile: {vc.noiseSource === 'selected-region' ? 'Selected region' : 'Auto quietest 10%'}
            </span>
          </div>
          {vc.warnings && vc.warnings.length > 0 && (
            <ul className="space-y-1 text-amber-800 dark:text-amber-300">
              {vc.warnings.map((w, idx) => (
                <li key={idx} className="leading-snug">
                  • {w}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
