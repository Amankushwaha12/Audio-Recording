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
        className="bg-[var(--surface)] border border-[var(--border)] p-5"
      >
        <h2 className="text-sm font-bold tracking-tight text-[var(--text)]">
          Analysis Panel (Before vs After)
        </h2>
        <p className="text-xs text-[var(--text-muted)] mt-1">
          Upload an audio file, record from your microphone, or load Demo Audio to inspect
          duration, sample rate, peak/RMS levels, ITU-R BS.1770 LUFS loudness, estimated noise
          floor, reverb, silence percentage, and clipping samples.
        </p>
      </section>
    );
  }

  const orig = originalAnalysis;
  const proc = processedAnalysis;
  const vc = processingReport?.voiceClean || null;
  const vi = processingReport?.voiceIsolation || null;
  const dd = processingReport?.diffusionDereverb || null;
  const humReport = vi?.humReport || vc?.humReport || processingReport?.humReport || null;
  const adaptiveDenoiseStrength =
    vc?.adaptiveDenoiseStrength ?? processingReport?.adaptiveDenoiseStrength ?? null;
  const measuredSnrDb =
    vi?.measuredSnrDb ?? vc?.measuredSnrDb ?? processingReport?.measuredSnrDb ?? null;
  const activeLraRange = processingReport?.lraRange || options?.lraRange || null;
  const lraWarning = processingReport?.lraWarning || null;
  const isLoudnessMatchedMode = Boolean(
    voiceCleanOptions?.enabled && voiceCleanOptions?.keepOriginalLoudness
  );
  const targetToleranceLu =
    processingReport?.loudnessTolerance ?? options?.loudnessTolerance ?? 1.2;

  const lufsStatus = (() => {
    if (!proc) return null;
    if (isLoudnessMatchedMode && orig.integratedLufs > -65) {
      const diff = Math.abs(proc.integratedLufs - orig.integratedLufs);
      return diff <= 0.55
        ? { text: 'Matched (±0.5 LU)', tone: 'good' }
        : { text: `Offset ${diff.toFixed(1)} LU`, tone: 'warn' };
    }
    if (options.normalizeEnabled && options.normalizeMode === 'lufs') {
      const diff = Math.abs(proc.integratedLufs - options.targetLoudness);
      return diff <= targetToleranceLu + 0.05
        ? {
            text:
              targetToleranceLu <= 0.6
                ? `On target (±${targetToleranceLu} LU)`
                : 'On target',
            tone: 'good',
          }
        : { text: `Offset ${diff.toFixed(1)} LU`, tone: 'warn' };
    }
    return null;
  })();

  const lraStatus = (() => {
    if (!proc || !Number.isFinite(proc.lraLu) || !activeLraRange) return null;
    const [minLra, maxLra] = activeLraRange;
    if (proc.lraLu < minLra) {
      return { text: `Low (<${minLra} LU)`, tone: 'warn' };
    }
    if (proc.lraLu > maxLra) {
      return { text: `Wide (>${maxLra} LU)`, tone: 'warn' };
    }
    return { text: `In range (${minLra}–${maxLra} LU)`, tone: 'good' };
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
        : options.loudnessTolerance
          ? `ITU-R BS.1770-4 · target ${options.targetLoudness} LUFS (±${options.loudnessTolerance} LU)`
          : 'ITU-R BS.1770-4 K-weighted & gated',
      before: `${fmtDb(orig.integratedLufs)} LUFS`,
      after: proc ? `${fmtDb(proc.integratedLufs)} LUFS` : 'Pending',
      delta: proc ? fmtDelta(orig.integratedLufs, proc.integratedLufs, ' LU', 1) : '—',
      status: lufsStatus,
    },
    {
      label: 'Loudness range (LRA)',
      note: activeLraRange
        ? `EBU Tech 3342 dynamic spread · target ${activeLraRange[0]}–${activeLraRange[1]} LU`
        : 'EBU Tech 3342 short-term loudness spread (p95 − p10)',
      before: Number.isFinite(orig.lraLu) ? `${orig.lraLu.toFixed(1)} LU` : '—',
      after: proc && Number.isFinite(proc.lraLu) ? `${proc.lraLu.toFixed(1)} LU` : 'Pending',
      delta:
        proc && Number.isFinite(orig.lraLu) && Number.isFinite(proc.lraLu)
          ? fmtDelta(orig.lraLu, proc.lraLu, ' LU', 1)
          : '—',
      status: lraStatus,
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

  const toneStyle = (tone) => {
    if (tone === 'good') return { color: 'var(--status-good)' };
    if (tone === 'warn') return { color: 'var(--status-warn)' };
    return { color: 'var(--status-error)' };
  };

  return (
    <section
      id="metrics"
      aria-label="Before and after audio metrics"
      className="bg-[var(--surface)] border border-[var(--border)] overflow-hidden"
    >
      <div className="px-4 py-3 border-b border-[var(--border)] flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-bold tracking-tight text-[var(--text)]">
          Analysis Panel (Before vs After)
        </h2>
        <span className="text-xs font-mono tabular-nums text-[var(--text-subtle)]">
          Noise floor: {orig.estimatedNoiseFloorDb.toFixed(1)} dBFS
          {proc ? ` → ${proc.estimatedNoiseFloorDb.toFixed(1)} dBFS` : ''}
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-[var(--border)] bg-[var(--surface-2)] text-[11px] font-mono uppercase tracking-wider text-[var(--text-subtle)]">
              <th className="py-2.5 px-4 font-medium">Metric</th>
              <th className="py-2.5 px-4 font-medium">Original (Before)</th>
              <th className="py-2.5 px-4 font-medium">Processed (After)</th>
              <th className="py-2.5 px-4 font-medium">Delta</th>
              <th className="py-2.5 px-4 font-medium text-right">State</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border)] text-xs">
            {rows.map((row) => (
              <tr
                key={row.label}
                className="hover:bg-[var(--surface-hover)] transition-colors"
              >
                <td className="py-2.5 px-4">
                  <div className="font-medium text-[var(--text)]">{row.label}</div>
                  <div className="text-[11px] text-[var(--text-subtle)]">{row.note}</div>
                </td>
                <td className="py-2.5 px-4 font-mono tabular-nums text-[var(--text-muted)] whitespace-nowrap">
                  {row.before}
                </td>
                <td
                  className={`py-2.5 px-4 font-mono tabular-nums whitespace-nowrap ${
                    proc ? 'text-[var(--text)] font-medium' : 'text-[var(--text-subtle)]'
                  }`}
                >
                  {row.after}
                </td>
                <td className="py-2.5 px-4 font-mono tabular-nums text-[var(--text-muted)] whitespace-nowrap">
                  {row.delta}
                </td>
                <td className="py-2.5 px-4 text-right font-mono whitespace-nowrap">
                  {row.status ? (
                    <span className="font-medium" style={toneStyle(row.status.tone)}>
                      {row.status.tone === 'good'
                        ? '✓ '
                        : row.status.tone === 'warn'
                          ? '▲ '
                          : '✕ '}
                      {row.status.text}
                    </span>
                  ) : (
                    <span className="text-[var(--text-subtle)]">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Voice Clean / Voice Target / Voice Isolation / Diffusion Dereverb Diagnostics Footer */}
      {(dd || vc || vi || humReport || adaptiveDenoiseStrength != null || lraWarning) && (
        <div className="px-4 py-3 border-t border-[var(--border)] bg-[var(--surface-2)] space-y-2 text-xs">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-semibold text-[var(--text)]">
              {dd
                ? 'Diffusion Dereverb Report (Saito et al. 2023)'
                : vi
                  ? 'Voice Isolation Report'
                  : adaptiveDenoiseStrength != null
                    ? 'Voice Target Diagnostics'
                    : 'Voice Clean Report'}
            </span>
            <span className="font-mono tabular-nums text-[11px] text-[var(--text-subtle)]">
              {dd
                ? `Mode: ${dd.mode} · Device: ${dd.device || 'remote'} · ${dd.elapsed_sec ?? dd.roundTripSec} s (${
                    dd.sec_per_minute_audio != null ? `${dd.sec_per_minute_audio} s/min` : 'direct upload'
                  }) · Windows: ${dd.num_windows ?? 1}`
                : vi
                  ? `${vi.usedAiBoost ? 'DeepFilterNet3 + ' : ''}Center ${
                      vi.stereoCenterApplied ? `ON (${vi.centerFocus}%)` : 'Auto-skipped'
                    } · Speech ${vi.speechActivityPct}% · Strength ${vi.isolationStrength}%`
                  : vc
                    ? `Denoiser: ${vc.usedAiDenoise ? 'AI (DeepFilterNet3 WASM)' : 'Spectral STFT (2048-pt)'}`
                    : 'Engine: Current DSP'}
              {!dd && !vi && adaptiveDenoiseStrength != null && measuredSnrDb != null
                ? ` · SNR ${measuredSnrDb.toFixed(1)} dB → ${Math.round(adaptiveDenoiseStrength * 100)}%`
                : !dd && !vi && vc
                  ? ` · Profile: ${vc.noiseSource === 'selected-region' ? 'Selected region' : 'Auto quietest 10%'}`
                  : ''}
              {!dd && humReport
                ? humReport.detected
                  ? ` · Hum notched: ${humReport.mainsHz} Hz (${humReport.harmonicsNotched.join('/')} Hz)`
                  : ' · Hum: none detected'
                : ''}
            </span>
          </div>
          {(dd?.cpu_warning ||
            (vc?.warnings && vc.warnings.length > 0) ||
            (vi?.warnings && vi.warnings.length > 0) ||
            lraWarning) && (
            <ul className="space-y-1 text-[var(--status-warn)]">
              {dd?.cpu_warning && (
                <li key="dd-cpu" className="leading-snug">
                  • {dd.cpu_warning}
                </li>
              )}
              {vi?.warnings?.map((w, idx) => (
                <li key={`vi-${idx}`} className="leading-snug">
                  • {w}
                </li>
              ))}
              {vc?.warnings?.map((w, idx) => (
                <li key={`vc-${idx}`} className="leading-snug">
                  • {w}
                </li>
              ))}
              {lraWarning && (
                <li key="lra-warn" className="leading-snug">
                  • {lraWarning}
                </li>
              )}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
