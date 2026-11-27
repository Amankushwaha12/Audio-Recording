import React from 'react';
import { Play, X, SlidersHorizontal, ArrowRight } from 'lucide-react';
import { VOICE_CLEAN_PRESETS } from '../audio/voiceCleanCore.js';
import { ENABLE_AI_DENOISER } from '../audio/aiDenoiser.js';
import TransportDeck from './TransportDeck.jsx';
import Visualizations from './Visualizations.jsx';
import MetricsComparison from './MetricsComparison.jsx';

export default function SimpleWorkspace({
  voiceCleanOptions,
  onChangeVoiceCleanOption,
  onApplyVoiceCleanPreset,
  voiceIsolationOptions,
  onChangeVoiceIsolationOption,
  options,
  onChangeOption,
  onProcess,
  onCancelProcess,
  isProcessing,
  progress,
  hasAudio,
  originalBuffer,
  processedBuffer,
  originalAnalysis,
  processedAnalysis,
  processingReport,
  activeSource,
  isPlaying,
  currentTime,
  abMatchLoudness,
  onToggleAbMatchLoudness,
  onTogglePlayTrack,
  onSeek,
  onDownloadOriginalWav,
  onDownloadWav,
  theme,
  noiseRegion,
  isSelectingNoise,
  onSelectNoiseRegion,
  onToggleSelectingNoise,
  onOpenAdvanced,
}) {
  const presets = Object.values(VOICE_CLEAN_PRESETS);
  const isIsolationActive = voiceCleanOptions.preset === 'voiceIsolation';

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[380px_minmax(0,1fr)] gap-6 items-start">
      {/* Left Column: Streamlined One-Click Voice Clean Console */}
      <aside
        aria-label="Simple voice clean controls"
        className="bg-[var(--surface)] border border-[var(--border)] flex flex-col"
      >
        <div className="p-5 border-b border-[var(--border)] space-y-1.5">
          <h2 className="text-sm font-bold tracking-tight text-[var(--text)]">
            Simple Voice Clean
          </h2>
          <p className="text-xs text-[var(--text-muted)] leading-relaxed">
            Pick a cleanup or Voice Isolation profile and click Process. All DSP runs in a
            dedicated Web Worker so playback stays responsive.
          </p>
        </div>

        <div className="p-5 space-y-5">
          {/* One-click Voice Clean Profile Selector */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium text-[var(--text-secondary)]">Cleanup Profile</span>
              <span className="font-mono text-[11px] text-[var(--text-subtle)]">
                {isIsolationActive
                  ? `Isolate ${voiceIsolationOptions?.isolationStrength ?? 75}% · Center ${voiceIsolationOptions?.centerFocus ?? 70}%`
                  : voiceCleanOptions.preset === 'voiceTarget'
                    ? 'Auto SNR · -16 LUFS'
                    : `Noise ${voiceCleanOptions.noiseReduction}% · Reverb ${voiceCleanOptions.reverbReduction}%`}
              </span>
            </div>

            <div className="grid grid-cols-1 gap-2">
              {presets.map((preset) => {
                const isActive = voiceCleanOptions.preset === preset.id;
                const isVoiceTarget = preset.id === 'voiceTarget';
                const isVoiceIsolation = preset.id === 'voiceIsolation';
                return (
                  <button
                    key={preset.id}
                    type="button"
                    onClick={() => onApplyVoiceCleanPreset(preset.id)}
                    className={`w-full text-left px-3.5 py-3 border transition-colors cursor-pointer ${
                      isActive
                        ? 'border-[var(--accent)] bg-[var(--accent-muted)]'
                        : 'border-[var(--border)] bg-[var(--surface-2)] hover:border-[var(--border-hover)] hover:bg-[var(--surface-hover)]'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className={`text-xs font-semibold ${
                          isActive ? 'text-[var(--accent)]' : 'text-[var(--text)]'
                        }`}
                      >
                        {preset.label}
                      </span>
                      <span className="font-mono tabular-nums text-[11px] text-[var(--text-subtle)]">
                        {isVoiceIsolation
                          ? 'Center + Speech Mask'
                          : isVoiceTarget
                            ? 'Auto SNR / 15% dry / LRA 4–8'
                            : `${preset.noiseReduction}% / ${preset.reverbReduction}% / ${preset.naturalBlend}%`}
                      </span>
                    </div>
                    <p className="text-xs text-[var(--text-muted)] mt-1 leading-snug">
                      {preset.description}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Focused Voice Isolation controls when Voice Isolation profile is selected */}
          {isIsolationActive && voiceIsolationOptions && (
            <div className="pt-4 border-t border-[var(--border)] space-y-3">
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="simple-iso-strength" className="text-[var(--text-secondary)]">
                    Isolation strength
                  </label>
                  <span className="font-mono tabular-nums text-[var(--accent)] font-medium">
                    {voiceIsolationOptions.isolationStrength}%
                  </span>
                </div>
                <input
                  id="simple-iso-strength"
                  type="range"
                  min={20}
                  max={100}
                  step={5}
                  value={voiceIsolationOptions.isolationStrength}
                  onChange={(e) =>
                    onChangeVoiceIsolationOption?.('isolationStrength', parseFloat(e.target.value))
                  }
                  className="lab-slider"
                />
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="simple-iso-center" className="text-[var(--text-secondary)]">
                    Stereo center focus
                  </label>
                  <span className="font-mono tabular-nums text-[var(--text)] font-medium">
                    {voiceIsolationOptions.centerFocus}%
                  </span>
                </div>
                <input
                  id="simple-iso-center"
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={voiceIsolationOptions.centerFocus}
                  onChange={(e) =>
                    onChangeVoiceIsolationOption?.('centerFocus', parseFloat(e.target.value))
                  }
                  className="lab-slider"
                />
              </div>
            </div>
          )}

          {/* Essential Simple Toggles */}
          <div className="pt-4 border-t border-[var(--border)] space-y-3">
            {voiceCleanOptions.preset === 'voiceTarget' && (
              <label className="flex items-start gap-2.5 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={(options.limiterCeilingDbtp ?? -1.0) <= -1.45}
                  onChange={(e) =>
                    onChangeOption('limiterCeilingDbtp', e.target.checked ? -1.5 : -1.0)
                  }
                  className="mt-0.5 w-4 h-4 accent-[var(--accent)] cursor-pointer"
                />
                <span className="text-xs text-[var(--text-secondary)] leading-snug">
                  MP3 / AAC true-peak headroom (-1.5 dBTP)
                  <span className="block text-[11px] text-[var(--text-subtle)] mt-0.5">
                    Default is -1.0 dBTP; enable -1.5 dBTP when encoding lossy MP3 or AAC
                  </span>
                </span>
              </label>
            )}

            <label className="flex items-start gap-2.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={Boolean(voiceCleanOptions.keepOriginalLoudness)}
                onChange={(e) =>
                  onChangeVoiceCleanOption('keepOriginalLoudness', e.target.checked)
                }
                className="mt-0.5 w-4 h-4 accent-[var(--accent)] cursor-pointer"
              />
              <span className="text-xs text-[var(--text-secondary)] leading-snug">
                Keep original loudness
                <span className="block text-[11px] text-[var(--text-subtle)] mt-0.5">
                  {voiceCleanOptions.preset === 'voiceTarget' || isIsolationActive
                    ? 'Off by default (-16 LUFS broadcast polish active)'
                    : 'Matches input LUFS within ±0.5 LU with -1.5 dBTP protection (never clips)'}
                </span>
              </span>
            </label>

            <label className="flex items-start gap-2.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={Boolean(options.smoothToneEnabled)}
                onChange={(e) => onChangeOption('smoothToneEnabled', e.target.checked)}
                className="mt-0.5 w-4 h-4 accent-[var(--accent)] cursor-pointer"
              />
              <span className="text-xs text-[var(--text-secondary)] leading-snug">
                Smooth tone
                <span className="block text-[11px] text-[var(--text-subtle)] mt-0.5">
                  Softens sharp upper-mid harshness for a warmer voice
                </span>
              </span>
            </label>

            {ENABLE_AI_DENOISER && (
              <label className="flex items-start gap-2.5 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={Boolean(voiceCleanOptions.useAiDenoise)}
                  onChange={(e) => onChangeVoiceCleanOption('useAiDenoise', e.target.checked)}
                  className="mt-0.5 w-4 h-4 accent-[var(--accent)] cursor-pointer"
                />
                <span className="text-xs text-[var(--text-secondary)] leading-snug">
                  {isIsolationActive
                    ? 'AI neural boost (DeepFilterNet3)'
                    : 'AI denoise (DeepFilterNet3)'}
                  <span className="block text-[11px] text-[var(--text-subtle)] mt-0.5">
                    Lazy-loads 48 kHz neural speech denoiser inside the Web Worker
                  </span>
                </span>
              </label>
            )}
          </div>
        </div>

        {/* Primary Gold CTA + Progress / Cancel */}
        <div className="p-5 border-t border-[var(--border)] bg-[var(--surface-2)] space-y-3">
          <button
            type="button"
            onClick={onProcess}
            disabled={!hasAudio || isProcessing}
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 text-xs font-bold tracking-wider uppercase bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-[var(--accent-ink)] transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            {isProcessing
              ? isIsolationActive
                ? 'Isolating Voice...'
                : 'Cleaning Audio...'
              : isIsolationActive
                ? 'Isolate Voice'
                : 'Clean Audio'}
          </button>

          {isProcessing && (
            <div className="space-y-1.5" aria-live="polite">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="text-[var(--text-muted)] truncate">{progress.stage}</span>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="font-mono tabular-nums text-[var(--accent)] font-medium">
                    {progress.percent}%
                  </span>
                  <button
                    type="button"
                    onClick={onCancelProcess}
                    className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-medium border border-[var(--status-error)] text-[var(--status-error)] hover:bg-[var(--status-error-bg)] transition-colors cursor-pointer"
                  >
                    <X className="w-3 h-3" />
                    Cancel
                  </button>
                </div>
              </div>
              <div className="w-full h-1 bg-[var(--border)] overflow-hidden">
                <div
                  className="h-full bg-[var(--accent)] transition-all duration-150"
                  style={{ width: `${Math.max(4, progress.percent)}%` }}
                />
              </div>
            </div>
          )}

          {/* Switch to Advanced Lab link */}
          <div className="pt-3 border-t border-[var(--border)] flex items-center justify-between gap-2">
            <span className="text-[11px] text-[var(--text-subtle)] inline-flex items-center gap-1.5">
              <SlidersHorizontal className="w-3 h-3" />
              Want full EQ, compressor &amp; spectrograms?
            </span>
            <button
              type="button"
              onClick={onOpenAdvanced}
              className="inline-flex items-center gap-1 text-[11px] font-semibold text-[var(--accent)] hover:underline underline-offset-4 whitespace-nowrap cursor-pointer"
            >
              Advanced
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        </div>
      </aside>

      {/* Right Column: Matched-Loudness A/B Player, Waveforms & Before/After Report */}
      <div className="space-y-6 min-w-0">
        <TransportDeck
          originalDuration={originalBuffer?.duration || 0}
          processedDuration={processedBuffer?.duration || 0}
          activeSource={activeSource}
          isPlaying={isPlaying}
          currentTime={currentTime}
          hasOriginal={Boolean(originalBuffer)}
          hasProcessed={Boolean(processedBuffer)}
          abMatchLoudness={abMatchLoudness}
          onToggleAbMatchLoudness={onToggleAbMatchLoudness}
          originalLufs={originalAnalysis?.integratedLufs}
          processedLufs={processedAnalysis?.integratedLufs}
          onTogglePlayTrack={onTogglePlayTrack}
          onSeek={onSeek}
          onDownloadOriginalWav={onDownloadOriginalWav}
          onDownloadWav={onDownloadWav}
        />

        <Visualizations
          originalAnalysis={originalAnalysis}
          processedAnalysis={processedAnalysis}
          activeSource={activeSource}
          currentTime={currentTime}
          options={options}
          voiceCleanOptions={voiceCleanOptions}
          theme={theme}
          onSeek={onSeek}
          noiseRegion={noiseRegion}
          isSelectingNoise={isSelectingNoise}
          onSelectNoiseRegion={onSelectNoiseRegion}
          onToggleSelectingNoise={onToggleSelectingNoise}
          compact
        />

        <MetricsComparison
          originalAnalysis={originalAnalysis}
          processedAnalysis={processedAnalysis}
          options={options}
          processingReport={processingReport}
          voiceCleanOptions={voiceCleanOptions}
        />
      </div>
    </div>
  );
}
