import React from 'react';
import { SlidersHorizontal, Play, RotateCcw, Cpu, Sparkles, X } from 'lucide-react';
import { LOUDNESS_PRESETS, PRESETS } from '../audio/processor.js';
import { VOICE_CLEAN_PRESETS } from '../audio/voiceCleanCore.js';
import { ENABLE_AI_DENOISER } from '../audio/aiDenoiser.js';

function SliderField({
  id,
  label,
  value,
  displayValue,
  min,
  max,
  step,
  disabled,
  onChange,
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2 text-xs">
        <label
          htmlFor={id}
          className={disabled ? 'text-[var(--text-subtle)] opacity-50' : 'text-[var(--text-muted)]'}
        >
          {label}
        </label>
        <span
          className={`font-mono tabular-nums font-medium ${
            disabled ? 'text-[var(--text-subtle)] opacity-50' : 'text-[var(--text)]'
          }`}
        >
          {displayValue}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="lab-slider"
      />
    </div>
  );
}

function StageSection({ indexLabel, title, checked, onToggle, subtitle, children }) {
  return (
    <div className="py-3.5 border-b border-[var(--border)] last:border-b-0">
      <div className="flex items-start justify-between gap-3">
        <label className="flex items-start gap-2.5 cursor-pointer select-none min-w-0">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => onToggle(e.target.checked)}
            className="mt-0.5 w-4 h-4 accent-[var(--accent)] cursor-pointer"
          />
          <div className="min-w-0">
            <div className="text-sm font-medium text-[var(--text)] leading-snug">
              <span className="font-mono text-xs text-[var(--accent)] mr-1.5">{indexLabel}</span>
              {title}
            </div>
            {subtitle && (
              <p className="text-xs text-[var(--text-subtle)] mt-0.5">{subtitle}</p>
            )}
          </div>
        </label>
        <span className="text-[11px] font-mono text-[var(--text-subtle)] whitespace-nowrap shrink-0 mt-0.5">
          {checked ? 'Active' : 'Bypass'}
        </span>
      </div>

      <div
        className={`mt-3 pl-6.5 space-y-3 transition-opacity ${
          checked ? 'opacity-100' : 'opacity-40'
        }`}
      >
        {children}
      </div>
    </div>
  );
}

export default function ProcessingRack({
  processingEngine,
  onChangeEngine,
  dfOptions,
  onChangeDfOption,
  voiceCleanOptions,
  onChangeVoiceCleanOption,
  onApplyVoiceCleanPreset,
  noiseRegion,
  isSelectingNoise,
  onToggleSelectingNoise,
  onClearNoiseRegion,
  options,
  activePresetId,
  onApplyPreset,
  onChangeOption,
  onProcess,
  onCancelProcess,
  isProcessing,
  progress,
  hasAudio,
  processingReport,
}) {
  const isDeepFilter = processingEngine === 'deepfilternet';
  const showMasteringStages = !isDeepFilter || dfOptions.chainMastering;
  const vcEnabled = Boolean(voiceCleanOptions?.enabled);

  return (
    <aside
      aria-label="Processing chain controls"
      className="bg-[var(--surface)] border border-[var(--border)] flex flex-col"
    >
      {/* Header & Presets */}
      <div className="p-4 border-b border-[var(--border)] space-y-3.5">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-[var(--accent)]" />
            <h2 className="text-sm font-bold tracking-tight text-[var(--text)]">
              Processing Chain
            </h2>
          </div>
          <button
            type="button"
            onClick={() => onApplyPreset('podcast')}
            className="inline-flex items-center gap-1 text-xs text-[var(--text-subtle)] hover:text-[var(--text)] transition-colors cursor-pointer"
            title="Reset controls to default Podcast preset"
          >
            <RotateCcw className="w-3 h-3" />
            Reset
          </button>
        </div>

        {/* Engine Selector: Current DSP vs DeepFilterNet3 */}
        <div>
          <p className="text-[11px] font-mono uppercase tracking-wider text-[var(--text-subtle)] mb-1.5">
            Processing engine
          </p>
          <div
            role="group"
            aria-label="Processing engine"
            className="grid grid-cols-2 gap-1 p-1 bg-[var(--surface-2)] border border-[var(--border)]"
          >
            <button
              type="button"
              onClick={() => onChangeEngine('dsp')}
              className={`px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap truncate cursor-pointer ${
                !isDeepFilter
                  ? 'bg-[var(--surface)] text-[var(--text)] border border-[var(--border-hover)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)] border border-transparent'
              }`}
            >
              Current DSP
            </button>
            <button
              type="button"
              onClick={() => onChangeEngine('deepfilternet')}
              className={`px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap truncate cursor-pointer ${
                isDeepFilter
                  ? 'bg-[var(--accent-muted)] text-[var(--accent)] border border-[var(--accent-border)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)] border border-transparent'
              }`}
            >
              DeepFilterNet3
            </button>
          </div>
          <p className="text-xs text-[var(--text-subtle)] mt-1.5">
            {isDeepFilter
              ? '48 kHz neural speech denoiser (WASM) — removes noise even during speech'
              : 'Classic 6-stage deterministic Web Audio DSP mastering pipeline'}
          </p>
        </div>

        {/* DeepFilterNet3 Neural Controls (when DeepFilterNet3 engine is active) */}
        {isDeepFilter && (
          <div className="pt-3 border-t border-[var(--border)] space-y-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-[var(--text)]">
                <Cpu className="w-3.5 h-3.5 text-[var(--accent)]" />
                DeepFilterNet3 Neural Stage
              </div>
              <span className="text-[11px] font-mono text-[var(--accent)]">48 kHz WASM</span>
            </div>

            <SliderField
              id="df-atten"
              label="Max noise attenuation"
              value={dfOptions.attenuationLimit}
              displayValue={`${dfOptions.attenuationLimit} dB`}
              min={6}
              max={100}
              step={2}
              disabled={false}
              onChange={(val) => onChangeDfOption('attenuationLimit', val)}
            />

            <SliderField
              id="df-beta"
              label="Post-filter strength (beta)"
              value={dfOptions.postFilterBeta}
              displayValue={dfOptions.postFilterBeta.toFixed(3)}
              min={0}
              max={0.05}
              step={0.005}
              disabled={false}
              onChange={(val) => onChangeDfOption('postFilterBeta', val)}
            />

            <label className="flex items-start gap-2.5 pt-1 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={dfOptions.chainMastering}
                onChange={(e) => onChangeDfOption('chainMastering', e.target.checked)}
                className="mt-0.5 w-4 h-4 accent-[var(--accent)] cursor-pointer"
              />
              <span className="text-xs text-[var(--text-muted)] leading-snug">
                Follow with mastering stages (HPF, Compressor, Leveler, LUFS Norm &amp; Limiter)
              </span>
            </label>

            {processingReport?.deepFilterElapsedMs != null && (
              <p className="text-xs font-mono tabular-nums text-[var(--text-subtle)]">
                DeepFilterNet3 runtime: {(processingReport.deepFilterElapsedMs / 1000).toFixed(2)} s
              </p>
            )}
          </div>
        )}

        {/* One-click mastering presets */}
        <div className={isDeepFilter ? 'pt-3 border-t border-[var(--border)]' : ''}>
          <div className="flex items-center justify-between gap-2 mb-1.5">
            <p className="text-[11px] font-mono uppercase tracking-wider text-[var(--text-subtle)]">
              One-click presets
            </p>
            <span className="text-[11px] font-mono text-[var(--text-subtle)]">6 profiles</span>
          </div>
          <div
            role="group"
            aria-label="Mastering presets"
            className="grid grid-cols-2 gap-1 p-1 bg-[var(--surface-2)] border border-[var(--border)]"
          >
            {Object.values(PRESETS).map((preset) => {
              const active = activePresetId === preset.id;
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => onApplyPreset(preset.id)}
                  className={`px-2 py-1.5 text-xs font-medium transition-colors whitespace-nowrap truncate cursor-pointer ${
                    active
                      ? 'bg-[var(--accent-muted)] text-[var(--accent)] border border-[var(--accent-border)]'
                      : 'text-[var(--text-muted)] hover:text-[var(--text)] border border-transparent'
                  }`}
                  title={preset.summary}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
          {activePresetId && PRESETS[activePresetId] ? (
            <p className="text-xs text-[var(--text-subtle)] mt-1.5 leading-snug">
              {PRESETS[activePresetId].summary}
            </p>
          ) : (
            <p className="text-xs text-[var(--text-subtle)] mt-1.5">Custom parameters active</p>
          )}
        </div>
      </div>

      {/* Voice Clean Section (runs in a Web Worker BEFORE existing mastering stages) */}
      {voiceCleanOptions && (
        <div className="px-4 py-3.5 border-b border-[var(--border)] bg-[var(--accent-muted)] space-y-3">
          <div className="flex items-start justify-between gap-3">
            <label className="flex items-start gap-2.5 cursor-pointer select-none min-w-0">
              <input
                type="checkbox"
                checked={vcEnabled}
                onChange={(e) => onChangeVoiceCleanOption('enabled', e.target.checked)}
                className="mt-0.5 w-4 h-4 accent-[var(--accent)] cursor-pointer"
              />
              <div className="min-w-0">
                <div className="flex items-center gap-1.5 text-sm font-bold text-[var(--text)] leading-snug">
                  <Sparkles className="w-3.5 h-3.5 text-[var(--accent)] shrink-0" />
                  <span>Voice Clean</span>
                </div>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">
                  Worker DSP: noise &amp; room reverb cleanup, natural blend, de-ess &amp; loudness match
                </p>
              </div>
            </label>
            <span className="text-[11px] font-mono text-[var(--accent)] whitespace-nowrap shrink-0 mt-0.5">
              {vcEnabled ? 'Active' : 'Bypass'}
            </span>
          </div>

          <div className={`pl-6.5 space-y-3 transition-opacity ${vcEnabled ? 'opacity-100' : 'opacity-40'}`}>
            {/* Preset dropdown */}
            <div className="flex items-center justify-between gap-2 text-xs">
              <label
                htmlFor="vc-preset"
                className={vcEnabled ? 'text-[var(--text-secondary)]' : 'text-[var(--text-subtle)]'}
              >
                Preset
              </label>
              <select
                id="vc-preset"
                value={voiceCleanOptions.preset || 'custom'}
                disabled={!vcEnabled}
                onChange={(e) => onApplyVoiceCleanPreset(e.target.value)}
                className="px-2.5 py-1 text-xs border border-[var(--border-hover)] bg-[var(--surface)] text-[var(--text)] cursor-pointer disabled:opacity-50"
              >
                {Object.values(VOICE_CLEAN_PRESETS).map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
                {voiceCleanOptions.preset === 'custom' && <option value="custom">Custom</option>}
              </select>
            </div>

            <SliderField
              id="vc-noise"
              label="Noise reduction"
              value={voiceCleanOptions.noiseReduction}
              displayValue={`${voiceCleanOptions.noiseReduction}%`}
              min={0}
              max={100}
              step={1}
              disabled={!vcEnabled}
              onChange={(val) => onChangeVoiceCleanOption('noiseReduction', val)}
            />

            <SliderField
              id="vc-reverb"
              label="Reverb reduction"
              value={voiceCleanOptions.reverbReduction}
              displayValue={`${voiceCleanOptions.reverbReduction}%`}
              min={0}
              max={100}
              step={1}
              disabled={!vcEnabled}
              onChange={(val) => onChangeVoiceCleanOption('reverbReduction', val)}
            />

            <SliderField
              id="vc-blend"
              label="Keep natural voice"
              value={voiceCleanOptions.naturalBlend}
              displayValue={`${voiceCleanOptions.naturalBlend}%`}
              min={0}
              max={50}
              step={1}
              disabled={!vcEnabled}
              onChange={(val) => onChangeVoiceCleanOption('naturalBlend', val)}
            />

            {/* Keep original loudness toggle */}
            <label className="flex items-start gap-2 cursor-pointer select-none pt-0.5">
              <input
                type="checkbox"
                checked={Boolean(voiceCleanOptions.keepOriginalLoudness)}
                disabled={!vcEnabled}
                onChange={(e) => onChangeVoiceCleanOption('keepOriginalLoudness', e.target.checked)}
                className="mt-0.5 w-3.5 h-3.5 accent-[var(--accent)] cursor-pointer"
              />
              <span className="text-xs text-[var(--text-secondary)] leading-snug">
                Keep original loudness{' '}
                <span className="font-mono text-[11px] text-[var(--text-subtle)]">
                  (±0.5 LU, -1.5 dBTP)
                </span>
              </span>
            </label>

            {/* Stage 2: Optional DeepFilterNet AI denoise behind feature flag */}
            {ENABLE_AI_DENOISER && (
              <label className="flex items-start gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={Boolean(voiceCleanOptions.useAiDenoise)}
                  disabled={!vcEnabled}
                  onChange={(e) => onChangeVoiceCleanOption('useAiDenoise', e.target.checked)}
                  className="mt-0.5 w-3.5 h-3.5 accent-[var(--accent)] cursor-pointer"
                />
                <span className="text-xs text-[var(--text-secondary)] leading-snug">
                  AI denoise (DeepFilterNet)
                  <span className="block text-[11px] text-[var(--text-subtle)]">
                    Replaces spectral denoise in worker; keeps reverb, blend &amp; loudness match
                  </span>
                </span>
              </label>
            )}

            {/* User-selected noise sample region control */}
            {!voiceCleanOptions.useAiDenoise && (
              <div className="pt-1 flex flex-wrap items-center justify-between gap-2">
                <button
                  type="button"
                  disabled={!vcEnabled || !hasAudio}
                  onClick={() => onToggleSelectingNoise?.(!isSelectingNoise)}
                  className={`px-2.5 py-1 text-[11px] font-medium border transition-colors cursor-pointer disabled:opacity-40 ${
                    isSelectingNoise
                      ? 'border-[var(--accent)] bg-[var(--accent)] text-[var(--accent-ink)]'
                      : 'border-[var(--border-hover)] bg-[var(--surface)] text-[var(--text-secondary)] hover:border-[var(--accent)]'
                  }`}
                >
                  {isSelectingNoise
                    ? 'Drag noise region on waveform...'
                    : noiseRegion
                      ? `Noise sample: ${noiseRegion.startSec.toFixed(2)}–${noiseRegion.endSec.toFixed(2)}s`
                      : 'Use selected region as noise sample'}
                </button>
                {noiseRegion && (
                  <button
                    type="button"
                    onClick={onClearNoiseRegion}
                    className="text-[11px] text-[var(--text-muted)] hover:text-[var(--text)] underline cursor-pointer"
                  >
                    Auto
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Ordered DSP Stages (a through f) */}
      {showMasteringStages ? (
        <div className="px-4 py-1">
          {/* Stage a: High-pass filter */}
          <StageSection
            indexLabel="a."
            title="High-Pass Filter"
            subtitle="Removes low-frequency rumble & desk thumps"
            checked={options.highPassEnabled}
            onToggle={(val) => onChangeOption('highPassEnabled', val)}
          >
            <SliderField
              id="hp-freq"
              label="Cutoff frequency"
              value={options.highPassFreq}
              displayValue={`${options.highPassFreq} Hz`}
              min={40}
              max={200}
              step={5}
              disabled={!options.highPassEnabled}
              onChange={(val) => onChangeOption('highPassFreq', val)}
            />
          </StageSection>

          {/* Stage b: Noise reduction */}
          {isDeepFilter || vcEnabled ? (
            <div className="py-3 border-b border-[var(--border)] flex items-center justify-between gap-2 text-xs text-[var(--text-muted)]">
              <div>
                <span className="font-mono text-[var(--accent)] mr-1.5">b.</span>
                <span className="font-medium text-[var(--text-secondary)]">
                  Noise Reduction handled by {vcEnabled ? 'Voice Clean' : 'DeepFilterNet3'}
                </span>
              </div>
              <span className="font-mono text-[11px] text-[var(--accent)] shrink-0">
                {vcEnabled ? (voiceCleanOptions.useAiDenoise ? 'AI Worker' : 'Worker STFT') : 'Neural'}
              </span>
            </div>
          ) : (
            <StageSection
              indexLabel="b."
              title="Noise Reduction"
              subtitle="Spectral Wiener removes hiss during speech; Gate silences pauses"
              checked={options.noiseGateEnabled}
              onToggle={(val) => onChangeOption('noiseGateEnabled', val)}
            >
              <div className="grid grid-cols-2 gap-1 p-0.5 bg-[var(--surface-2)] border border-[var(--border)]">
                <button
                  type="button"
                  disabled={!options.noiseGateEnabled}
                  onClick={() => onChangeOption('noiseMode', 'spectral')}
                  className={`px-2 py-1 text-[11px] font-medium transition-colors whitespace-nowrap cursor-pointer disabled:opacity-40 ${
                    (options.noiseMode || 'spectral') === 'spectral'
                      ? 'bg-[var(--accent-muted)] text-[var(--accent)] border border-[var(--accent-border)]'
                      : 'text-[var(--text-muted)] hover:text-[var(--text)] border border-transparent'
                  }`}
                >
                  Spectral Wiener
                </button>
                <button
                  type="button"
                  disabled={!options.noiseGateEnabled}
                  onClick={() => onChangeOption('noiseMode', 'gate')}
                  className={`px-2 py-1 text-[11px] font-medium transition-colors whitespace-nowrap cursor-pointer disabled:opacity-40 ${
                    options.noiseMode === 'gate'
                      ? 'bg-[var(--surface)] text-[var(--text)] border border-[var(--border-hover)]'
                      : 'text-[var(--text-muted)] hover:text-[var(--text)] border border-transparent'
                  }`}
                >
                  Simple Gate
                </button>
              </div>

              <SliderField
                id="gate-reduction"
                label="Reduction strength"
                value={options.noiseGateReductionDb}
                displayValue={`-${options.noiseGateReductionDb} dB`}
                min={0}
                max={36}
                step={1}
                disabled={!options.noiseGateEnabled}
                onChange={(val) => onChangeOption('noiseGateReductionDb', val)}
              />
              {options.noiseMode === 'gate' && (
                <SliderField
                  id="gate-margin"
                  label="Gate threshold above floor"
                  value={options.noiseGateMarginDb}
                  displayValue={`+${options.noiseGateMarginDb} dB`}
                  min={2}
                  max={16}
                  step={1}
                  disabled={!options.noiseGateEnabled}
                  onChange={(val) => onChangeOption('noiseGateMarginDb', val)}
                />
              )}
              {processingReport?.estimatedNoiseFloorDb != null && (
                <p className="text-xs font-mono tabular-nums text-[var(--text-subtle)]">
                  Detected noise floor: {processingReport.estimatedNoiseFloorDb.toFixed(1)} dBFS
                </p>
              )}
            </StageSection>
          )}

          {/* Studio Voice Clarity EQ & De-Esser */}
          <StageSection
            indexLabel="+"
            title="Voice Clarity EQ & De-Esser"
            subtitle="Cuts 260 Hz mud, adds balanced clarity & softens sibilance"
            checked={Boolean(options.voiceClarityEnabled)}
            onToggle={(val) => onChangeOption('voiceClarityEnabled', val)}
          >
            <SliderField
              id="voice-clarity"
              label="Clarity & presence"
              value={Math.round((options.voiceClarityAmount ?? 0.6) * 100)}
              displayValue={`${Math.round((options.voiceClarityAmount ?? 0.6) * 100)}%`}
              min={0}
              max={100}
              step={5}
              disabled={!options.voiceClarityEnabled}
              onChange={(val) => onChangeOption('voiceClarityAmount', val / 100)}
            />
          </StageSection>

          {/* Smooth Tone */}
          <StageSection
            indexLabel="+"
            title="Smooth Tone"
            subtitle="Rounds off sharp 2.5–5 kHz harshness for a warm, smooth voice"
            checked={Boolean(options.smoothToneEnabled)}
            onToggle={(val) => onChangeOption('smoothToneEnabled', val)}
          >
            <SliderField
              id="smooth-tone"
              label="Smoothness"
              value={Math.round((options.smoothToneAmount ?? 0.6) * 100)}
              displayValue={`${Math.round((options.smoothToneAmount ?? 0.6) * 100)}%`}
              min={0}
              max={100}
              step={5}
              disabled={!options.smoothToneEnabled}
              onChange={(val) => onChangeOption('smoothToneAmount', val / 100)}
            />
          </StageSection>

          {/* Stage c: Compressor */}
          <StageSection
            indexLabel="c."
            title="Dynamics Compressor"
            subtitle="Tames loud peaks for consistent vocal presence"
            checked={options.compressorEnabled}
            onToggle={(val) => onChangeOption('compressorEnabled', val)}
          >
            <SliderField
              id="comp-threshold"
              label="Threshold"
              value={options.compThresholdDb}
              displayValue={`${options.compThresholdDb} dBFS`}
              min={-40}
              max={-6}
              step={1}
              disabled={!options.compressorEnabled}
              onChange={(val) => onChangeOption('compThresholdDb', val)}
            />
            <SliderField
              id="comp-ratio"
              label="Ratio"
              value={options.compRatio}
              displayValue={`${options.compRatio.toFixed(1)} : 1`}
              min={1.2}
              max={10}
              step={0.1}
              disabled={!options.compressorEnabled}
              onChange={(val) => onChangeOption('compRatio', val)}
            />
            <div className="grid grid-cols-2 gap-3">
              <SliderField
                id="comp-attack"
                label="Attack"
                value={options.compAttackMs}
                displayValue={`${options.compAttackMs} ms`}
                min={1}
                max={80}
                step={1}
                disabled={!options.compressorEnabled}
                onChange={(val) => onChangeOption('compAttackMs', val)}
              />
              <SliderField
                id="comp-release"
                label="Release"
                value={options.compReleaseMs}
                displayValue={`${options.compReleaseMs} ms`}
                min={30}
                max={450}
                step={10}
                disabled={!options.compressorEnabled}
                onChange={(val) => onChangeOption('compReleaseMs', val)}
              />
            </div>
            {processingReport && processingReport.compressorMaxGrDb > 0.05 && (
              <p className="text-xs font-mono tabular-nums text-[var(--text-subtle)]">
                Peak gain reduction: -{processingReport.compressorMaxGrDb.toFixed(1)} dB
              </p>
            )}
          </StageSection>

          {/* Stage d: Auto-leveler */}
          <StageSection
            indexLabel="d."
            title="Auto-Leveler"
            subtitle="0.5 s speech windows ride gain without boosting silence"
            checked={options.levelerEnabled}
            onToggle={(val) => onChangeOption('levelerEnabled', val)}
          >
            <SliderField
              id="leveler-strength"
              label="Leveling strength"
              value={Math.round(options.levelerStrength * 100)}
              displayValue={`${Math.round(options.levelerStrength * 100)}%`}
              min={20}
              max={100}
              step={5}
              disabled={!options.levelerEnabled}
              onChange={(val) => onChangeOption('levelerStrength', val / 100)}
            />
          </StageSection>

          {/* Stage e: Loudness normalization + True-peak limiter */}
          <StageSection
            indexLabel="e."
            title="Loudness Norm & Limiter"
            subtitle="Targets standard loudness then protects true-peak ceiling"
            checked={options.normalizeEnabled}
            onToggle={(val) => onChangeOption('normalizeEnabled', val)}
          >
            {vcEnabled && voiceCleanOptions.keepOriginalLoudness && (
              <p className="text-[11px] text-[var(--accent)] leading-snug">
                Voice Clean loudness match is ON (output matches input ±0.5 LU, -1.5 dBTP). Pick a
                preset below to target a fixed LUFS instead.
              </p>
            )}
            <div className="flex flex-wrap gap-1.5">
              {LOUDNESS_PRESETS.map((lp) => {
                const isSelected =
                  !(vcEnabled && voiceCleanOptions.keepOriginalLoudness) &&
                  options.normalizeMode === lp.mode &&
                  options.targetLoudness === lp.target;
                return (
                  <button
                    key={lp.id}
                    type="button"
                    disabled={!options.normalizeEnabled}
                    onClick={() => {
                      onChangeOption('normalizeMode', lp.mode);
                      onChangeOption('targetLoudness', lp.target);
                    }}
                    className={`px-2 py-1 text-[11px] font-mono border transition-colors whitespace-nowrap cursor-pointer disabled:opacity-40 ${
                      isSelected
                        ? 'border-[var(--accent)] bg-[var(--accent-muted)] text-[var(--accent)] font-medium'
                        : 'border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--border-hover)] hover:text-[var(--text)]'
                    }`}
                  >
                    {lp.label}
                  </button>
                );
              })}
            </div>

            <SliderField
              id="norm-target"
              label={
                options.normalizeMode === 'rms'
                  ? 'Target level (RMS)'
                  : 'Target loudness (LUFS)'
              }
              value={options.targetLoudness}
              displayValue={`${options.targetLoudness} ${
                options.normalizeMode === 'rms' ? 'dBFS' : 'LUFS'
              }`}
              min={-26}
              max={-12}
              step={0.5}
              disabled={!options.normalizeEnabled}
              onChange={(val) => onChangeOption('targetLoudness', val)}
            />

            <SliderField
              id="limiter-ceiling"
              label="True-peak ceiling"
              value={options.limiterCeilingDbtp}
              displayValue={`${options.limiterCeilingDbtp.toFixed(1)} dBTP`}
              min={-4.0}
              max={-0.1}
              step={0.1}
              disabled={!options.normalizeEnabled}
              onChange={(val) => onChangeOption('limiterCeilingDbtp', val)}
            />

            {processingReport && options.normalizeEnabled && processingReport.normalizeMakeupDb != null && (
              <p className="text-xs font-mono tabular-nums text-[var(--text-subtle)]">
                Makeup: {processingReport.normalizeMakeupDb >= 0 ? '+' : ''}
                {processingReport.normalizeMakeupDb.toFixed(1)} dB · Limiter max GR: -
                {processingReport.limiterMaxGrDb.toFixed(1)} dB
              </p>
            )}
          </StageSection>

          {/* Stage f: Optional silence trimming */}
          <StageSection
            indexLabel="f."
            title="Silence Trimming"
            subtitle="Shortens long pauses smoothly with cosine crossfades"
            checked={options.silenceTrimEnabled}
            onToggle={(val) => onChangeOption('silenceTrimEnabled', val)}
          >
            <SliderField
              id="silence-max"
              label="Trim pauses longer than"
              value={options.silenceMaxPauseSec}
              displayValue={`${options.silenceMaxPauseSec.toFixed(1)} s`}
              min={0.4}
              max={3.0}
              step={0.1}
              disabled={!options.silenceTrimEnabled}
              onChange={(val) => onChangeOption('silenceMaxPauseSec', val)}
            />
            {processingReport && processingReport.trimmedSeconds > 0.05 && (
              <p className="text-xs font-mono tabular-nums text-[var(--status-good)]">
                Removed {processingReport.trimmedSeconds.toFixed(2)} s of dead air
              </p>
            )}
          </StageSection>
        </div>
      ) : (
        <div className="px-4 py-4 text-xs text-[var(--text-muted)] leading-relaxed">
          Pure DeepFilterNet3 mode active — outputs raw neural speech enhancement without
          additional compression, leveling, or loudness normalization so you can compare neural
          denoising directly against the original recording.
        </div>
      )}

      {/* Anchored Process Action & Progress Indicator */}
      <div className="p-4 border-t border-[var(--border)] bg-[var(--surface-2)] space-y-3">
        <button
          type="button"
          onClick={onProcess}
          disabled={!hasAudio || isProcessing}
          className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 text-xs font-bold tracking-wider uppercase bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-[var(--accent-ink)] transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          {isProcessing
            ? 'Processing...'
            : isDeepFilter
              ? 'Process with DeepFilterNet3'
              : 'Process Audio'}
        </button>

        {isProcessing && (
          <div className="space-y-1.5" aria-live="polite">
            <div className="flex items-center justify-between gap-2 text-xs">
              <span className="text-[var(--text-muted)] truncate">{progress.stage}</span>
              <div className="flex items-center gap-2 shrink-0">
                <span className="font-mono tabular-nums text-[var(--accent)] font-medium">
                  {progress.percent}%
                </span>
                {onCancelProcess && (
                  <button
                    type="button"
                    onClick={onCancelProcess}
                    className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-medium border border-[var(--status-error)] text-[var(--status-error)] hover:bg-[var(--status-error-bg)] transition-colors cursor-pointer"
                  >
                    <X className="w-3 h-3" />
                    Cancel
                  </button>
                )}
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
      </div>
    </aside>
  );
}
