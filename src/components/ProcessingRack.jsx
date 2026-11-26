import React from 'react';
import { SlidersHorizontal, Play, RotateCcw, Cpu } from 'lucide-react';
import { LOUDNESS_PRESETS, PRESETS } from '../audio/processor.js';

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
          className={
            disabled
              ? 'text-slate-400 dark:text-slate-600'
              : 'text-slate-600 dark:text-slate-300'
          }
        >
          {label}
        </label>
        <span
          className={`font-mono tabular-nums font-medium ${
            disabled
              ? 'text-slate-400 dark:text-slate-600'
              : 'text-slate-900 dark:text-slate-100'
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
    <div className="py-3.5 border-b border-slate-200 dark:border-slate-800 last:border-b-0">
      <div className="flex items-start justify-between gap-3">
        <label className="flex items-start gap-2.5 cursor-pointer select-none min-w-0">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => onToggle(e.target.checked)}
            className="mt-0.5 w-4 h-4 rounded border-slate-300 dark:border-slate-600 text-sky-600 focus:ring-sky-500 cursor-pointer accent-sky-600 dark:accent-cyan-500"
          />
          <div className="min-w-0">
            <div className="text-sm font-medium text-slate-900 dark:text-slate-100 leading-snug">
              <span className="font-mono text-xs text-slate-400 dark:text-slate-500 mr-1.5">
                {indexLabel}
              </span>
              {title}
            </div>
            {subtitle && (
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                {subtitle}
              </p>
            )}
          </div>
        </label>
        <span className="text-xs font-mono text-slate-400 dark:text-slate-500 whitespace-nowrap shrink-0 mt-0.5">
          {checked ? 'Active' : 'Bypass'}
        </span>
      </div>

      <div
        className={`mt-3 pl-6.5 space-y-3 transition-opacity ${
          checked ? 'opacity-100' : 'opacity-45'
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
  options,
  activePresetId,
  onApplyPreset,
  onChangeOption,
  onProcess,
  isProcessing,
  progress,
  hasAudio,
  processingReport,
}) {
  const isDeepFilter = processingEngine === 'deepfilternet';
  const showMasteringStages = !isDeepFilter || dfOptions.chainMastering;

  return (
    <aside
      aria-label="Processing chain controls"
      className="bg-white dark:bg-[#111827] border border-slate-200 dark:border-slate-800 rounded-lg flex flex-col"
    >
      {/* Header & Presets */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-800 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-sky-600 dark:text-cyan-400" />
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Processing Chain
            </h2>
          </div>
          <button
            type="button"
            onClick={() => onApplyPreset('podcast')}
            className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 transition-colors cursor-pointer"
            title="Reset controls to default Podcast preset"
          >
            <RotateCcw className="w-3 h-3" />
            Reset
          </button>
        </div>

        {/* Engine Selector: Current DSP vs DeepFilterNet3 */}
        <div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-1.5">
            Processing engine
          </p>
          <div
            role="group"
            aria-label="Processing engine"
            className="grid grid-cols-2 gap-1 p-1 bg-slate-100 dark:bg-[#090D16] rounded-md border border-slate-200/80 dark:border-slate-800"
          >
            <button
              type="button"
              onClick={() => onChangeEngine('dsp')}
              className={`px-2.5 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap truncate cursor-pointer ${
                !isDeepFilter
                  ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
            >
              Current DSP
            </button>
            <button
              type="button"
              onClick={() => onChangeEngine('deepfilternet')}
              className={`px-2.5 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap truncate cursor-pointer ${
                isDeepFilter
                  ? 'bg-white dark:bg-slate-800 text-sky-700 dark:text-cyan-300 shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
            >
              DeepFilterNet3
            </button>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1.5">
            {isDeepFilter
              ? '48 kHz neural speech denoiser (WASM) — removes noise even during speech'
              : 'Classic 6-stage deterministic Web Audio DSP mastering pipeline'}
          </p>
        </div>

        {/* DeepFilterNet3 Neural Controls (when DeepFilterNet3 engine is active) */}
        {isDeepFilter && (
          <div className="pt-3 border-t border-slate-200 dark:border-slate-800 space-y-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-900 dark:text-slate-100">
                <Cpu className="w-3.5 h-3.5 text-sky-600 dark:text-cyan-400" />
                DeepFilterNet3 Neural Stage
              </div>
              <span className="text-[11px] font-mono text-sky-700 dark:text-cyan-400">
                48 kHz WASM
              </span>
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
                className="mt-0.5 w-4 h-4 rounded border-slate-300 dark:border-slate-600 accent-sky-600 dark:accent-cyan-500 cursor-pointer"
              />
              <span className="text-xs text-slate-600 dark:text-slate-300 leading-snug">
                Follow with mastering stages (HPF, Compressor, Leveler, LUFS Norm &amp; Limiter)
              </span>
            </label>

            {processingReport?.deepFilterElapsedMs != null && (
              <p className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
                DeepFilterNet3 runtime: {(processingReport.deepFilterElapsedMs / 1000).toFixed(2)} s
              </p>
            )}
          </div>
        )}

        {/* One-click mastering presets */}
        <div className={isDeepFilter ? 'pt-3 border-t border-slate-200 dark:border-slate-800' : ''}>
          <div className="flex items-center justify-between gap-2 mb-1.5">
            <p className="text-xs text-slate-500 dark:text-slate-400">
              One-click presets
            </p>
            <span className="text-[11px] font-mono text-slate-400 dark:text-slate-500">
              6 profiles
            </span>
          </div>
          <div
            role="group"
            aria-label="Mastering presets"
            className="grid grid-cols-2 gap-1 p-1 bg-slate-100 dark:bg-[#090D16] rounded-md border border-slate-200/80 dark:border-slate-800"
          >
            {Object.values(PRESETS).map((preset) => {
              const active = activePresetId === preset.id;
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => onApplyPreset(preset.id)}
                  className={`px-2 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap truncate cursor-pointer ${
                    active
                      ? 'bg-white dark:bg-slate-800 text-sky-700 dark:text-cyan-300 shadow-xs'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                  }`}
                  title={preset.summary}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
          {activePresetId && PRESETS[activePresetId] ? (
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1.5 leading-snug">
              {PRESETS[activePresetId].summary}
            </p>
          ) : (
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1.5">
              Custom parameters active
            </p>
          )}
        </div>
      </div>

      {/* Ordered DSP Stages (a through f) */}
      {showMasteringStages ? (
        <div className="px-4 py-1 divide-y-0">
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

        {/* Stage b: Noise reduction (Spectral Wiener / Gate, or replaced by DeepFilterNet3) */}
        {isDeepFilter ? (
          <div className="py-3 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between gap-2 text-xs text-slate-500 dark:text-slate-400">
            <div>
              <span className="font-mono mr-1.5">b.</span>
              <span className="font-medium text-slate-700 dark:text-slate-300">
                Noise Reduction handled by DeepFilterNet3
              </span>
            </div>
            <span className="font-mono text-sky-700 dark:text-cyan-400 shrink-0">Neural</span>
          </div>
        ) : (
          <StageSection
            indexLabel="b."
            title="Noise Reduction"
            subtitle="Spectral Wiener removes hiss during speech; Gate silences pauses"
            checked={options.noiseGateEnabled}
            onToggle={(val) => onChangeOption('noiseGateEnabled', val)}
          >
            <div className="grid grid-cols-2 gap-1 p-0.5 bg-slate-100 dark:bg-[#090D16] rounded border border-slate-200/80 dark:border-slate-800">
              <button
                type="button"
                disabled={!options.noiseGateEnabled}
                onClick={() => onChangeOption('noiseMode', 'spectral')}
                className={`px-2 py-1 text-[11px] font-medium rounded transition-colors whitespace-nowrap cursor-pointer disabled:opacity-40 ${
                  (options.noiseMode || 'spectral') === 'spectral'
                    ? 'bg-white dark:bg-slate-800 text-sky-700 dark:text-cyan-300 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                Spectral Wiener
              </button>
              <button
                type="button"
                disabled={!options.noiseGateEnabled}
                onClick={() => onChangeOption('noiseMode', 'gate')}
                className={`px-2 py-1 text-[11px] font-medium rounded transition-colors whitespace-nowrap cursor-pointer disabled:opacity-40 ${
                  options.noiseMode === 'gate'
                    ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
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
              <p className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
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

        {/* Smooth Tone: Dynamic Harshness Tamer & Warm Velvety Tone */}
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
            <p className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
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
          <div className="flex flex-wrap gap-1.5">
            {LOUDNESS_PRESETS.map((lp) => {
              const isSelected =
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
                  className={`px-2 py-1 text-[11px] font-mono rounded border transition-colors whitespace-nowrap cursor-pointer disabled:opacity-40 ${
                    isSelected
                      ? 'border-sky-500 dark:border-cyan-500 bg-sky-50 dark:bg-cyan-950/50 text-sky-800 dark:text-cyan-200 font-medium'
                      : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:border-slate-300 dark:hover:border-slate-600'
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

          {processingReport && options.normalizeEnabled && (
            <p className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
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
            <p className="text-xs font-mono tabular-nums text-emerald-700 dark:text-emerald-400">
              Removed {processingReport.trimmedSeconds.toFixed(2)} s of dead air
            </p>
          )}
        </StageSection>
      </div>
      ) : (
        <div className="px-4 py-4 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
          Pure DeepFilterNet3 mode active — outputs raw neural speech enhancement without
          additional compression, leveling, or loudness normalization so you can compare neural
          denoising directly against the original recording.
        </div>
      )}

      {/* Anchored Process Action & Progress Indicator */}
      <div className="p-4 border-t border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0D1320] rounded-b-lg space-y-3">
        <button
          type="button"
          onClick={onProcess}
          disabled={!hasAudio || isProcessing}
          className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-semibold rounded-md bg-sky-600 hover:bg-sky-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 transition-colors cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed whitespace-nowrap"
        >
          <Play className="w-4 h-4 fill-current" />
          {isProcessing
            ? 'Processing...'
            : isDeepFilter
              ? 'Process with DeepFilterNet3'
              : 'Process with Current DSP'}
        </button>

        {isProcessing && (
          <div className="space-y-1.5" aria-live="polite">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-600 dark:text-slate-300 truncate">
                {progress.stage}
              </span>
              <span className="font-mono tabular-nums text-sky-700 dark:text-cyan-400 font-medium shrink-0 ml-2">
                {progress.percent}%
              </span>
            </div>
            <div className="w-full h-1.5 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-sky-600 dark:bg-cyan-400 transition-all duration-150"
                style={{ width: `${Math.max(4, progress.percent)}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
