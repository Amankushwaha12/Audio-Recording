import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Download, Upload, Sparkles, SlidersHorizontal } from 'lucide-react';
import { analyzeAudioBuffer } from './audio/analyzer.js';
import {
  PRESETS,
  encodeWav16Bit,
  generateDemoAudioBuffer,
  processAudioBuffer,
} from './audio/processor.js';
import { denoiseWithDeepFilterNet } from './audio/deepfilter.js';
import {
  DEFAULT_VOICE_CLEAN_OPTIONS,
  VOICE_CLEAN_PRESETS,
} from './audio/voiceCleanCore.js';
import { runVoiceCleanInWorker } from './audio/voiceCleanClient.js';
import {
  STORE_ORIGINALS,
  STORE_PROCESSED,
  clearStore,
  deleteClipFromStore,
  listStoredClips,
  saveClipToStore,
  triggerBlobDownload,
} from './audio/storage.js';
import { useABPlayer } from './hooks/useABPlayer.js';
import { useMicRecorder } from './hooks/useMicRecorder.js';
import UploadBar from './components/UploadBar.jsx';
import SimpleWorkspace from './components/SimpleWorkspace.jsx';
import ProcessingRack from './components/ProcessingRack.jsx';
import TransportDeck from './components/TransportDeck.jsx';
import MetricsComparison from './components/MetricsComparison.jsx';
import Visualizations from './components/Visualizations.jsx';
import StorageLibrary from './components/StorageLibrary.jsx';
import HowItWorks from './components/HowItWorks.jsx';

function getInitialTheme() {
  if (typeof document !== 'undefined') {
    return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
  }
  return 'dark';
}

function getPageFromHash() {
  if (typeof window === 'undefined') return 'simple';
  const hash = window.location.hash.replace(/^#\/?/, '').toLowerCase();
  if (hash === 'advanced') return 'advanced';
  if (hash === 'how-it-works') return 'how-it-works';
  return 'simple';
}

export default function App() {
  const [activePage, setActivePage] = useState(getPageFromHash); // 'simple' | 'advanced' | 'how-it-works'
  const [theme, setTheme] = useState(getInitialTheme);
  const [fileInfo, setFileInfo] = useState(null);
  const [originalBuffer, setOriginalBuffer] = useState(null);
  const [processedBuffer, setProcessedBuffer] = useState(null);
  const [originalAnalysis, setOriginalAnalysis] = useState(null);
  const [processedAnalysis, setProcessedAnalysis] = useState(null);
  const [processingReport, setProcessingReport] = useState(null);

  const [processingEngine, setProcessingEngine] = useState('dsp'); // 'dsp' | 'deepfilternet'
  const [dfOptions, setDfOptions] = useState({
    attenuationLimit: 48,
    postFilterBeta: 0.0,
    chainMastering: true,
  });

  const [activePresetId, setActivePresetId] = useState('podcast');
  const [options, setOptions] = useState(() => ({ ...PRESETS.podcast.options }));

  // Stage 1 + optional Stage 2 Voice Clean state (runs off main thread in Web Worker)
  const [voiceCleanOptions, setVoiceCleanOptions] = useState(() => ({
    ...DEFAULT_VOICE_CLEAN_OPTIONS,
  }));
  const [abMatchLoudness, setAbMatchLoudness] = useState(true);
  const [noiseRegion, setNoiseRegion] = useState(null);
  const [isSelectingNoise, setIsSelectingNoise] = useState(false);

  const [isDecoding, setIsDecoding] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState({ percent: 0, stage: '' });
  const [errorMessage, setErrorMessage] = useState(null);

  // Separate IndexedDB storage lists for Original vs Processed comparison
  const [originalClips, setOriginalClips] = useState([]);
  const [processedClips, setProcessedClips] = useState([]);
  const [activeOriginalId, setActiveOriginalId] = useState(null);
  const [activeProcessedId, setActiveProcessedId] = useState(null);

  const fileInputRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Sync page view with URL hash so browser Back/Forward works cleanly across pages
  const navigateToPage = useCallback((page) => {
    setActivePage(page);
    if (typeof window !== 'undefined') {
      const targetHash = `#${page}`;
      if (window.location.hash !== targetHash) {
        window.history.pushState(null, '', targetHash);
      }
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, []);

  useEffect(() => {
    const onHashChange = () => setActivePage(getPageFromHash());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const {
    activeSource,
    isPlaying,
    currentTime,
    togglePlayTrack,
    switchSource,
    pause,
    seek,
    getAudioContext,
  } = useABPlayer(originalBuffer, processedBuffer, {
    enabled: abMatchLoudness,
    originalLufs: originalAnalysis?.integratedLufs,
    processedLufs: processedAnalysis?.integratedLufs,
  });

  // Load persisted Original & Processed clips from separate IndexedDB stores on mount
  const refreshStorageLists = useCallback(async () => {
    try {
      const [origList, procList] = await Promise.all([
        listStoredClips(STORE_ORIGINALS),
        listStoredClips(STORE_PROCESSED),
      ]);
      setOriginalClips(origList);
      setProcessedClips(procList);
    } catch (err) {
      console.warn('IndexedDB storage unavailable:', err);
    }
  }, []);

  useEffect(() => {
    refreshStorageLists();
  }, [refreshStorageLists]);

  const persistOriginalBuffer = useCallback(
    async (buffer, analysis, displayName) => {
      try {
        const wavBlob = encodeWav16Bit(buffer);
        const record = {
          id: `orig_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          store: STORE_ORIGINALS,
          name: displayName.endsWith('.wav') ? displayName : `${displayName}.wav`,
          createdAt: Date.now(),
          duration: analysis.duration,
          sampleRate: analysis.sampleRate,
          channels: analysis.channels,
          integratedLufs: analysis.integratedLufs,
          peakDb: analysis.peakDb,
          rmsDb: analysis.rmsDb,
          sizeBytes: wavBlob.size,
          wavBlob,
        };
        await saveClipToStore(STORE_ORIGINALS, record);
        setActiveOriginalId(record.id);
        await refreshStorageLists();
        return record;
      } catch (err) {
        console.warn('Could not save original clip to IndexedDB:', err);
        return null;
      }
    },
    [refreshStorageLists]
  );

  const persistProcessedBuffer = useCallback(
    async (buffer, analysis, sourceFileName, presetLabel) => {
      try {
        const wavBlob = encodeWav16Bit(buffer);
        const baseName = (sourceFileName || 'audio')
          .replace(/\s*\(synthetic\)$/i, '')
          .replace(/\.[^/.]+$/, '');
        const record = {
          id: `proc_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          store: STORE_PROCESSED,
          name: `${baseName}_mastered.wav`,
          sourceName: sourceFileName || 'audio',
          presetLabel,
          createdAt: Date.now(),
          duration: analysis.duration,
          sampleRate: analysis.sampleRate,
          channels: analysis.channels,
          integratedLufs: analysis.integratedLufs,
          peakDb: analysis.peakDb,
          rmsDb: analysis.rmsDb,
          sizeBytes: wavBlob.size,
          wavBlob,
        };
        await saveClipToStore(STORE_PROCESSED, record);
        setActiveProcessedId(record.id);
        await refreshStorageLists();
        return record;
      } catch (err) {
        console.warn('Could not save processed clip to IndexedDB:', err);
        return null;
      }
    },
    [refreshStorageLists]
  );

  const handleRecordingComplete = useCallback(
    (recordedBuffer) => {
      setErrorMessage(null);
      setIsDecoding(true);
      setTimeout(async () => {
        try {
          const analysis = analyzeAudioBuffer(recordedBuffer);
          const now = new Date();
          const stamp = [
            String(now.getHours()).padStart(2, '0'),
            String(now.getMinutes()).padStart(2, '0'),
            String(now.getSeconds()).padStart(2, '0'),
          ].join('');
          const recordingName = `mic_recording_${stamp}.wav`;

          setFileInfo({
            name: recordingName,
            size: recordedBuffer.length * recordedBuffer.numberOfChannels * 2 + 44,
          });
          setOriginalBuffer(recordedBuffer);
          setOriginalAnalysis(analysis);
          setProcessedBuffer(null);
          setProcessedAnalysis(null);
          setProcessingReport(null);
          setActiveProcessedId(null);
          setNoiseRegion(null);
          setIsSelectingNoise(false);
          if (analysis.peakDb < -70) {
            setErrorMessage(
              'The recorded clip appears silent (peak below -70 dBFS). Check your microphone input level.'
            );
          }

          await persistOriginalBuffer(recordedBuffer, analysis, recordingName);
        } catch (err) {
          console.error('Recorded audio analysis error:', err);
          setErrorMessage(
            'Could not analyze the recorded audio clip. Please try recording again.'
          );
        } finally {
          setIsDecoding(false);
        }
      }, 16);
    },
    [persistOriginalBuffer]
  );

  const {
    isRecording,
    elapsedSec: recordingElapsedSec,
    inputLevelDb,
    startRecording,
    stopRecording,
  } = useMicRecorder({
    getAudioContext,
    onRecordingComplete: handleRecordingComplete,
    onError: (msg) => setErrorMessage(msg),
  });

  const handleStartRecording = useCallback(() => {
    setErrorMessage(null);
    pause();
    startRecording();
  }, [pause, startRecording]);

  // Sync light/dark mode classes & data-theme attribute
  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute('data-theme', theme);
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [theme]);

  const handleToggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === 'dark' ? 'light' : 'dark';
      try {
        localStorage.setItem('maml-theme', next);
      } catch {
        // Ignore storage restrictions
      }
      return next;
    });
  }, []);

  // Keyboard shortcuts for fast A/B auditioning (Space = play/pause, A = original, B = processed)
  useEffect(() => {
    const onKeyDown = (e) => {
      const tag = e.target?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'BUTTON') {
        return;
      }
      if (!originalBuffer || isRecording || activePage === 'how-it-works') return;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlayTrack(activeSource);
      } else if (e.key === 'a' || e.key === 'A') {
        switchSource('original');
      } else if ((e.key === 'b' || e.key === 'B') && processedBuffer) {
        switchSource('processed');
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    originalBuffer,
    processedBuffer,
    activeSource,
    isRecording,
    activePage,
    togglePlayTrack,
    switchSource,
  ]);

  const handleFileSelected = useCallback(
    async (file, validationError) => {
      if (validationError) {
        setErrorMessage(validationError);
        return;
      }
      if (!file) return;

      setErrorMessage(null);
      setIsDecoding(true);
      pause();

      try {
        const arrayBuffer = await file.arrayBuffer();
        const audioCtx = getAudioContext();
        const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));

        if (!decoded || decoded.length === 0) {
          throw new Error('The audio file decoded to zero samples.');
        }

        const analysis = analyzeAudioBuffer(decoded);
        setFileInfo({ name: file.name, size: file.size });
        setOriginalBuffer(decoded);
        setOriginalAnalysis(analysis);
        setProcessedBuffer(null);
        setProcessedAnalysis(null);
        setProcessingReport(null);
        setActiveOriginalId(null);
        setActiveProcessedId(null);
        setNoiseRegion(null);
        setIsSelectingNoise(false);

        if (analysis.peakDb < -70) {
          setErrorMessage(
            `"${file.name}" appears silent (peak ${analysis.peakDb.toFixed(1)} dBFS). Voice Clean will handle it safely without crashing, though there is no speech to enhance.`
          );
        } else if (file.size > 20 * 1024 * 1024) {
          setErrorMessage(
            `Large file loaded (${(file.size / (1024 * 1024)).toFixed(1)} MB). Chunked Web Worker processing is enabled automatically to keep the interface responsive.`
          );
        }
      } catch (err) {
        console.error('Audio decode error:', err);
        setErrorMessage(
          `Could not decode "${file.name}". The file may be corrupted or use a codec unsupported by your browser. Try standard MP3, WAV, or AAC-encoded M4A.`
        );
      } finally {
        setIsDecoding(false);
      }
    },
    [getAudioContext, pause]
  );

  const handleLoadDemo = useCallback(() => {
    setErrorMessage(null);
    setIsDecoding(true);
    pause();

    setTimeout(() => {
      try {
        const audioCtx = getAudioContext();
        const demoBuffer = generateDemoAudioBuffer(audioCtx);
        const analysis = analyzeAudioBuffer(demoBuffer);
        setFileInfo({ name: 'demo_noisy_voice_clip.wav (synthetic)', size: 0 });
        setOriginalBuffer(demoBuffer);
        setOriginalAnalysis(analysis);
        setProcessedBuffer(null);
        setProcessedAnalysis(null);
        setProcessingReport(null);
        setActiveOriginalId(null);
        setActiveProcessedId(null);
        setNoiseRegion(null);
        setIsSelectingNoise(false);
      } catch (err) {
        console.error('Demo generation error:', err);
        setErrorMessage('Failed to initialize Web Audio API context for the demo clip.');
      } finally {
        setIsDecoding(false);
      }
    }, 20);
  }, [getAudioContext, pause]);

  const handleApplyPreset = useCallback((presetId) => {
    const preset = PRESETS[presetId];
    if (!preset) return;
    setActivePresetId(presetId);
    setOptions({ ...preset.options });
    if (preset.dfOptions) {
      setDfOptions({ ...preset.dfOptions });
    }
    if (preset.recommendedEngine) {
      setProcessingEngine(preset.recommendedEngine);
    }
    if (preset.options.normalizeEnabled) {
      setVoiceCleanOptions((prev) => ({ ...prev, keepOriginalLoudness: false }));
    }
  }, []);

  const handleChangeOption = useCallback((key, value) => {
    setActivePresetId(null);
    setOptions((prev) => ({ ...prev, [key]: value }));
    if (key === 'targetLoudness' || key === 'normalizeMode') {
      setVoiceCleanOptions((prev) => ({ ...prev, keepOriginalLoudness: false }));
    }
  }, []);

  const handleChangeDfOption = useCallback((key, value) => {
    setDfOptions((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleApplyVoiceCleanPreset = useCallback((presetId) => {
    const p = VOICE_CLEAN_PRESETS[presetId];
    if (!p) return;
    setVoiceCleanOptions((prev) => ({
      ...prev,
      enabled: true,
      preset: p.id,
      noiseReduction: p.noiseReduction,
      reverbReduction: p.reverbReduction,
      naturalBlend: p.naturalBlend,
    }));
  }, []);

  const handleChangeVoiceCleanOption = useCallback((key, value) => {
    setVoiceCleanOptions((prev) => {
      const next = { ...prev, [key]: value };
      if (key === 'noiseReduction' || key === 'reverbReduction' || key === 'naturalBlend') {
        next.preset = 'custom';
      }
      return next;
    });
  }, []);

  const handleCancelProcess = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  const handleProcess = useCallback(
    async (isSimpleMode = false) => {
      if (!originalBuffer || isProcessing) return;
      setErrorMessage(null);
      setIsProcessing(true);
      setIsSelectingNoise(false);

      const controller = new AbortController();
      abortControllerRef.current = controller;
      const { signal } = controller;

      const vcActive = isSimpleMode ? true : Boolean(voiceCleanOptions.enabled);
      const keepInputLufs = Boolean(vcActive && voiceCleanOptions.keepOriginalLoudness);

      setProgress({
        percent: 3,
        stage: vcActive
          ? 'Starting Voice Clean Web Worker...'
          : processingEngine === 'deepfilternet'
            ? 'Initializing DeepFilterNet3 neural engine...'
            : 'Preparing PCM channel buffers...',
      });
      pause();

      try {
        let workingBuffer = originalBuffer;
        let voiceCleanReport = null;
        let report = {};
        let storageTag = '';

        // STEP 0: Stage 1 (and optional Stage 2 AI denoise) Voice Clean in Web Worker
        if (vcActive) {
          const vcRes = await runVoiceCleanInWorker(
            originalBuffer,
            {
              ...voiceCleanOptions,
              enabled: true,
              noiseSampleRegion: noiseRegion,
            },
            {
              signal,
              onProgress: (prog) => {
                const scale = isSimpleMode
                  ? options.smoothToneEnabled
                    ? 0.82
                    : 0.96
                  : processingEngine === 'deepfilternet'
                    ? 0.55
                    : 0.68;
                setProgress({
                  percent: Math.max(4, Math.round(prog.percent * scale)),
                  stage: prog.stage,
                });
              },
            }
          );

          if (signal.aborted) {
            const err = new Error('Processing cancelled.');
            err.name = 'AbortError';
            throw err;
          }

          workingBuffer = vcRes.audioBuffer;
          voiceCleanReport = vcRes.report;
        }

        let outBuffer = workingBuffer;

        if (isSimpleMode) {
          // Simple mode: clean Voice Clean + optional Smooth Tone + loudness match / -1.5 dBTP protection
          if (options.smoothToneEnabled) {
            const simplePost = await processAudioBuffer(
              workingBuffer,
              {
                ...options,
                highPassEnabled: false,
                noiseGateEnabled: false,
                voiceClarityEnabled: false,
                smoothToneEnabled: true,
                compressorEnabled: false,
                levelerEnabled: false,
                silenceTrimEnabled: false,
                normalizeEnabled: true,
                normalizeMode: 'lufs',
                targetLoudness:
                  keepInputLufs && originalAnalysis?.integratedLufs > -65
                    ? originalAnalysis.integratedLufs
                    : options.targetLoudness,
                limiterCeilingDbtp: -1.5,
              },
              (prog) => {
                setProgress({
                  percent: Math.round(82 + prog.percent * 0.16),
                  stage: prog.stage,
                });
              }
            );
            outBuffer = simplePost.audioBuffer;
            report = simplePost.report;
          }
          storageTag = `Simple · Voice Clean (${voiceCleanOptions.useAiDenoise ? 'AI' : voiceCleanOptions.preset})`;
        } else {
          // Advanced mode: full configurable mastering rack after Voice Clean
          const effectiveOptions = vcActive
            ? {
                ...options,
                noiseGateEnabled: false,
                highPassEnabled: options.highPassEnabled && options.highPassFreq > 85,
                ...(keepInputLufs && originalAnalysis?.integratedLufs > -65
                  ? {
                      normalizeEnabled: true,
                      normalizeMode: 'lufs',
                      targetLoudness: originalAnalysis.integratedLufs,
                      limiterCeilingDbtp: Math.min(options.limiterCeilingDbtp, -1.5),
                    }
                  : {}),
              }
            : options;

          const hasPostStages =
            effectiveOptions.highPassEnabled ||
            effectiveOptions.noiseGateEnabled ||
            effectiveOptions.voiceClarityEnabled ||
            effectiveOptions.smoothToneEnabled ||
            effectiveOptions.compressorEnabled ||
            effectiveOptions.levelerEnabled ||
            (effectiveOptions.normalizeEnabled && !keepInputLufs) ||
            effectiveOptions.silenceTrimEnabled;

          if (processingEngine === 'deepfilternet' && !(vcActive && voiceCleanOptions.useAiDenoise)) {
            try {
              const dfRes = await denoiseWithDeepFilterNet(workingBuffer, dfOptions, (prog) => {
                const basePct = vcActive ? 55 : 0;
                const span = dfOptions.chainMastering
                  ? vcActive
                    ? 25
                    : 65
                  : vcActive
                    ? 41
                    : 96;
                setProgress({
                  percent: Math.min(96, Math.round(basePct + (prog.percent / 100) * span)),
                  stage: prog.stage,
                });
              });

              if (signal.aborted) {
                const err = new Error('Processing cancelled.');
                err.name = 'AbortError';
                throw err;
              }

              if (dfOptions.chainMastering) {
                const postRes = await processAudioBuffer(
                  dfRes.audioBuffer,
                  { ...effectiveOptions, noiseGateEnabled: false, postNeuralCleanup: true },
                  (prog) => {
                    const basePct = vcActive ? 80 : 65;
                    const span = vcActive ? 18 : 33;
                    setProgress({
                      percent: Math.round(basePct + (prog.percent / 100) * span),
                      stage: prog.stage,
                    });
                  }
                );
                outBuffer = postRes.audioBuffer;
                report = { ...postRes.report, deepFilterElapsedMs: dfRes.elapsedMs };
                const presetPart =
                  activePresetId && PRESETS[activePresetId]
                    ? PRESETS[activePresetId].label
                    : `${options.targetLoudness} ${options.normalizeMode === 'rms' ? 'RMS' : 'LUFS'}`;
                storageTag = vcActive
                  ? `Voice Clean + DeepFilterNet3 + ${presetPart}`
                  : `DeepFilterNet3 + ${presetPart}`;
              } else {
                outBuffer = dfRes.audioBuffer;
                report = { deepFilterElapsedMs: dfRes.elapsedMs };
                storageTag = vcActive
                  ? `Voice Clean + DeepFilterNet3 (${dfOptions.attenuationLimit} dB)`
                  : `DeepFilterNet3 (${dfOptions.attenuationLimit} dB)`;
              }
            } catch (dfErr) {
              if (dfErr?.name === 'AbortError') throw dfErr;
              console.warn('DeepFilterNet3 failed, falling back to Studio Spectral Wiener:', dfErr);
              setErrorMessage(
                `DeepFilterNet3 WASM could not run here (${dfErr?.message || 'WASM error'}) — automatically completed using Studio Spectral Wiener Denoiser + Mastering instead.`
              );
              const fallbackRes = await processAudioBuffer(
                workingBuffer,
                {
                  ...effectiveOptions,
                  noiseGateEnabled: !vcActive,
                  noiseMode: 'spectral',
                  noiseGateReductionDb: Math.max(
                    18,
                    Math.min(30, Math.round(dfOptions.attenuationLimit * 0.3))
                  ),
                },
                (prog) => setProgress(prog)
              );
              outBuffer = fallbackRes.audioBuffer;
              report = fallbackRes.report;
              storageTag = vcActive ? 'Voice Clean + Spectral Fallback' : 'Spectral Wiener Fallback';
            }
          } else if (!vcActive || hasPostStages) {
            const res = await processAudioBuffer(workingBuffer, effectiveOptions, (prog) => {
              const pct = vcActive ? Math.round(68 + prog.percent * 0.3) : prog.percent;
              setProgress({ percent: pct, stage: prog.stage });
            });
            outBuffer = res.audioBuffer;
            report = res.report;
            const presetPart = keepInputLufs
              ? 'Matched LUFS'
              : activePresetId && PRESETS[activePresetId]
                ? PRESETS[activePresetId].label
                : `${options.targetLoudness} ${options.normalizeMode === 'rms' ? 'RMS' : 'LUFS'}`;
            storageTag = vcActive
              ? `Voice Clean (${voiceCleanOptions.useAiDenoise ? 'AI' : voiceCleanOptions.preset}) · ${presetPart}`
              : `Current DSP · ${presetPart}`;
          } else {
            outBuffer = workingBuffer;
            storageTag = `Voice Clean (${voiceCleanOptions.useAiDenoise ? 'AI' : voiceCleanOptions.preset})`;
          }
        }

        if (signal.aborted) {
          const err = new Error('Processing cancelled.');
          err.name = 'AbortError';
          throw err;
        }

        if (voiceCleanReport) {
          report = { ...report, voiceClean: voiceCleanReport };
        }

        const afterAnalysis = analyzeAudioBuffer(outBuffer);

        setProcessedBuffer(outBuffer);
        setProcessedAnalysis(afterAnalysis);
        setProcessingReport(report);
        setProgress({ percent: 100, stage: 'Complete' });

        // Auto-save processed output into separate Processed Storage (guard >45MB WAVs)
        const estWavBytes = outBuffer.length * outBuffer.numberOfChannels * 2 + 44;
        if (estWavBytes <= 45 * 1024 * 1024) {
          await persistProcessedBuffer(outBuffer, afterAnalysis, fileInfo?.name, storageTag);
        } else {
          setActiveProcessedId(null);
        }
      } catch (err) {
        if (err?.name === 'AbortError') {
          setProgress({ percent: 0, stage: '' });
          setErrorMessage('Processing cancelled. Your original audio buffer was not modified.');
        } else {
          console.error('Processing error:', err);
          setErrorMessage(
            processingEngine === 'deepfilternet'
              ? `DeepFilterNet3 processing failed (${err?.message || 'WASM error'}). Ensure WebAssembly SIMD is supported in your browser, or switch to Current DSP.`
              : `Audio processing failed (${err?.message || 'unexpected error'}). Try a shorter clip or fewer simultaneous stages.`
          );
        }
      } finally {
        abortControllerRef.current = null;
        setIsProcessing(false);
      }
    },
    [
      originalBuffer,
      originalAnalysis,
      isProcessing,
      processingEngine,
      dfOptions,
      voiceCleanOptions,
      noiseRegion,
      options,
      pause,
      activePresetId,
      fileInfo,
      persistProcessedBuffer,
    ]
  );

  const handleDownloadOriginalWav = useCallback(() => {
    if (!originalBuffer) return;
    try {
      const wavBlob = encodeWav16Bit(originalBuffer);
      const baseName = (fileInfo?.name || 'recording')
        .replace(/\s*\(synthetic\)$/i, '')
        .replace(/\.[^/.]+$/, '');
      triggerBlobDownload(wavBlob, `${baseName}_original.wav`);
    } catch (err) {
      console.error('Original WAV export error:', err);
      setErrorMessage('Failed to encode original 16-bit WAV file for download.');
    }
  }, [originalBuffer, fileInfo]);

  const handleDownloadWav = useCallback(() => {
    if (!processedBuffer) return;
    try {
      const wavBlob = encodeWav16Bit(processedBuffer);
      const baseName = (fileInfo?.name || 'audio')
        .replace(/\s*\(synthetic\)$/i, '')
        .replace(/\.[^/.]+$/, '');
      triggerBlobDownload(wavBlob, `${baseName}_mastered.wav`);
    } catch (err) {
      console.error('WAV export error:', err);
      setErrorMessage('Failed to encode 16-bit WAV file for download.');
    }
  }, [processedBuffer, fileInfo]);

  const handleSaveCurrentOriginal = useCallback(async () => {
    if (!originalBuffer || !originalAnalysis) return;
    const baseName = (fileInfo?.name || 'original_audio')
      .replace(/\s*\(synthetic\)$/i, '')
      .replace(/\.[^/.]+$/, '');
    await persistOriginalBuffer(originalBuffer, originalAnalysis, `${baseName}_original.wav`);
  }, [originalBuffer, originalAnalysis, fileInfo, persistOriginalBuffer]);

  const handleSaveCurrentProcessed = useCallback(async () => {
    if (!processedBuffer || !processedAnalysis) return;
    const presetPart =
      activePresetId && PRESETS[activePresetId]
        ? PRESETS[activePresetId].label
        : `${options.targetLoudness} ${options.normalizeMode === 'rms' ? 'RMS' : 'LUFS'}`;
    const presetLabel =
      processingEngine === 'deepfilternet'
        ? dfOptions.chainMastering
          ? `DeepFilterNet3 + ${presetPart}`
          : `DeepFilterNet3 (${dfOptions.attenuationLimit} dB)`
        : `Current DSP · ${presetPart}`;
    await persistProcessedBuffer(processedBuffer, processedAnalysis, fileInfo?.name, presetLabel);
  }, [
    processedBuffer,
    processedAnalysis,
    processingEngine,
    dfOptions,
    activePresetId,
    options,
    fileInfo,
    persistProcessedBuffer,
  ]);

  const handleLoadOriginalClip = useCallback(
    async (clip) => {
      if (!clip?.wavBlob) return;
      setErrorMessage(null);
      setIsDecoding(true);
      pause();
      try {
        const arrBuf = await clip.wavBlob.arrayBuffer();
        const audioCtx = getAudioContext();
        const decoded = await audioCtx.decodeAudioData(arrBuf.slice(0));
        const analysis = analyzeAudioBuffer(decoded);
        setFileInfo({ name: clip.name, size: clip.sizeBytes });
        setOriginalBuffer(decoded);
        setOriginalAnalysis(analysis);
        setActiveOriginalId(clip.id);
        setNoiseRegion(null);
        setIsSelectingNoise(false);
      } catch (err) {
        console.error('Failed to load stored original clip:', err);
        setErrorMessage(`Could not load "${clip.name}" from Original Storage.`);
      } finally {
        setIsDecoding(false);
      }
    },
    [getAudioContext, pause]
  );

  const handleLoadProcessedClip = useCallback(
    async (clip) => {
      if (!clip?.wavBlob) return;
      setErrorMessage(null);
      setIsDecoding(true);
      pause();
      try {
        const arrBuf = await clip.wavBlob.arrayBuffer();
        const audioCtx = getAudioContext();
        const decoded = await audioCtx.decodeAudioData(arrBuf.slice(0));
        const analysis = analyzeAudioBuffer(decoded);
        setProcessedBuffer(decoded);
        setProcessedAnalysis(analysis);
        setProcessingReport(null);
        setActiveProcessedId(clip.id);
      } catch (err) {
        console.error('Failed to load stored processed clip:', err);
        setErrorMessage(`Could not load "${clip.name}" from Processed Storage.`);
      } finally {
        setIsDecoding(false);
      }
    },
    [getAudioContext, pause]
  );

  const handleDownloadStoredClip = useCallback((clip) => {
    if (!clip?.wavBlob) return;
    triggerBlobDownload(clip.wavBlob, clip.name);
  }, []);

  const handleDeleteOriginalClip = useCallback(
    async (id) => {
      await deleteClipFromStore(STORE_ORIGINALS, id);
      if (activeOriginalId === id) setActiveOriginalId(null);
      await refreshStorageLists();
    },
    [activeOriginalId, refreshStorageLists]
  );

  const handleDeleteProcessedClip = useCallback(
    async (id) => {
      await deleteClipFromStore(STORE_PROCESSED, id);
      if (activeProcessedId === id) setActiveProcessedId(null);
      await refreshStorageLists();
    },
    [activeProcessedId, refreshStorageLists]
  );

  const handleClearOriginals = useCallback(async () => {
    await clearStore(STORE_ORIGINALS);
    setActiveOriginalId(null);
    await refreshStorageLists();
  }, [refreshStorageLists]);

  const handleClearProcessed = useCallback(async () => {
    await clearStore(STORE_PROCESSED);
    setActiveProcessedId(null);
    await refreshStorageLists();
  }, [refreshStorageLists]);

  const handleHeaderLoadAudio = () => {
    if (activePage === 'how-it-works') {
      navigateToPage('simple');
      setTimeout(() => fileInputRef.current?.click(), 50);
    } else {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[var(--bg)] text-[var(--text)]">
      {/* Top Navbar: Exact Quotly / Lobaverse 3-zone obsidian + gold architecture */}
      <header className="sticky top-0 z-30 border-b border-[var(--border)] bg-[var(--bg)]/95 backdrop-blur-xs">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-6 sm:gap-8">
          {/* Zone 1: Wide-tracked Quotly / Lobaverse Wordmark */}
          <button
            type="button"
            onClick={() => navigateToPage('simple')}
            className="text-xs sm:text-sm font-extrabold tracking-[0.2em] uppercase text-[var(--text)] hover:text-[var(--accent)] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            LOBA VERSE <span className="text-[var(--accent)] mx-1">·</span> AUDIO LAB
          </button>

          {/* Zone 2: Clean single-line navigation links (Normal / Simple, Advanced, How It Works) */}
          <nav
            aria-label="Primary navigation"
            className="flex items-center gap-4 sm:gap-7 h-full text-[11px] sm:text-xs font-semibold tracking-[0.14em] uppercase"
          >
            <button
              type="button"
              onClick={() => navigateToPage('simple')}
              aria-current={activePage === 'simple' ? 'page' : undefined}
              className={`h-full inline-flex items-center border-b-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activePage === 'simple'
                  ? 'border-[var(--accent)] text-[var(--text)]'
                  : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              Normal
            </button>
            <button
              type="button"
              onClick={() => navigateToPage('advanced')}
              aria-current={activePage === 'advanced' ? 'page' : undefined}
              className={`h-full inline-flex items-center border-b-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activePage === 'advanced'
                  ? 'border-[var(--accent)] text-[var(--text)]'
                  : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              Advanced
            </button>
            <button
              type="button"
              onClick={() => navigateToPage('how-it-works')}
              aria-current={activePage === 'how-it-works' ? 'page' : undefined}
              className={`h-full inline-flex items-center border-b-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activePage === 'how-it-works'
                  ? 'border-[var(--accent)] text-[var(--text)]'
                  : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              How It Works
            </button>
          </nav>

          {/* Zone 3: Single Primary Action */}
          <div className="flex items-center shrink-0">
            {processedBuffer ? (
              <button
                type="button"
                onClick={handleDownloadWav}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-[11px] font-bold tracking-[0.12em] uppercase bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-[var(--accent-ink)] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                Export WAV
              </button>
            ) : (
              <button
                type="button"
                onClick={handleHeaderLoadAudio}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-[11px] font-bold tracking-[0.12em] uppercase bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-[var(--accent-ink)] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <Upload className="w-3.5 h-3.5" />
                Load Audio
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Dedicated Separate Page: How It Works */}
      {activePage === 'how-it-works' ? (
        <HowItWorks
          onOpenSimple={() => navigateToPage('simple')}
          onOpenAdvanced={() => navigateToPage('advanced')}
        />
      ) : (
        <>
          {/* Audio Source Dropzone + Live Mic Recorder + Theme Switch */}
          <UploadBar
            fileInfo={fileInfo}
            originalAnalysis={originalAnalysis}
            isDecoding={isDecoding}
            isRecording={isRecording}
            recordingElapsedSec={recordingElapsedSec}
            inputLevelDb={inputLevelDb}
            errorMessage={errorMessage}
            onClearError={() => setErrorMessage(null)}
            onFileSelected={handleFileSelected}
            onLoadDemo={handleLoadDemo}
            onStartRecording={handleStartRecording}
            onStopRecording={stopRecording}
            onDownloadOriginalWav={handleDownloadOriginalWav}
            theme={theme}
            onToggleTheme={handleToggleTheme}
            fileInputRef={fileInputRef}
          />

          {/* Main Landing Mode Selector + Workspace */}
          <main
            id="workspace"
            className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-6 space-y-6"
          >
            {/* Simple vs Advanced Landing Options Bar */}
            <section
              aria-label="Choose Simple or Advanced mastering mode"
              className="grid grid-cols-1 md:grid-cols-2 border border-[var(--border)] bg-[var(--surface)] divide-y md:divide-y-0 md:divide-x divide-[var(--border)]"
            >
              <button
                type="button"
                onClick={() => navigateToPage('simple')}
                className={`text-left px-5 py-4 transition-colors cursor-pointer flex items-start justify-between gap-4 ${
                  activePage === 'simple'
                    ? 'bg-[var(--accent-muted)]'
                    : 'hover:bg-[var(--surface-hover)]'
                }`}
              >
                <div className="space-y-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <Sparkles
                      className={`w-4 h-4 shrink-0 ${
                        activePage === 'simple' ? 'text-[var(--accent)]' : 'text-[var(--text-subtle)]'
                      }`}
                    />
                    <span className="text-sm font-bold tracking-tight text-[var(--text)]">
                      Simple (Normal)
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    Fast one-click Voice Clean with Natural / Clear / Strong profiles, matched-loudness
                    A/B auditioning, and instant WAV download.
                  </p>
                </div>
                <span
                  className={`font-mono text-[11px] tracking-wider uppercase shrink-0 mt-0.5 ${
                    activePage === 'simple' ? 'text-[var(--accent)] font-semibold' : 'text-[var(--text-subtle)]'
                  }`}
                >
                  {activePage === 'simple' ? 'Active' : 'Select'}
                </span>
              </button>

              <button
                type="button"
                onClick={() => navigateToPage('advanced')}
                className={`text-left px-5 py-4 transition-colors cursor-pointer flex items-start justify-between gap-4 ${
                  activePage === 'advanced'
                    ? 'bg-[var(--accent-muted)]'
                    : 'hover:bg-[var(--surface-hover)]'
                }`}
              >
                <div className="space-y-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <SlidersHorizontal
                      className={`w-4 h-4 shrink-0 ${
                        activePage === 'advanced' ? 'text-[var(--accent)]' : 'text-[var(--text-subtle)]'
                      }`}
                    />
                    <span className="text-sm font-bold tracking-tight text-[var(--text)]">
                      Advanced Lab
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    Full DSP rack: Voice Clean sliders, DeepFilterNet3 controls, compressor, auto-leveler,
                    LUFS targets, FFT spectrograms &amp; IndexedDB clip storage.
                  </p>
                </div>
                <span
                  className={`font-mono text-[11px] tracking-wider uppercase shrink-0 mt-0.5 ${
                    activePage === 'advanced' ? 'text-[var(--accent)] font-semibold' : 'text-[var(--text-subtle)]'
                  }`}
                >
                  {activePage === 'advanced' ? 'Active' : 'Select'}
                </span>
              </button>
            </section>

            {activePage === 'simple' ? (
              <SimpleWorkspace
                voiceCleanOptions={voiceCleanOptions}
                onChangeVoiceCleanOption={handleChangeVoiceCleanOption}
                onApplyVoiceCleanPreset={handleApplyVoiceCleanPreset}
                options={options}
                onChangeOption={handleChangeOption}
                onProcess={() => handleProcess(true)}
                onCancelProcess={handleCancelProcess}
                isProcessing={isProcessing}
                progress={progress}
                hasAudio={Boolean(originalBuffer)}
                originalBuffer={originalBuffer}
                processedBuffer={processedBuffer}
                originalAnalysis={originalAnalysis}
                processedAnalysis={processedAnalysis}
                processingReport={processingReport}
                activeSource={activeSource}
                isPlaying={isPlaying}
                currentTime={currentTime}
                abMatchLoudness={abMatchLoudness}
                onToggleAbMatchLoudness={setAbMatchLoudness}
                onTogglePlayTrack={togglePlayTrack}
                onSeek={seek}
                onDownloadOriginalWav={handleDownloadOriginalWav}
                onDownloadWav={handleDownloadWav}
                theme={theme}
                noiseRegion={noiseRegion}
                isSelectingNoise={isSelectingNoise}
                onSelectNoiseRegion={setNoiseRegion}
                onToggleSelectingNoise={setIsSelectingNoise}
                onOpenAdvanced={() => navigateToPage('advanced')}
              />
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-[360px_minmax(0,1fr)] gap-6 items-start">
                {/* Left Parameter & Processing Column */}
                <ProcessingRack
                  processingEngine={processingEngine}
                  onChangeEngine={setProcessingEngine}
                  dfOptions={dfOptions}
                  onChangeDfOption={handleChangeDfOption}
                  voiceCleanOptions={voiceCleanOptions}
                  onChangeVoiceCleanOption={handleChangeVoiceCleanOption}
                  onApplyVoiceCleanPreset={handleApplyVoiceCleanPreset}
                  noiseRegion={noiseRegion}
                  isSelectingNoise={isSelectingNoise}
                  onToggleSelectingNoise={setIsSelectingNoise}
                  onClearNoiseRegion={() => setNoiseRegion(null)}
                  options={options}
                  activePresetId={activePresetId}
                  onApplyPreset={handleApplyPreset}
                  onChangeOption={handleChangeOption}
                  onProcess={() => handleProcess(false)}
                  onCancelProcess={handleCancelProcess}
                  isProcessing={isProcessing}
                  progress={progress}
                  hasAudio={Boolean(originalBuffer)}
                  processingReport={processingReport}
                />

                {/* Right Analysis, A/B Transport, Visualization & Comparison Storage Stage */}
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
                    onToggleAbMatchLoudness={setAbMatchLoudness}
                    originalLufs={originalAnalysis?.integratedLufs}
                    processedLufs={processedAnalysis?.integratedLufs}
                    onTogglePlayTrack={togglePlayTrack}
                    onSeek={seek}
                    onDownloadOriginalWav={handleDownloadOriginalWav}
                    onDownloadWav={handleDownloadWav}
                  />

                  <Visualizations
                    originalAnalysis={originalAnalysis}
                    processedAnalysis={processedAnalysis}
                    activeSource={activeSource}
                    currentTime={currentTime}
                    options={options}
                    voiceCleanOptions={voiceCleanOptions}
                    theme={theme}
                    onSeek={seek}
                    noiseRegion={noiseRegion}
                    isSelectingNoise={isSelectingNoise}
                    onSelectNoiseRegion={setNoiseRegion}
                    onToggleSelectingNoise={setIsSelectingNoise}
                  />

                  <MetricsComparison
                    originalAnalysis={originalAnalysis}
                    processedAnalysis={processedAnalysis}
                    options={options}
                    processingReport={processingReport}
                    voiceCleanOptions={voiceCleanOptions}
                  />

                  <StorageLibrary
                    originalClips={originalClips}
                    processedClips={processedClips}
                    activeOriginalId={activeOriginalId}
                    activeProcessedId={activeProcessedId}
                    hasOriginal={Boolean(originalBuffer)}
                    hasProcessed={Boolean(processedBuffer)}
                    onSaveCurrentOriginal={handleSaveCurrentOriginal}
                    onSaveCurrentProcessed={handleSaveCurrentProcessed}
                    onLoadOriginalClip={handleLoadOriginalClip}
                    onLoadProcessedClip={handleLoadProcessedClip}
                    onDownloadClip={handleDownloadStoredClip}
                    onDeleteOriginalClip={handleDeleteOriginalClip}
                    onDeleteProcessedClip={handleDeleteProcessedClip}
                    onClearOriginals={handleClearOriginals}
                    onClearProcessed={handleClearProcessed}
                  />
                </div>
              </div>
            )}
          </main>
        </>
      )}

      <footer className="border-t border-[var(--border)] bg-[var(--surface)]">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-4 flex flex-wrap items-center justify-between gap-3 text-xs text-[var(--text-subtle)]">
          <span className="font-mono tracking-wider uppercase text-[11px]">
            LOBA VERSE · Mini Audio Mastering Lab
          </span>
          <div className="flex items-center gap-5">
            <button
              type="button"
              onClick={() => navigateToPage('simple')}
              className="hover:text-[var(--text)] transition-colors cursor-pointer"
            >
              Normal
            </button>
            <button
              type="button"
              onClick={() => navigateToPage('advanced')}
              className="hover:text-[var(--text)] transition-colors cursor-pointer"
            >
              Advanced
            </button>
            <button
              type="button"
              onClick={() => navigateToPage('how-it-works')}
              className="hover:text-[var(--text)] transition-colors cursor-pointer"
            >
              How It Works
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
