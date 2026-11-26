import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Download, Upload } from 'lucide-react';
import { analyzeAudioBuffer } from './audio/analyzer.js';
import {
  PRESETS,
  encodeWav16Bit,
  generateDemoAudioBuffer,
  processAudioBuffer,
} from './audio/processor.js';
import { denoiseWithDeepFilterNet } from './audio/deepfilter.js';
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
  return 'light';
}

export default function App() {
  const [theme, setTheme] = useState(getInitialTheme);
  const [fileInfo, setFileInfo] = useState(null);
  const [originalBuffer, setOriginalBuffer] = useState(null);
  const [processedBuffer, setProcessedBuffer] = useState(null);
  const [originalAnalysis, setOriginalAnalysis] = useState(null);
  const [processedAnalysis, setProcessedAnalysis] = useState(null);
  const [processingReport, setProcessingReport] = useState(null);

  const [processingEngine, setProcessingEngine] = useState('dsp'); // 'dsp' | 'deepfilternet'
  const [dfOptions, setDfOptions] = useState({
    attenuationLimit: 80,
    postFilterBeta: 0.02,
    chainMastering: true,
  });

  const [activePresetId, setActivePresetId] = useState('podcast');
  const [options, setOptions] = useState(() => ({ ...PRESETS.podcast.options }));

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

  const {
    activeSource,
    isPlaying,
    currentTime,
    togglePlayTrack,
    switchSource,
    pause,
    seek,
    getAudioContext,
  } = useABPlayer(originalBuffer, processedBuffer);

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

          // Automatically persist raw mic recording into Original Storage
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

  // Sync light/dark mode with system prefers-color-scheme and manual toggle
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [theme]);

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handleSystemChange = (e) => {
      try {
        if (!localStorage.getItem('maml-theme')) {
          setTheme(e.matches ? 'dark' : 'light');
        }
      } catch {
        setTheme(e.matches ? 'dark' : 'light');
      }
    };
    mq.addEventListener('change', handleSystemChange);
    return () => mq.removeEventListener('change', handleSystemChange);
  }, []);

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
      if (!originalBuffer || isRecording) return;

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
  }, [originalBuffer, processedBuffer, activeSource, isRecording, togglePlayTrack, switchSource]);

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
        // Decode audio bytes into uncompressed 32-bit float PCM AudioBuffer
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
  }, []);

  const handleChangeOption = useCallback((key, value) => {
    setActivePresetId(null);
    setOptions((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleChangeDfOption = useCallback((key, value) => {
    setDfOptions((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleProcess = useCallback(async () => {
    if (!originalBuffer || isProcessing) return;
    setErrorMessage(null);
    setIsProcessing(true);
    setProgress({
      percent: 4,
      stage:
        processingEngine === 'deepfilternet'
          ? 'Initializing DeepFilterNet3 neural engine...'
          : 'Preparing PCM channel buffers...',
    });
    pause();

    try {
      let outBuffer;
      let report = {};
      let storageTag = '';

      if (processingEngine === 'deepfilternet') {
        try {
          const dfRes = await denoiseWithDeepFilterNet(
            originalBuffer,
            dfOptions,
            (prog) => {
              const scaledPct = dfOptions.chainMastering
                ? Math.round(prog.percent * 0.65)
                : Math.min(96, prog.percent);
              setProgress({ percent: scaledPct, stage: prog.stage });
            }
          );

          if (dfOptions.chainMastering) {
            const postRes = await processAudioBuffer(
              dfRes.audioBuffer,
              { ...options, noiseGateEnabled: false },
              (prog) => {
                const scaledPct = Math.round(65 + prog.percent * 0.33);
                setProgress({ percent: scaledPct, stage: prog.stage });
              }
            );
            outBuffer = postRes.audioBuffer;
            report = { ...postRes.report, deepFilterElapsedMs: dfRes.elapsedMs };
            const presetPart =
              activePresetId && PRESETS[activePresetId]
                ? PRESETS[activePresetId].label
                : `${options.targetLoudness} ${options.normalizeMode === 'rms' ? 'RMS' : 'LUFS'}`;
            storageTag = `DeepFilterNet3 + ${presetPart}`;
          } else {
            outBuffer = dfRes.audioBuffer;
            report = { deepFilterElapsedMs: dfRes.elapsedMs };
            storageTag = `DeepFilterNet3 (${dfOptions.attenuationLimit} dB)`;
          }
        } catch (dfErr) {
          console.warn('DeepFilterNet3 failed, falling back to Studio Spectral Wiener:', dfErr);
          setErrorMessage(
            `DeepFilterNet3 WASM could not run here (${dfErr?.message || 'WASM error'}) — automatically completed using Studio Spectral Wiener Denoiser + Mastering instead.`
          );
          const fallbackRes = await processAudioBuffer(
            originalBuffer,
            {
              ...options,
              noiseGateEnabled: true,
              noiseMode: 'spectral',
              noiseGateReductionDb: Math.max(18, Math.min(30, Math.round(dfOptions.attenuationLimit * 0.3))),
            },
            (prog) => setProgress(prog)
          );
          outBuffer = fallbackRes.audioBuffer;
          report = fallbackRes.report;
          storageTag = 'Spectral Wiener Fallback';
        }
      } else {
        const res = await processAudioBuffer(originalBuffer, options, (prog) =>
          setProgress(prog)
        );
        outBuffer = res.audioBuffer;
        report = res.report;
        const presetPart =
          activePresetId && PRESETS[activePresetId]
            ? PRESETS[activePresetId].label
            : `${options.targetLoudness} ${options.normalizeMode === 'rms' ? 'RMS' : 'LUFS'}`;
        storageTag = `Current DSP · ${presetPart}`;
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
      console.error('Processing error:', err);
      setErrorMessage(
        processingEngine === 'deepfilternet'
          ? `DeepFilterNet3 processing failed (${err?.message || 'WASM error'}). Ensure WebAssembly SIMD is supported in your browser, or switch to Current DSP.`
          : 'An unexpected error occurred while processing the audio buffer. Try a shorter clip or fewer simultaneous stages.'
      );
    } finally {
      setIsProcessing(false);
    }
  }, [
    originalBuffer,
    isProcessing,
    processingEngine,
    dfOptions,
    options,
    pause,
    activePresetId,
    fileInfo,
    persistProcessedBuffer,
  ]);

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

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-[#090D16] text-slate-900 dark:text-slate-100">
      {/* Top Bar Contract: strictly 3 zones separated by gap-8 */}
      <header className="sticky top-0 z-30 border-b border-slate-200 dark:border-slate-800 bg-white/95 dark:bg-[#090D16]/95 backdrop-blur-xs">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-8">
          <a
            href="#workspace"
            className="text-base font-bold tracking-tight text-slate-900 dark:text-slate-100 whitespace-nowrap shrink-0"
          >
            Mini Audio Mastering Lab
          </a>

          <nav
            aria-label="Primary navigation"
            className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600 dark:text-slate-400"
          >
            <a
              href="#workspace"
              className="hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0"
            >
              Workspace
            </a>
            <a
              href="#metrics"
              className="hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0"
            >
              Metrics
            </a>
            <a
              href="#visualizers"
              className="hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0"
            >
              Visualizers
            </a>
            <a
              href="#storage"
              className="hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0"
            >
              Storage
            </a>
            <a
              href="#how-it-works"
              className="hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0"
            >
              How It Works
            </a>
          </nav>

          <div className="flex items-center shrink-0">
            {processedBuffer ? (
              <button
                type="button"
                onClick={handleDownloadWav}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-md bg-sky-600 hover:bg-sky-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                Export WAV
              </button>
            ) : (
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-md bg-slate-900 hover:bg-slate-800 dark:bg-slate-100 dark:hover:bg-white text-white dark:text-slate-950 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <Upload className="w-3.5 h-3.5" />
                Load Audio
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Audio Source Dropzone + Live Mic Recorder + Theme Toggle */}
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

      {/* Asymmetric Split Console Workspace */}
      <main id="workspace" className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-[360px_minmax(0,1fr)] gap-6 items-start">
          {/* Left Parameter & Processing Column */}
          <ProcessingRack
            processingEngine={processingEngine}
            onChangeEngine={setProcessingEngine}
            dfOptions={dfOptions}
            onChangeDfOption={handleChangeDfOption}
            options={options}
            activePresetId={activePresetId}
            onApplyPreset={handleApplyPreset}
            onChangeOption={handleChangeOption}
            onProcess={handleProcess}
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
              theme={theme}
              onSeek={seek}
            />

            <MetricsComparison
              originalAnalysis={originalAnalysis}
              processedAnalysis={processedAnalysis}
              options={options}
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
      </main>

      {/* Plain-Language "How It Works" + Limitations & Extension Ideas */}
      <HowItWorks />

      <footer className="border-t border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#090D16]">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-4 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500 dark:text-slate-400">
          <span>Mini Audio Mastering Lab · Local Web Audio API DSP Workbench</span>
          <span>All decoding, analysis, and 16-bit WAV encoding run locally in your browser.</span>
        </div>
      </footer>
    </div>
  );
}
