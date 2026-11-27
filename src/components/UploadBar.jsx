import React, { useRef, useState } from 'react';
import { Upload, AudioLines, Sun, Moon, AlertTriangle, X, Mic, Square, Download } from 'lucide-react';

const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB recommended local limit
const SUPPORTED_EXTENSIONS = ['mp3', 'wav', 'm4a'];

function formatBytes(bytes) {
  if (!bytes) return '0 B';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatDuration(sec) {
  if (!Number.isFinite(sec)) return '0:00';
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, '0');
  return `${m}:${s}`;
}

export default function UploadBar({
  fileInfo,
  originalAnalysis,
  isDecoding,
  isRecording,
  recordingElapsedSec,
  inputLevelDb,
  errorMessage,
  onClearError,
  onFileSelected,
  onLoadDemo,
  onStartRecording,
  onStopRecording,
  onDownloadOriginalWav,
  theme,
  onToggleTheme,
  fileInputRef,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const internalRef = useRef(null);
  const inputRef = fileInputRef || internalRef;

  const validateAndEmit = (file) => {
    if (!file) return;
    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    const isAudioMime = file.type && file.type.startsWith('audio/');
    const isSupportedExt = SUPPORTED_EXTENSIONS.includes(ext);

    if (!isSupportedExt && !isAudioMime) {
      onFileSelected(
        null,
        `Unsupported file format ("${file.name}"). Please upload an MP3, WAV, or M4A audio file.`
      );
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      onFileSelected(
        null,
        `File is too large (${formatBytes(file.size)}). Because decoding and DSP run locally in browser memory, please keep files under 50 MB.`
      );
      return;
    }

    onFileSelected(file, null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer?.files?.[0];
    validateAndEmit(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleChange = (e) => {
    const file = e.target.files?.[0];
    validateAndEmit(file);
    e.target.value = '';
  };

  return (
    <section
      aria-label="Audio source upload"
      className="border-b border-[var(--border)] bg-[var(--surface)]"
    >
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-4">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
          {/* Drag-and-drop & file selection zone */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => !isDecoding && !isRecording && inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if ((e.key === 'Enter' || e.key === ' ') && !isDecoding && !isRecording) {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            className={`flex-1 flex flex-col sm:flex-row sm:items-center justify-between gap-4 px-4 py-3 border border-dashed transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)] ${
              isDragging
                ? 'border-[var(--accent)] bg-[var(--accent-muted)]'
                : 'border-[var(--border-hover)] hover:border-[var(--border-strong)] bg-[var(--surface-2)]'
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".mp3,.wav,.m4a,audio/mpeg,audio/wav,audio/x-wav,audio/mp4,audio/x-m4a"
              onChange={handleChange}
              className="hidden"
              aria-label="Choose an audio file"
            />

            <div className="flex items-center gap-3.5 min-w-0">
              <div className="w-9 h-9 border border-[var(--border)] bg-[var(--surface)] flex items-center justify-center text-[var(--accent)] shrink-0">
                <Upload className="w-4 h-4" />
              </div>

              <div className="min-w-0">
                {isDecoding ? (
                  <p className="text-sm font-medium text-[var(--accent)]">
                    Decoding audio file locally with Web Audio API...
                  </p>
                ) : fileInfo && originalAnalysis ? (
                  <>
                    <p className="text-sm font-semibold text-[var(--text)] truncate">
                      {fileInfo.name}
                    </p>
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs font-mono tabular-nums text-[var(--text-muted)] mt-0.5">
                      <span>{formatDuration(originalAnalysis.duration)}</span>
                      <span aria-hidden="true">·</span>
                      <span>{(originalAnalysis.sampleRate / 1000).toFixed(1)} kHz</span>
                      <span aria-hidden="true">·</span>
                      <span>
                        {originalAnalysis.channels === 1
                          ? 'Mono (1 ch)'
                          : `Stereo (${originalAnalysis.channels} ch)`}
                      </span>
                      {fileInfo.size > 0 && (
                        <>
                          <span aria-hidden="true">·</span>
                          <span>{formatBytes(fileInfo.size)}</span>
                        </>
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    <p className="text-sm font-medium text-[var(--text)]">
                      Drop an MP3, WAV, or M4A file here, or click to browse
                    </p>
                    <p className="text-xs text-[var(--text-subtle)] mt-0.5">
                      Runs 100% locally in your browser · No upload, no login · Max 50 MB recommended
                    </p>
                  </>
                )}
              </div>
            </div>

            <span className="text-xs font-semibold tracking-wider uppercase text-[var(--accent)] whitespace-nowrap shrink-0">
              {fileInfo ? 'Replace file' : 'Select file'}
            </span>
          </div>

          {/* Utility actions: Live mic recorder + Demo synthesizer + Theme toggle */}
          <div className="flex flex-wrap items-center gap-2 shrink-0">
            {isRecording ? (
              <button
                type="button"
                onClick={() => onStopRecording(false)}
                className="inline-flex items-center gap-2 px-3.5 py-2.5 text-xs font-semibold border border-[var(--status-error)] bg-[var(--status-error)] text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                title="Stop recording and load into the workspace"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
                Stop &amp; Load
              </button>
            ) : (
              <button
                type="button"
                onClick={onStartRecording}
                disabled={isDecoding}
                className="inline-flex items-center gap-2 px-3.5 py-2.5 text-xs font-medium border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-hover)] hover:bg-[var(--surface-hover)] hover:text-[var(--text)] transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-40"
                title="Record live audio directly from your microphone"
              >
                <Mic className="w-3.5 h-3.5 text-[var(--status-error)]" />
                Record Mic
              </button>
            )}

            {originalAnalysis && !isRecording && (
              <button
                type="button"
                onClick={onDownloadOriginalWav}
                className="inline-flex items-center gap-2 px-3.5 py-2.5 text-xs font-medium border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-hover)] hover:bg-[var(--surface-hover)] hover:text-[var(--text)] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                title="Download the original raw recording as 16-bit WAV"
              >
                <Download className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                Original WAV
              </button>
            )}

            <button
              type="button"
              onClick={onLoadDemo}
              disabled={isDecoding || isRecording}
              className="inline-flex items-center gap-2 px-3.5 py-2.5 text-xs font-medium border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-hover)] hover:bg-[var(--surface-hover)] hover:text-[var(--text)] transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-40"
              title="Synthesize an 8.5s noisy speech-like test clip with rumble, hiss, dynamic swings, and a 2.2s pause"
            >
              <AudioLines className="w-3.5 h-3.5 text-[var(--accent)]" />
              Demo Audio
            </button>

            <button
              type="button"
              onClick={onToggleTheme}
              aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
              className="inline-flex items-center justify-center w-10 h-10 border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-hover)] hover:bg-[var(--surface-hover)] hover:text-[var(--text)] transition-colors shrink-0 cursor-pointer"
              title={theme === 'dark' ? 'Switch to Light Paper theme' : 'Switch to Dark Obsidian theme'}
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Live Microphone Recording Monitor Strip */}
        {isRecording && (
          <div
            aria-live="polite"
            className="mt-3 px-4 py-2.5 border border-[var(--status-error)] bg-[var(--status-error-bg)] flex flex-wrap items-center justify-between gap-4"
          >
            <div className="flex items-center gap-3 min-w-0">
              <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-[var(--status-error)] whitespace-nowrap">
                <span className="w-2 h-2 bg-[var(--status-error)]" />
                Recording live
              </span>
              <span className="font-mono tabular-nums text-xs font-semibold text-[var(--text)]">
                {formatDuration(recordingElapsedSec)}
              </span>
              <span className="hidden sm:inline text-xs text-[var(--text-muted)]">
                Raw mic PCM capture (browser AGC &amp; denoising disabled)
              </span>
            </div>

            <div className="flex items-center gap-3 flex-1 sm:flex-initial sm:min-w-[240px]">
              <div className="flex-1">
                <div className="w-full h-1.5 bg-[var(--border)] overflow-hidden">
                  <div
                    className="h-full transition-all duration-75"
                    style={{
                      width: `${Math.max(2, Math.min(100, ((inputLevelDb + 60) / 60) * 100))}%`,
                      backgroundColor:
                        inputLevelDb > -6
                          ? 'var(--status-error)'
                          : inputLevelDb > -18
                            ? 'var(--status-warn)'
                            : 'var(--accent)',
                    }}
                  />
                </div>
              </div>
              <span className="font-mono tabular-nums text-[11px] text-[var(--text-secondary)] w-16 text-right shrink-0">
                {inputLevelDb <= -59 ? '-∞ dB' : `${inputLevelDb.toFixed(0)} dBFS`}
              </span>
              <button
                type="button"
                onClick={() => onStopRecording(true)}
                className="text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text)] underline underline-offset-4 cursor-pointer whitespace-nowrap"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Clear inline error / warning banner */}
        {errorMessage && (
          <div
            role="alert"
            className="mt-3 flex items-start justify-between gap-3 px-3.5 py-2.5 border border-[var(--accent-border)] bg-[var(--status-warn-bg)] text-[var(--text)] text-xs"
          >
            <div className="flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 text-[var(--status-warn)] shrink-0 mt-0.5" />
              <span className="leading-relaxed">{errorMessage}</span>
            </div>
            <button
              type="button"
              onClick={onClearError}
              aria-label="Dismiss error"
              className="p-0.5 text-[var(--text-muted)] hover:text-[var(--text)] cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
