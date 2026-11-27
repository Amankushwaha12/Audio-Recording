import React from 'react';
import { Play, Pause, Download } from 'lucide-react';

function formatTimecode(sec) {
  if (!Number.isFinite(sec) || sec < 0) return '00:00.0';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const tenths = Math.floor((sec - Math.floor(sec)) * 10);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${tenths}`;
}

export default function TransportDeck({
  originalDuration,
  processedDuration,
  activeSource,
  isPlaying,
  currentTime,
  hasOriginal,
  hasProcessed,
  abMatchLoudness,
  onToggleAbMatchLoudness,
  originalLufs,
  processedLufs,
  onTogglePlayTrack,
  onSeek,
  onDownloadOriginalWav,
  onDownloadWav,
}) {
  const activeDuration =
    activeSource === 'processed' && hasProcessed
      ? processedDuration
      : originalDuration || 0;

  const durationTrimmed =
    hasOriginal &&
    hasProcessed &&
    Math.abs(originalDuration - processedDuration) > 0.15;

  const isPlayingOriginal = isPlaying && activeSource === 'original';
  const isPlayingProcessed = isPlaying && activeSource === 'processed';

  const hasValidLufs =
    hasOriginal &&
    hasProcessed &&
    Number.isFinite(originalLufs) &&
    Number.isFinite(processedLufs) &&
    originalLufs > -65 &&
    processedLufs > -65;

  const lufsDiff = hasValidLufs ? processedLufs - originalLufs : 0;
  let matchReadout = '';
  if (hasValidLufs && abMatchLoudness) {
    if (Math.abs(lufsDiff) < 0.15) {
      matchReadout = '0.0 LU diff';
    } else if (lufsDiff > 0) {
      matchReadout = `B trimmed -${lufsDiff.toFixed(1)} dB`;
    } else {
      matchReadout = `A trimmed -${Math.abs(lufsDiff).toFixed(1)} dB`;
    }
  }

  return (
    <div className="bg-[var(--surface)] border border-[var(--border)] p-4 space-y-3.5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* A/B Dedicated Play/Pause Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={!hasOriginal}
            onClick={() => onTogglePlayTrack('original')}
            className={`inline-flex items-center gap-2 px-3.5 py-2 text-xs font-semibold border transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
              activeSource === 'original' && hasOriginal
                ? 'border-[var(--text)] bg-[var(--text)] text-[var(--bg)]'
                : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-hover)] hover:bg-[var(--surface-hover)]'
            }`}
          >
            {isPlayingOriginal ? (
              <Pause className="w-3.5 h-3.5 fill-current" />
            ) : (
              <Play className="w-3.5 h-3.5 fill-current" />
            )}
            {isPlayingOriginal ? 'Pause Original (A)' : 'Play Original (A)'}
          </button>

          <button
            type="button"
            disabled={!hasProcessed}
            onClick={() => onTogglePlayTrack('processed')}
            className={`inline-flex items-center gap-2 px-3.5 py-2 text-xs font-semibold border transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
              activeSource === 'processed' && hasProcessed
                ? 'border-[var(--accent)] bg-[var(--accent)] text-[var(--accent-ink)]'
                : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-hover)] hover:bg-[var(--surface-hover)]'
            }`}
          >
            {isPlayingProcessed ? (
              <Pause className="w-3.5 h-3.5 fill-current" />
            ) : (
              <Play className="w-3.5 h-3.5 fill-current" />
            )}
            {isPlayingProcessed ? 'Pause Processed (B)' : 'Play Processed (B)'}
          </button>

          <label
            className="inline-flex items-center gap-1.5 ml-1 text-xs text-[var(--text-muted)] cursor-pointer select-none"
            title="Attenuates the louder track during A/B playback so loudness differences never bias your ears"
          >
            <input
              type="checkbox"
              checked={Boolean(abMatchLoudness)}
              onChange={(e) => onToggleAbMatchLoudness?.(e.target.checked)}
              className="w-3.5 h-3.5 accent-[var(--accent)] cursor-pointer"
            />
            <span>Matched-loudness A/B</span>
            {matchReadout && (
              <span className="font-mono tabular-nums text-[11px] text-[var(--accent)]">
                ({matchReadout})
              </span>
            )}
          </label>
        </div>

        {/* Download Original & Processed WAV */}
        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <button
            type="button"
            disabled={!hasOriginal}
            onClick={onDownloadOriginalWav}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-hover)] hover:bg-[var(--surface-hover)] hover:text-[var(--text)] transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
            title="Download the original / raw recording as a 16-bit WAV file"
          >
            <Download className="w-3.5 h-3.5 text-[var(--text-muted)]" />
            Original WAV
          </button>

          <button
            type="button"
            disabled={!hasProcessed}
            onClick={onDownloadWav}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium border border-[var(--accent-border)] bg-[var(--accent-muted)] text-[var(--text)] hover:border-[var(--accent)] transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
            title="Download the processed / mastered result as a 16-bit WAV file"
          >
            <Download className="w-3.5 h-3.5 text-[var(--accent)]" />
            Processed WAV
          </button>
        </div>
      </div>

      {/* Shared Seek Scrubber & Timecode */}
      <div className="flex items-center gap-3">
        <span className="text-xs font-mono tabular-nums text-[var(--text)] w-16 shrink-0">
          {formatTimecode(currentTime)}
        </span>

        <input
          type="range"
          aria-label="Shared playback seek position"
          min={0}
          max={Math.max(0.01, activeDuration)}
          step={0.01}
          value={Math.min(currentTime, activeDuration)}
          disabled={!hasOriginal}
          onChange={(e) => onSeek(parseFloat(e.target.value))}
          className="lab-slider flex-1"
        />

        <span className="text-xs font-mono tabular-nums text-[var(--text-subtle)] w-16 text-right shrink-0">
          {formatTimecode(activeDuration)}
        </span>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-[var(--text-subtle)]">
        <span>Switch mid-playback to A/B compare · Keys: Space / A / B</span>
        {durationTrimmed && (
          <span>
            Silence trimming shortened track from{' '}
            <span className="font-mono tabular-nums text-[var(--text-muted)]">
              {originalDuration.toFixed(2)} s
            </span>{' '}
            to{' '}
            <span className="font-mono tabular-nums text-[var(--text-muted)]">
              {processedDuration.toFixed(2)} s
            </span>
          </span>
        )}
      </div>
    </div>
  );
}
