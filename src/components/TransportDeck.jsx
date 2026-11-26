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

  return (
    <div className="bg-white dark:bg-[#111827] border border-slate-200 dark:border-slate-800 rounded-lg p-4 space-y-3.5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* A/B Dedicated Play/Pause Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={!hasOriginal}
            onClick={() => onTogglePlayTrack('original')}
            className={`inline-flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
              activeSource === 'original' && hasOriginal
                ? 'border-slate-900 dark:border-slate-200 bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-950'
                : 'border-slate-300 dark:border-slate-700 bg-white dark:bg-[#0D1320] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800'
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
            className={`inline-flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
              activeSource === 'processed' && hasProcessed
                ? 'border-sky-600 dark:border-cyan-400 bg-sky-600 dark:bg-cyan-500 text-white dark:text-slate-950'
                : 'border-slate-300 dark:border-slate-700 bg-white dark:bg-[#0D1320] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800'
            }`}
          >
            {isPlayingProcessed ? (
              <Pause className="w-3.5 h-3.5 fill-current" />
            ) : (
              <Play className="w-3.5 h-3.5 fill-current" />
            )}
            {isPlayingProcessed ? 'Pause Processed (B)' : 'Play Processed (B)'}
          </button>

          <span className="hidden sm:inline text-xs text-slate-500 dark:text-slate-400 ml-1">
            Switch mid-playback to A/B compare · Keys: Space / A / B
          </span>
        </div>

        {/* Download Original & Processed WAV */}
        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <button
            type="button"
            disabled={!hasOriginal}
            onClick={onDownloadOriginalWav}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-[#0D1320] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
            title="Download the original / raw recording as a 16-bit WAV file"
          >
            <Download className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />
            Original WAV
          </button>

          <button
            type="button"
            disabled={!hasProcessed}
            onClick={onDownloadWav}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-[#0D1320] text-slate-800 dark:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
            title="Download the processed / mastered result as a 16-bit WAV file"
          >
            <Download className="w-3.5 h-3.5 text-sky-600 dark:text-cyan-400" />
            Processed WAV
          </button>
        </div>
      </div>

      {/* Shared Seek Scrubber & Timecode */}
      <div className="flex items-center gap-3">
        <span className="text-xs font-mono tabular-nums text-slate-700 dark:text-slate-200 w-16 shrink-0">
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

        <span className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400 w-16 text-right shrink-0">
          {formatTimecode(activeDuration)}
        </span>
      </div>

      {durationTrimmed && (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Note: Silence trimming shortened the processed track from{' '}
          <span className="font-mono tabular-nums">{originalDuration.toFixed(2)} s</span> to{' '}
          <span className="font-mono tabular-nums">{processedDuration.toFixed(2)} s</span>.
          Shared seek stays synchronized up to the processed end time.
        </p>
      )}
    </div>
  );
}
