import React from 'react';
import { Download, Trash2, FolderDown, ArrowUpRight, HardDrive } from 'lucide-react';

function formatBytes(bytes) {
  if (!bytes) return '0 KB';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatDuration(sec) {
  if (!Number.isFinite(sec)) return '0.0s';
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, '0');
  return m > 0 ? `${m}:${s}` : `${sec.toFixed(1)}s`;
}

function formatTimestamp(ms) {
  if (!ms) return '';
  const d = new Date(ms);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function StorageColumn({
  title,
  subtitle,
  clips,
  activeClipId,
  loadLabel,
  canSaveCurrent,
  saveLabel,
  onSaveCurrent,
  onLoadClip,
  onDownloadClip,
  onDeleteClip,
  onClearAll,
  emptyText,
}) {
  const totalBytes = clips.reduce((acc, c) => acc + (c.sizeBytes || 0), 0);

  return (
    <div className="flex flex-col min-w-0">
      {/* Column Header */}
      <div className="px-4 py-3 border-b border-[var(--border)] bg-[var(--surface-2)] flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <h3 className="text-xs font-semibold text-[var(--text)]">{title}</h3>
          <div className="text-[11px] font-mono tabular-nums text-[var(--text-subtle)] mt-0.5">
            {clips.length} {clips.length === 1 ? 'clip' : 'clips'} · {formatBytes(totalBytes)}
            {subtitle ? ` · ${subtitle}` : ''}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            disabled={!canSaveCurrent}
            onClick={onSaveCurrent}
            className="inline-flex items-center gap-1 px-2.5 py-1.5 text-[11px] font-medium border border-[var(--border)] bg-[var(--surface)] text-[var(--text-secondary)] hover:border-[var(--border-hover)] hover:bg-[var(--surface-hover)] transition-colors whitespace-nowrap cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <FolderDown className="w-3 h-3" />
            {saveLabel}
          </button>

          {clips.length > 0 && (
            <button
              type="button"
              onClick={onClearAll}
              className="px-2 py-1.5 text-[11px] font-medium text-[var(--text-subtle)] hover:text-[var(--status-error)] transition-colors whitespace-nowrap cursor-pointer"
              title={`Clear all clips from ${title}`}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Stored Clips List */}
      <div className="divide-y divide-[var(--border)] max-h-[300px] overflow-y-auto">
        {clips.length === 0 ? (
          <p className="px-4 py-6 text-xs text-[var(--text-muted)]">{emptyText}</p>
        ) : (
          clips.map((clip) => {
            const isLoaded = activeClipId === clip.id;
            return (
              <div
                key={clip.id}
                className={`px-4 py-3 flex items-center justify-between gap-3 transition-colors ${
                  isLoaded ? 'bg-[var(--accent-muted)]' : 'hover:bg-[var(--surface-hover)]'
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-xs font-medium text-[var(--text)] truncate">
                      {clip.name}
                    </span>
                    {isLoaded && (
                      <span className="text-[11px] font-mono font-medium shrink-0 text-[var(--accent)]">
                        Loaded
                      </span>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] font-mono tabular-nums text-[var(--text-subtle)] mt-0.5">
                    <span>{formatDuration(clip.duration)}</span>
                    <span aria-hidden="true">·</span>
                    <span>{clip.integratedLufs.toFixed(1)} LUFS</span>
                    <span aria-hidden="true">·</span>
                    <span>Peak {clip.peakDb.toFixed(1)} dB</span>
                    <span aria-hidden="true">·</span>
                    <span>{formatBytes(clip.sizeBytes)}</span>
                    {clip.presetLabel && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="font-sans text-[var(--text-muted)]">
                          {clip.presetLabel}
                        </span>
                      </>
                    )}
                    <span aria-hidden="true">·</span>
                    <span>{formatTimestamp(clip.createdAt)}</span>
                  </div>
                </div>

                <div className="flex items-center gap-1.5 shrink-0">
                  <button
                    type="button"
                    onClick={() => onLoadClip(clip)}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 text-[11px] font-medium border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-hover)] hover:bg-[var(--surface-hover)] transition-colors whitespace-nowrap cursor-pointer"
                    title={`Load into ${loadLabel} for comparison`}
                  >
                    <ArrowUpRight className="w-3 h-3" />
                    {loadLabel}
                  </button>

                  <button
                    type="button"
                    onClick={() => onDownloadClip(clip)}
                    aria-label={`Download ${clip.name}`}
                    className="inline-flex items-center justify-center w-8 h-8 border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-secondary)] hover:text-[var(--accent)] hover:border-[var(--border-hover)] transition-colors cursor-pointer"
                    title="Download 16-bit WAV"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </button>

                  <button
                    type="button"
                    onClick={() => onDeleteClip(clip.id)}
                    aria-label={`Delete ${clip.name}`}
                    className="inline-flex items-center justify-center w-8 h-8 border border-[var(--border)] text-[var(--text-subtle)] hover:text-[var(--status-error)] hover:border-[var(--status-error)] transition-colors cursor-pointer"
                    title="Remove from browser storage"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

export default function StorageLibrary({
  originalClips,
  processedClips,
  activeOriginalId,
  activeProcessedId,
  hasOriginal,
  hasProcessed,
  onSaveCurrentOriginal,
  onSaveCurrentProcessed,
  onLoadOriginalClip,
  onLoadProcessedClip,
  onDownloadClip,
  onDeleteOriginalClip,
  onDeleteProcessedClip,
  onClearOriginals,
  onClearProcessed,
}) {
  return (
    <section
      id="storage"
      aria-label="Separate local storage for original and processed recordings"
      className="bg-[var(--surface)] border border-[var(--border)] overflow-hidden"
    >
      <div className="px-4 py-3 border-b border-[var(--border)] flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <HardDrive className="w-4 h-4 text-[var(--accent)]" />
          <h2 className="text-sm font-bold tracking-tight text-[var(--text)]">
            Local Comparison Storage (Original vs Processed)
          </h2>
        </div>
        <span className="text-xs text-[var(--text-subtle)]">
          Stored separately in IndexedDB · Load any Original (A) with any Processed (B)
        </span>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 divide-y xl:divide-y-0 xl:divide-x divide-[var(--border)]">
        <StorageColumn
          title="Original Storage (Raw / Source)"
          subtitle="Auto-saves mic recordings"
          clips={originalClips}
          activeClipId={activeOriginalId}
          loadLabel="Load A"
          canSaveCurrent={hasOriginal}
          saveLabel="Save Original"
          onSaveCurrent={onSaveCurrentOriginal}
          onLoadClip={onLoadOriginalClip}
          onDownloadClip={onDownloadClip}
          onDeleteClip={onDeleteOriginalClip}
          onClearAll={onClearOriginals}
          emptyText="No original clips stored yet. Record from your microphone (auto-saved here) or click Save Original after uploading a file."
        />

        <StorageColumn
          title="Processed Storage (Mastered)"
          subtitle="Auto-saves mastered outputs"
          clips={processedClips}
          activeClipId={activeProcessedId}
          loadLabel="Load B"
          canSaveCurrent={hasProcessed}
          saveLabel="Save Processed"
          onSaveCurrent={onSaveCurrentProcessed}
          onLoadClip={onLoadProcessedClip}
          onDownloadClip={onDownloadClip}
          onDeleteClip={onDeleteProcessedClip}
          onClearAll={onClearProcessed}
          emptyText="No processed clips stored yet. Run Process Audio to store mastered takes separately and compare presets side by side."
        />
      </div>
    </section>
  );
}
