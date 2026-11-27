import React, { useEffect, useRef, useState } from 'react';

function useResizeTick(ref) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => setTick((t) => t + 1));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return tick;
}

/**
 * Perceptual scientific color map for FFT spectrogram rendering.
 * Maps normalized intensity t in [0, 1] (-90 dB to -6 dB) to RGB.
 */
function spectrogramColor(t, isDark) {
  const clamped = Math.max(0, Math.min(1, t));
  if (isDark) {
    // Deep slate -> indigo -> teal/cyan -> warm amber -> bright white-yellow
    if (clamped < 0.25) {
      const k = clamped / 0.25;
      return [
        Math.round(9 + k * 14),
        Math.round(13 + k * 24),
        Math.round(24 + k * 56),
      ];
    }
    if (clamped < 0.55) {
      const k = (clamped - 0.25) / 0.3;
      return [
        Math.round(23 - k * 15),
        Math.round(37 + k * 125),
        Math.round(80 + k * 115),
      ];
    }
    if (clamped < 0.82) {
      const k = (clamped - 0.55) / 0.27;
      return [
        Math.round(8 + k * 237),
        Math.round(162 + k * 15),
        Math.round(195 - k * 170),
      ];
    }
    const k = (clamped - 0.82) / 0.18;
    return [
      Math.round(245 + k * 10),
      Math.round(177 + k * 75),
      Math.round(25 + k * 210),
    ];
  } else {
    // Light lab mode: crisp off-white -> pale slate -> cobalt/sky -> amber -> crimson
    if (clamped < 0.25) {
      const k = clamped / 0.25;
      return [
        Math.round(248 - k * 34),
        Math.round(250 - k * 22),
        Math.round(252 - k * 8),
      ];
    }
    if (clamped < 0.55) {
      const k = (clamped - 0.25) / 0.3;
      return [
        Math.round(214 - k * 195),
        Math.round(228 - k * 96),
        Math.round(244 - k * 46),
      ];
    }
    if (clamped < 0.82) {
      const k = (clamped - 0.55) / 0.27;
      return [
        Math.round(19 + k * 215),
        Math.round(132 - k * 20),
        Math.round(198 - k * 182),
      ];
    }
    const k = (clamped - 0.82) / 0.18;
    return [
      Math.round(234 - k * 45),
      Math.round(112 - k * 75),
      Math.round(16 + k * 30),
    ];
  }
}

/**
 * Single high-DPI Waveform Canvas with peak + RMS layers, dB gridlines,
 * synchronized playhead, and click-to-seek.
 */
function WaveformCanvas({
  title,
  analysis,
  variant, // 'original' | 'processed'
  currentTime,
  isActiveSource,
  isDark,
  onSeek,
  noiseRegion,
  isSelectingNoise,
  onSelectNoiseRegion,
  onToggleSelectingNoise,
}) {
  const canvasRef = useRef(null);
  const resizeTick = useResizeTick(canvasRef);
  const [hoverTime, setHoverTime] = useState(null);
  const [dragSelection, setDragSelection] = useState(null);
  const isDraggingSelectionRef = useRef(false);
  const dragStartSecRef = useRef(0);

  const activeSelection = variant === 'original' ? dragSelection || noiseRegion : null;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 400;
    const height = canvas.clientHeight || 136;

    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = isDark ? '#0B101B' : '#F8FAFC';
    ctx.fillRect(0, 0, width, height);

    const midY = height / 2;

    // Horizontal reference grid (0 dB, -6 dB = 0.5, center = -inf)
    ctx.strokeStyle = isDark ? 'rgba(148, 163, 184, 0.14)' : 'rgba(15, 23, 42, 0.08)';
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach((ratio) => {
      const y = Math.round(height * ratio) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    });

    if (!analysis?.waveform) {
      ctx.fillStyle = isDark ? '#64748B' : '#94A3B8';
      ctx.font = '12px "IBM Plex Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        variant === 'processed'
          ? 'Click "Process Audio" to generate processed waveform'
          : 'Load an audio file to inspect waveform',
        width / 2,
        midY + 4
      );
      ctx.restore();
      return;
    }

    const { bins, min, max, rms } = analysis.waveform;
    const peakColor =
      variant === 'processed'
        ? isDark
          ? 'rgba(6, 182, 212, 0.55)'
          : 'rgba(2, 132, 199, 0.45)'
        : isDark
          ? 'rgba(148, 163, 184, 0.48)'
          : 'rgba(71, 85, 105, 0.42)';

    const rmsColor =
      variant === 'processed'
        ? isDark
          ? '#22D3EE'
          : '#0284C7'
        : isDark
          ? '#CBD5E1'
          : '#334155';

    const usableHalfH = midY - 8;

    // Draw selected noise-profile region behind waveform
    if (activeSelection && analysis.duration > 0) {
      const x1 = Math.max(0, Math.min(width, (activeSelection.startSec / analysis.duration) * width));
      const x2 = Math.max(0, Math.min(width, (activeSelection.endSec / analysis.duration) * width));
      const rx = Math.min(x1, x2);
      const rw = Math.max(2, Math.abs(x2 - x1));
      ctx.fillStyle = isDark ? 'rgba(245, 158, 11, 0.18)' : 'rgba(217, 119, 6, 0.16)';
      ctx.fillRect(rx, 0, rw, height);
      ctx.strokeStyle = isDark ? 'rgba(251, 191, 36, 0.85)' : 'rgba(180, 83, 9, 0.8)';
      ctx.lineWidth = 1.25;
      ctx.strokeRect(rx + 0.5, 0.5, Math.max(1, rw - 1), height - 1);
    }

    // Draw peak envelope
    ctx.fillStyle = peakColor;
    for (let x = 0; x < width; x++) {
      const b = Math.min(bins - 1, Math.floor((x / width) * bins));
      const yTop = midY - max[b] * usableHalfH;
      const yBot = midY - min[b] * usableHalfH;
      const barH = Math.max(1, yBot - yTop);
      ctx.fillRect(x, yTop, 1, barH);
    }

    // Draw RMS inner body
    ctx.fillStyle = rmsColor;
    for (let x = 0; x < width; x++) {
      const b = Math.min(bins - 1, Math.floor((x / width) * bins));
      const h = rms[b] * usableHalfH;
      ctx.fillRect(x, midY - h, 1, Math.max(1, h * 2));
    }

    // dB scale hints
    ctx.fillStyle = isDark ? 'rgba(148, 163, 184, 0.65)' : 'rgba(71, 85, 105, 0.65)';
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('0 dB', 6, 12);
    ctx.fillText('-6 dB', 6, Math.round(height * 0.25) - 3);

    // Label selected noise region
    if (activeSelection && analysis.duration > 0) {
      const rx = Math.max(
        4,
        Math.min(width - 135, (Math.min(activeSelection.startSec, activeSelection.endSec) / analysis.duration) * width + 4)
      );
      ctx.fillStyle = isDark ? '#FCD34D' : '#92400E';
      ctx.font = '10px "IBM Plex Mono", monospace';
      ctx.fillText(
        `Noise ${activeSelection.startSec.toFixed(2)}–${activeSelection.endSec.toFixed(2)}s`,
        rx,
        height - 6
      );
    }

    // Synchronized playhead
    if (analysis.duration > 0) {
      const playX = Math.max(
        0,
        Math.min(width - 1, (currentTime / analysis.duration) * width)
      );
      ctx.strokeStyle = isActiveSource
        ? isDark
          ? '#F59E0B'
          : '#D97706'
        : isDark
          ? 'rgba(245, 158, 11, 0.45)'
          : 'rgba(217, 119, 6, 0.45)';
      ctx.lineWidth = isActiveSource ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(playX + 0.5, 0);
      ctx.lineTo(playX + 0.5, height);
      ctx.stroke();
    }

    // Hover cursor guide
    if (hoverTime != null && analysis.duration > 0) {
      const hx = (hoverTime / analysis.duration) * width;
      ctx.strokeStyle = isDark ? 'rgba(226, 232, 240, 0.35)' : 'rgba(15, 23, 42, 0.25)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hx + 0.5, 0);
      ctx.lineTo(hx + 0.5, height);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.restore();
  }, [analysis, variant, currentTime, isActiveSource, isDark, hoverTime, activeSelection, resizeTick]);

  const timeFromPointerEvent = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    return ratio * (analysis?.duration || 0);
  };

  const handleMouseDown = (e) => {
    if (!analysis?.duration) return;
    if (variant === 'original' && (isSelectingNoise || e.shiftKey)) {
      e.preventDefault();
      const t = timeFromPointerEvent(e);
      isDraggingSelectionRef.current = true;
      dragStartSecRef.current = t;
      setDragSelection({ startSec: t, endSec: t });
    }
  };

  const handleMouseUp = (e) => {
    if (!isDraggingSelectionRef.current) return;
    isDraggingSelectionRef.current = false;
    const endSec = timeFromPointerEvent(e);
    const startSec = dragStartSecRef.current;
    const minT = Math.min(startSec, endSec);
    const maxT = Math.max(startSec, endSec);
    setDragSelection(null);
    if (maxT - minT >= 0.08) {
      onSelectNoiseRegion?.({ startSec: minT, endSec: maxT });
      if (isSelectingNoise) {
        onToggleSelectingNoise?.(false);
      }
    }
  };

  const handleClick = (e) => {
    if (!analysis?.duration) return;
    if (variant === 'original' && (isSelectingNoise || e.shiftKey)) return;
    onSeek(timeFromPointerEvent(e));
  };

  const handleMouseMove = (e) => {
    if (!analysis?.duration) return;
    const t = timeFromPointerEvent(e);
    setHoverTime(t);
    if (isDraggingSelectionRef.current) {
      setDragSelection({
        startSec: Math.min(dragStartSecRef.current, t),
        endSec: Math.max(dragStartSecRef.current, t),
      });
    }
  };

  const handleMouseLeave = () => {
    setHoverTime(null);
    if (isDraggingSelectionRef.current && dragSelection) {
      isDraggingSelectionRef.current = false;
      if (dragSelection.endSec - dragSelection.startSec >= 0.08) {
        onSelectNoiseRegion?.(dragSelection);
        if (isSelectingNoise) onToggleSelectingNoise?.(false);
      }
      setDragSelection(null);
    }
  };

  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="font-medium text-slate-700 dark:text-slate-200">{title}</span>
          {variant === 'original' && analysis && onToggleSelectingNoise && (
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => onToggleSelectingNoise(!isSelectingNoise)}
                className={`px-1.5 py-0.5 text-[11px] font-medium rounded border transition-colors cursor-pointer ${
                  isSelectingNoise
                    ? 'border-amber-500 bg-amber-50 dark:bg-amber-950/50 text-amber-800 dark:text-amber-300'
                    : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
                title="Drag across a quiet noise-only pause on this waveform (or hold Shift + drag)"
              >
                {isSelectingNoise ? 'Drag on waveform...' : 'Select noise sample'}
              </button>
              {noiseRegion && (
                <button
                  type="button"
                  onClick={() => onSelectNoiseRegion?.(null)}
                  className="text-[11px] text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 underline underline-offset-2 cursor-pointer"
                >
                  Clear
                </button>
              )}
            </div>
          )}
        </div>
        <span className="font-mono tabular-nums text-[11px] text-slate-500 dark:text-slate-400">
          {hoverTime != null
            ? `Cursor: ${hoverTime.toFixed(2)} s`
            : analysis
              ? `Peak ${analysis.peakDb.toFixed(1)} dBFS · RMS ${analysis.rmsDb.toFixed(1)} dBFS`
              : '—'}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className={`w-full h-[136px] rounded border block select-none ${
          variant === 'original' && isSelectingNoise
            ? 'border-amber-500 dark:border-amber-400 cursor-col-resize'
            : 'border-slate-200 dark:border-slate-800 cursor-crosshair'
        }`}
        aria-label={`${title} waveform visualization`}
      />
    </div>
  );
}

/**
 * Single high-DPI FFT Spectrogram Canvas using offscreen ImageData for speed.
 */
function SpectrogramCanvas({
  title,
  analysis,
  currentTime,
  isDark,
  onSeek,
}) {
  const canvasRef = useRef(null);
  const resizeTick = useResizeTick(canvasRef);
  const imgCacheRef = useRef({ key: null, offscreen: null });
  const [hoverInfo, setHoverInfo] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 400;
    const height = canvas.clientHeight || 152;

    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);

    ctx.fillStyle = isDark ? '#0B101B' : '#F8FAFC';
    ctx.fillRect(0, 0, width, height);

    if (!analysis?.spectrogram) {
      ctx.fillStyle = isDark ? '#64748B' : '#94A3B8';
      ctx.font = '12px "IBM Plex Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting audio data for FFT spectrogram', width / 2, height / 2 + 4);
      ctx.restore();
      return;
    }

    const spec = analysis.spectrogram;
    const cacheKey = `${spec.frames}_${spec.bins}_${isDark}_${spec.data[0]}_${spec.data[spec.data.length - 1]}`;

    // Cache rendered spectrogram pixels so playhead animation costs almost zero CPU
    let offscreen = imgCacheRef.current.offscreen;
    if (imgCacheRef.current.key !== cacheKey) {
      offscreen = document.createElement('canvas');
      offscreen.width = spec.frames;
      offscreen.height = spec.bins;
      const octx = offscreen.getContext('2d');
      const imgData = octx.createImageData(spec.frames, spec.bins);
      const px = imgData.data;
      const range = spec.maxDb - spec.minDb;

      for (let f = 0; f < spec.frames; f++) {
        const frameOff = f * spec.bins;
        for (let b = 0; b < spec.bins; b++) {
          const db = spec.data[frameOff + b];
          const norm = (db - spec.minDb) / range;
          const [r, g, bl] = spectrogramColor(norm, isDark);
          // Flip vertically so low frequencies sit at the bottom
          const y = spec.bins - 1 - b;
          const idx = (y * spec.frames + f) * 4;
          px[idx] = r;
          px[idx + 1] = g;
          px[idx + 2] = bl;
          px[idx + 3] = 255;
        }
      }
      octx.putImageData(imgData, 0, 0);
      imgCacheRef.current = { key: cacheKey, offscreen };
    }

    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(offscreen, 0, 0, width, height);

    // Frequency grid lines & labels
    const maxKhz = spec.maxFreqHz / 1000;
    const freqMarks = [0.2, 1, 2, 4, 8].filter((k) => k < maxKhz);
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.textAlign = 'left';

    freqMarks.forEach((khz) => {
      const y = height - (khz / maxKhz) * height;
      ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.14)' : 'rgba(15, 23, 42, 0.14)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, Math.round(y) + 0.5);
      ctx.lineTo(width, Math.round(y) + 0.5);
      ctx.stroke();

      ctx.fillStyle = isDark ? 'rgba(241, 245, 249, 0.85)' : 'rgba(15, 23, 42, 0.85)';
      const label = khz < 1 ? `${Math.round(khz * 1000)} Hz` : `${khz} kHz`;
      ctx.fillText(label, 6, Math.max(11, y - 3));
    });

    // Playhead
    if (analysis.duration > 0) {
      const playX = Math.max(
        0,
        Math.min(width - 1, (currentTime / analysis.duration) * width)
      );
      ctx.strokeStyle = isDark ? '#F59E0B' : '#D97706';
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.moveTo(playX + 0.5, 0);
      ctx.lineTo(playX + 0.5, height);
      ctx.stroke();
    }

    ctx.restore();
  }, [analysis, currentTime, isDark, resizeTick]);

  const handleClick = (e) => {
    if (!analysis?.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(ratio * analysis.duration);
  };

  const handleMouseMove = (e) => {
    if (!analysis?.spectrogram) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const rx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const ry = Math.max(0, Math.min(1, 1 - (e.clientY - rect.top) / rect.height));
    const freqHz = ry * analysis.spectrogram.maxFreqHz;
    setHoverInfo({
      time: rx * analysis.duration,
      freqLabel:
        freqHz >= 1000 ? `${(freqHz / 1000).toFixed(2)} kHz` : `${Math.round(freqHz)} Hz`,
    });
  };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-slate-700 dark:text-slate-200">{title}</span>
        <span className="font-mono tabular-nums text-[11px] text-slate-500 dark:text-slate-400">
          {hoverInfo
            ? `${hoverInfo.time.toFixed(2)} s · ${hoverInfo.freqLabel}`
            : analysis?.spectrogram
              ? `0 Hz – ${(analysis.spectrogram.maxFreqHz / 1000).toFixed(1)} kHz`
              : '—'}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverInfo(null)}
        className="w-full h-[152px] rounded border border-slate-200 dark:border-slate-800 cursor-crosshair block"
        aria-label={`${title} spectrogram visualization`}
      />
    </div>
  );
}

/**
 * Synchronized Short-Term LUFS Loudness-Over-Time Canvas plotting Original
 * (dashed slate) and Processed (solid sky/cyan) with target reference line
 * and interactive crosshair readout.
 */
function LoudnessTimelineCanvas({
  originalAnalysis,
  processedAnalysis,
  targetLoudness,
  normalizeEnabled,
  normalizeMode,
  currentTime,
  isDark,
  onSeek,
}) {
  const canvasRef = useRef(null);
  const resizeTick = useResizeTick(canvasRef);
  const [probe, setProbe] = useState(null);

  const maxDuration = Math.max(
    originalAnalysis?.duration || 0,
    processedAnalysis?.duration || 0
  );

  const minLufs = -55;
  const maxLufs = -5;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 720;
    const height = canvas.clientHeight || 176;

    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);

    const padL = 48;
    const padR = 16;
    const padT = 14;
    const padB = 24;
    const plotW = Math.max(10, width - padL - padR);
    const plotH = Math.max(10, height - padT - padB);

    ctx.fillStyle = isDark ? '#0B101B' : '#F8FAFC';
    ctx.fillRect(0, 0, width, height);

    const yForLufs = (val) => {
      const clamped = Math.max(minLufs, Math.min(maxLufs, val));
      const norm = (clamped - minLufs) / (maxLufs - minLufs);
      return padT + (1 - norm) * plotH;
    };

    const xForTime = (t) => {
      if (maxDuration <= 0) return padL;
      return padL + (t / maxDuration) * plotW;
    };

    // Horizontal LUFS grid
    const gridLevels = [-10, -16, -23, -35, -50];
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.textAlign = 'right';

    gridLevels.forEach((lvl) => {
      const y = Math.round(yForLufs(lvl)) + 0.5;
      ctx.strokeStyle = isDark ? 'rgba(148, 163, 184, 0.14)' : 'rgba(15, 23, 42, 0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();

      ctx.fillStyle = isDark ? '#94A3B8' : '#64748B';
      ctx.fillText(`${lvl}`, padL - 8, y + 3);
    });

    if (!originalAnalysis?.shortTermSeries?.length) {
      ctx.fillStyle = isDark ? '#64748B' : '#94A3B8';
      ctx.font = '12px "IBM Plex Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Load an audio file to plot short-term LUFS loudness over time', width / 2, height / 2 + 4);
      ctx.restore();
      return;
    }

    // Target loudness guide line
    if (normalizeEnabled && Number.isFinite(targetLoudness)) {
      const ty = Math.round(yForLufs(targetLoudness)) + 0.5;
      ctx.strokeStyle = isDark ? 'rgba(16, 185, 129, 0.55)' : 'rgba(5, 150, 105, 0.55)';
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, ty);
      ctx.lineTo(width - padR, ty);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = isDark ? '#34D399' : '#059669';
      ctx.textAlign = 'right';
      ctx.fillText(
        `Target ${targetLoudness} ${normalizeMode === 'rms' ? 'RMS' : 'LUFS'}`,
        width - padR - 4,
        ty - 4
      );
    }

    // Helper to draw a curve
    const drawCurve = (series, strokeStyle, dashed, fillAlpha) => {
      if (!series || series.length < 2) return;

      if (fillAlpha) {
        ctx.beginPath();
        ctx.moveTo(xForTime(series[0].time), padT + plotH);
        series.forEach((pt) => {
          ctx.lineTo(xForTime(pt.time), yForLufs(pt.lufs));
        });
        ctx.lineTo(xForTime(series[series.length - 1].time), padT + plotH);
        ctx.closePath();
        ctx.fillStyle = fillAlpha;
        ctx.fill();
      }

      ctx.beginPath();
      series.forEach((pt, idx) => {
        const x = xForTime(pt.time);
        const y = yForLufs(pt.lufs);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = dashed ? 1.5 : 2;
      ctx.setLineDash(dashed ? [4, 3] : []);
      ctx.stroke();
      ctx.setLineDash([]);
    };

    // Original series (dashed slate)
    drawCurve(
      originalAnalysis.shortTermSeries,
      isDark ? '#94A3B8' : '#64748B',
      true,
      null
    );

    // Processed series (solid calibrated cyan/sky)
    if (processedAnalysis?.shortTermSeries?.length) {
      drawCurve(
        processedAnalysis.shortTermSeries,
        isDark ? '#06B6D4' : '#0284C7',
        false,
        isDark ? 'rgba(6, 182, 212, 0.12)' : 'rgba(2, 132, 199, 0.10)'
      );
    }

    // Time ticks along bottom
    const tickCount = 5;
    ctx.fillStyle = isDark ? '#94A3B8' : '#64748B';
    ctx.textAlign = 'center';
    for (let i = 0; i <= tickCount; i++) {
      const t = (i / tickCount) * maxDuration;
      const x = xForTime(t);
      ctx.fillText(`${t.toFixed(1)}s`, x, height - 7);
    }

    // Synchronized playhead
    if (maxDuration > 0) {
      const px = xForTime(Math.min(currentTime, maxDuration));
      ctx.strokeStyle = isDark ? '#F59E0B' : '#D97706';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px + 0.5, padT);
      ctx.lineTo(px + 0.5, padT + plotH);
      ctx.stroke();
    }

    // Hover crosshair
    if (probe) {
      const hx = xForTime(probe.time);
      ctx.strokeStyle = isDark ? 'rgba(226, 232, 240, 0.4)' : 'rgba(15, 23, 42, 0.3)';
      ctx.setLineDash([2, 2]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx + 0.5, padT);
      ctx.lineTo(hx + 0.5, padT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.restore();
  }, [
    originalAnalysis,
    processedAnalysis,
    targetLoudness,
    normalizeEnabled,
    normalizeMode,
    currentTime,
    isDark,
    maxDuration,
    probe,
    resizeTick,
  ]);

  const lookupNearest = (series, t) => {
    if (!series || series.length === 0) return null;
    let best = series[0];
    let bestDist = Math.abs(series[0].time - t);
    for (let i = 1; i < series.length; i++) {
      const d = Math.abs(series[i].time - t);
      if (d < bestDist) {
        best = series[i];
        bestDist = d;
      }
    }
    return best.lufs;
  };

  const handleMouseMove = (e) => {
    if (maxDuration <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const padL = 48;
    const padR = 16;
    const plotW = Math.max(10, rect.width - padL - padR);
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left - padL) / plotW));
    const time = ratio * maxDuration;
    setProbe({
      time,
      origLufs: lookupNearest(originalAnalysis?.shortTermSeries, time),
      procLufs: lookupNearest(processedAnalysis?.shortTermSeries, time),
    });
  };

  const handleClick = (e) => {
    if (maxDuration <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const padL = 48;
    const padR = 16;
    const plotW = Math.max(10, rect.width - padL - padR);
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left - padL) / plotW));
    onSeek(ratio * maxDuration);
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-4">
          <span className="font-medium text-slate-800 dark:text-slate-100">
            Loudness Over Time (Short-Term{' '}
            {originalAnalysis?.shortTermWindowSec
              ? `${originalAnalysis.shortTermWindowSec.toFixed(0)} s`
              : '3 s'}{' '}
            LUFS)
          </span>
          <div className="flex items-center gap-3 text-[11px] text-slate-500 dark:text-slate-400">
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block w-4 border-t-2 border-dashed border-slate-500 dark:border-slate-400" />
              Original
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block w-4 border-t-2 border-sky-600 dark:border-cyan-400" />
              Processed
            </span>
          </div>
        </div>

        <span className="font-mono tabular-nums text-[11px] text-slate-600 dark:text-slate-300">
          {probe
            ? `t=${probe.time.toFixed(2)} s · Orig: ${
                probe.origLufs != null ? `${probe.origLufs.toFixed(1)} LUFS` : '—'
              } · Proc: ${
                probe.procLufs != null ? `${probe.procLufs.toFixed(1)} LUFS` : '—'
              }`
            : 'Hover or click graph to inspect & seek'}
        </span>
      </div>

      <canvas
        ref={canvasRef}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setProbe(null)}
        className="w-full h-[176px] rounded border border-slate-200 dark:border-slate-800 cursor-crosshair block"
        aria-label="Short-term LUFS loudness over time graph"
      />
    </div>
  );
}

export default function Visualizations({
  originalAnalysis,
  processedAnalysis,
  activeSource,
  currentTime,
  options,
  theme,
  onSeek,
  noiseRegion,
  isSelectingNoise,
  onSelectNoiseRegion,
  onToggleSelectingNoise,
}) {
  const isDark = theme === 'dark';

  return (
    <section
      id="visualizers"
      aria-label="Audio visualizations"
      className="bg-white dark:bg-[#111827] border border-slate-200 dark:border-slate-800 rounded-lg divide-y divide-slate-200 dark:divide-slate-800"
    >
      {/* 1. Side-by-side Waveforms */}
      <div className="p-4 space-y-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Waveforms (Side by Side)
          </h2>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            Outer shade: peak amplitude · Solid core: RMS energy · Click to seek · Shift+drag Original for noise sample
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <WaveformCanvas
            title="Original (Before)"
            analysis={originalAnalysis}
            variant="original"
            currentTime={currentTime}
            isActiveSource={activeSource === 'original'}
            isDark={isDark}
            onSeek={onSeek}
            noiseRegion={noiseRegion}
            isSelectingNoise={isSelectingNoise}
            onSelectNoiseRegion={onSelectNoiseRegion}
            onToggleSelectingNoise={onToggleSelectingNoise}
          />
          <WaveformCanvas
            title="Processed (After)"
            analysis={processedAnalysis}
            variant="processed"
            currentTime={currentTime}
            isActiveSource={activeSource === 'processed'}
            isDark={isDark}
            onSeek={onSeek}
          />
        </div>
      </div>

      {/* 2. Side-by-side FFT Spectrograms */}
      <div className="p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              FFT Spectrograms (Side by Side)
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Compare low-end rumble (bottom band) and background hiss between words before and
              after filtering and gating.
            </p>
          </div>

          {/* Calibrated dB color scale legend */}
          <div className="flex items-center gap-2 text-[11px] font-mono tabular-nums text-slate-500 dark:text-slate-400">
            <span>-90 dB</span>
            <div
              aria-hidden="true"
              className="w-24 h-2.5 rounded-xs border border-slate-300 dark:border-slate-700"
              style={{
                background: isDark
                  ? 'linear-gradient(90deg, #090D18 0%, #172550 28%, #08A2C3 56%, #F5B119 84%, #FFFCEB 100%)'
                  : 'linear-gradient(90deg, #F8FAFC 0%, #D6E4F4 28%, #1384C6 56%, #EA7010 84%, #BD252E 100%)',
              }}
            />
            <span>-6 dB</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SpectrogramCanvas
            title="Original Spectrum"
            analysis={originalAnalysis}
            currentTime={currentTime}
            isDark={isDark}
            onSeek={onSeek}
          />
          <SpectrogramCanvas
            title="Processed Spectrum"
            analysis={processedAnalysis}
            currentTime={currentTime}
            isDark={isDark}
            onSeek={onSeek}
          />
        </div>
      </div>

      {/* 3. Loudness Over Time (Short-Term LUFS) */}
      <div className="p-4">
        <LoudnessTimelineCanvas
          originalAnalysis={originalAnalysis}
          processedAnalysis={processedAnalysis}
          targetLoudness={options.targetLoudness}
          normalizeEnabled={options.normalizeEnabled}
          normalizeMode={options.normalizeMode}
          currentTime={currentTime}
          isDark={isDark}
          onSeek={onSeek}
        />
      </div>
    </section>
  );
}
