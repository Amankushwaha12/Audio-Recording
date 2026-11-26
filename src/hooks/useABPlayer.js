/**
 * useABPlayer.js — Synchronized Web Audio API A/B Playback Hook
 *
 * Manages a shared seek position between the Original and Processed AudioBuffers
 * so the user can toggle back and forth instantaneously without losing their
 * place in the recording. Uses a tiny 12 ms gain ramp on start/stop to eliminate
 * audible switching clicks.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export function useABPlayer(originalBuffer, processedBuffer) {
  const [activeSource, setActiveSource] = useState('original'); // 'original' | 'processed'
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const audioCtxRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const gainNodeRef = useRef(null);
  const rafRef = useRef(null);

  // Playback timing anchors
  const startCtxTimeRef = useRef(0);
  const startOffsetRef = useRef(0);
  const activeSourceRef = useRef('original');
  const isPlayingRef = useRef(false);
  const currentTimeRef = useRef(0);

  const getAudioContext = useCallback(() => {
    if (!audioCtxRef.current) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      audioCtxRef.current = new AudioCtx();
    }
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }
    return audioCtxRef.current;
  }, []);

  const stopNodeOnly = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    const ctx = audioCtxRef.current;
    const node = sourceNodeRef.current;
    const gain = gainNodeRef.current;
    sourceNodeRef.current = null;
    gainNodeRef.current = null;

    if (node) {
      node.onended = null;
      try {
        if (gain && ctx) {
          const now = ctx.currentTime;
          gain.gain.cancelScheduledValues(now);
          gain.gain.setValueAtTime(gain.gain.value, now);
          gain.gain.linearRampToValueAtTime(0.0001, now + 0.012);
          node.stop(now + 0.015);
        } else {
          node.stop();
        }
      } catch {
        // Node may already have stopped
      }
    }
  }, []);

  const getLiveTime = useCallback(() => {
    if (!isPlayingRef.current || !audioCtxRef.current) {
      return currentTimeRef.current;
    }
    const elapsed = audioCtxRef.current.currentTime - startCtxTimeRef.current;
    return Math.max(0, startOffsetRef.current + elapsed);
  }, []);

  const startPlayback = useCallback(
    (which, offsetSec) => {
      const buffer = which === 'processed' ? processedBuffer : originalBuffer;
      if (!buffer) return;

      stopNodeOnly();
      const ctx = getAudioContext();

      // Clamp seek position inside the chosen buffer (handles silence-trimmed tracks)
      const safeOffset =
        offsetSec >= buffer.duration - 0.02
          ? 0
          : Math.max(0, Math.min(offsetSec, Math.max(0, buffer.duration - 0.01)));

      const source = ctx.createBufferSource();
      source.buffer = buffer;

      const gain = ctx.createGain();
      const now = ctx.currentTime;
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.linearRampToValueAtTime(1.0, now + 0.012);

      source.connect(gain);
      gain.connect(ctx.destination);

      sourceNodeRef.current = source;
      gainNodeRef.current = gain;
      startCtxTimeRef.current = now;
      startOffsetRef.current = safeOffset;
      activeSourceRef.current = which;
      isPlayingRef.current = true;
      currentTimeRef.current = safeOffset;

      setActiveSource(which);
      setIsPlaying(true);
      setCurrentTime(safeOffset);

      source.onended = () => {
        if (sourceNodeRef.current === source) {
          stopNodeOnly();
          isPlayingRef.current = false;
          currentTimeRef.current = 0;
          setIsPlaying(false);
          setCurrentTime(0);
        }
      };

      source.start(now, safeOffset);

      const tick = () => {
        if (!isPlayingRef.current || sourceNodeRef.current !== source) return;
        const pos = startOffsetRef.current + (ctx.currentTime - startCtxTimeRef.current);
        const clamped = Math.min(buffer.duration, Math.max(0, pos));
        currentTimeRef.current = clamped;
        setCurrentTime(clamped);
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    },
    [originalBuffer, processedBuffer, getAudioContext, stopNodeOnly]
  );

  const pause = useCallback(() => {
    const pos = getLiveTime();
    stopNodeOnly();
    isPlayingRef.current = false;
    currentTimeRef.current = pos;
    setIsPlaying(false);
    setCurrentTime(pos);
  }, [getLiveTime, stopNodeOnly]);

  const togglePlayTrack = useCallback(
    (which) => {
      const targetBuffer = which === 'processed' ? processedBuffer : originalBuffer;
      if (!targetBuffer) return;

      if (isPlayingRef.current && activeSourceRef.current === which) {
        pause();
      } else {
        const pos = getLiveTime();
        startPlayback(which, pos);
      }
    },
    [originalBuffer, processedBuffer, pause, getLiveTime, startPlayback]
  );

  const switchSource = useCallback(
    (which) => {
      const targetBuffer = which === 'processed' ? processedBuffer : originalBuffer;
      if (!targetBuffer) return;
      const pos = getLiveTime();
      const clampedPos = Math.min(pos, Math.max(0, targetBuffer.duration - 0.01));

      if (isPlayingRef.current) {
        startPlayback(which, clampedPos);
      } else {
        activeSourceRef.current = which;
        currentTimeRef.current = clampedPos;
        setActiveSource(which);
        setCurrentTime(clampedPos);
      }
    },
    [originalBuffer, processedBuffer, getLiveTime, startPlayback]
  );

  const seek = useCallback(
    (newTimeSec) => {
      const buffer =
        activeSourceRef.current === 'processed' && processedBuffer
          ? processedBuffer
          : originalBuffer;
      if (!buffer) return;
      const clamped = Math.max(0, Math.min(buffer.duration, newTimeSec));
      currentTimeRef.current = clamped;
      setCurrentTime(clamped);
      if (isPlayingRef.current) {
        startPlayback(activeSourceRef.current, clamped);
      }
    },
    [originalBuffer, processedBuffer, startPlayback]
  );

  // Stop & reset when a new original file is loaded
  useEffect(() => {
    stopNodeOnly();
    isPlayingRef.current = false;
    activeSourceRef.current = 'original';
    currentTimeRef.current = 0;
    setIsPlaying(false);
    setActiveSource('original');
    setCurrentTime(0);
  }, [originalBuffer, stopNodeOnly]);

  // If processed buffer is replaced while playing processed, stop cleanly
  useEffect(() => {
    if (!processedBuffer && activeSourceRef.current === 'processed') {
      stopNodeOnly();
      isPlayingRef.current = false;
      activeSourceRef.current = 'original';
      setIsPlaying(false);
      setActiveSource('original');
    }
  }, [processedBuffer, stopNodeOnly]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopNodeOnly();
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
      }
    };
  }, [stopNodeOnly]);

  return {
    activeSource,
    isPlaying,
    currentTime,
    togglePlayTrack,
    switchSource,
    pause,
    seek,
    getAudioContext,
  };
}
