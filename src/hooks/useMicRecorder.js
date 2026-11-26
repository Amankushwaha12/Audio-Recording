/**
 * useMicRecorder.js — Live Browser Microphone Capture Hook (Web Audio API)
 *
 * Captures raw, uncompressed 32-bit float PCM audio directly from the user's
 * microphone via MediaStreamAudioSourceNode so recordings can be analyzed,
 * mastered, A/B compared, and exported as 16-bit WAV with zero codec loss
 * or browser container decoding incompatibilities.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

const MAX_RECORDING_SEC = 300; // 5-minute safety guard for in-browser PCM memory

export function useMicRecorder({ getAudioContext, onRecordingComplete, onError }) {
  const [isRecording, setIsRecording] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [inputLevelDb, setInputLevelDb] = useState(-60);

  const streamRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const analyserRef = useRef(null);
  const processorRef = useRef(null);
  const muteGainRef = useRef(null);
  const chunksLeftRef = useRef([]);
  const chunksRightRef = useRef([]);
  const channelCountRef = useRef(1);
  const sampleRateRef = useRef(44100);
  const startTimeRef = useRef(0);
  const rafRef = useRef(null);
  const isRecordingRef = useRef(false);

  const cleanupNodes = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.onaudioprocess = null;
      try {
        processorRef.current.disconnect();
      } catch {
        // Ignore disconnect errors
      }
      processorRef.current = null;
    }
    if (analyserRef.current) {
      try {
        analyserRef.current.disconnect();
      } catch {
        // Ignore
      }
      analyserRef.current = null;
    }
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.disconnect();
      } catch {
        // Ignore
      }
      sourceNodeRef.current = null;
    }
    if (muteGainRef.current) {
      try {
        muteGainRef.current.disconnect();
      } catch {
        // Ignore
      }
      muteGainRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, []);

  const stopRecording = useCallback(
    (discard = false) => {
      if (!isRecordingRef.current) return;
      isRecordingRef.current = false;
      setIsRecording(false);
      setInputLevelDb(-60);

      const leftChunks = chunksLeftRef.current;
      const rightChunks = chunksRightRef.current;
      const chCount = channelCountRef.current;
      const sampleRate = sampleRateRef.current;

      cleanupNodes();
      chunksLeftRef.current = [];
      chunksRightRef.current = [];
      setElapsedSec(0);

      if (discard) return;

      const totalSamples = leftChunks.reduce((sum, arr) => sum + arr.length, 0);
      if (totalSamples < Math.round(sampleRate * 0.1)) {
        onError?.('Recording was too short. Please record at least a fraction of a second of audio.');
        return;
      }

      try {
        const ctx = getAudioContext();
        const outChannels = chCount >= 2 && rightChunks.length === leftChunks.length ? 2 : 1;
        const audioBuffer = ctx.createBuffer(outChannels, totalSamples, sampleRate);

        const leftOut = audioBuffer.getChannelData(0);
        let offset = 0;
        for (let i = 0; i < leftChunks.length; i++) {
          leftOut.set(leftChunks[i], offset);
          offset += leftChunks[i].length;
        }

        if (outChannels === 2) {
          const rightOut = audioBuffer.getChannelData(1);
          offset = 0;
          for (let i = 0; i < rightChunks.length; i++) {
            rightOut.set(rightChunks[i], offset);
            offset += rightChunks[i].length;
          }
        }

        onRecordingComplete?.(audioBuffer);
      } catch (err) {
        console.error('Failed to assemble recorded AudioBuffer:', err);
        onError?.('Could not assemble recorded audio buffer. Please try recording again.');
      }
    },
    [cleanupNodes, getAudioContext, onRecordingComplete, onError]
  );

  const startRecording = useCallback(async () => {
    if (isRecordingRef.current) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      onError?.(
        'Microphone recording is not supported in this browser or requires a secure (HTTPS / localhost) context.'
      );
      return;
    }

    try {
      // Request raw mic audio (disable browser AGC/denoising so our DSP chain has real work to demonstrate)
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          },
        });
      } catch {
        // Fallback for devices/browsers that reject advanced audio constraints
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }

      const ctx = getAudioContext();
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.6;

      // Capture raw Float32 PCM chunks reliably across all browsers
      const inputChannels = Math.min(2, Math.max(1, source.channelCount || 1));
      const processor = ctx.createScriptProcessor(4096, inputChannels, 1);

      // Connect through a silent GainNode (gain = 0) so the graph pulls audio without speaker feedback
      const muteGain = ctx.createGain();
      muteGain.gain.value = 0;

      chunksLeftRef.current = [];
      chunksRightRef.current = [];
      channelCountRef.current = inputChannels;
      sampleRateRef.current = ctx.sampleRate;

      processor.onaudioprocess = (event) => {
        if (!isRecordingRef.current) return;
        const input = event.inputBuffer;
        const ch0 = input.getChannelData(0);
        chunksLeftRef.current.push(new Float32Array(ch0));
        if (inputChannels >= 2 && input.numberOfChannels >= 2) {
          chunksRightRef.current.push(new Float32Array(input.getChannelData(1)));
        }
      };

      source.connect(analyser);
      analyser.connect(processor);
      processor.connect(muteGain);
      muteGain.connect(ctx.destination);

      streamRef.current = stream;
      sourceNodeRef.current = source;
      analyserRef.current = analyser;
      processorRef.current = processor;
      muteGainRef.current = muteGain;

      startTimeRef.current = performance.now();
      isRecordingRef.current = true;
      setIsRecording(true);
      setElapsedSec(0);

      const timeDomain = new Float32Array(analyser.fftSize);
      const updateMeter = () => {
        if (!isRecordingRef.current) return;
        const elapsed = (performance.now() - startTimeRef.current) / 1000;
        setElapsedSec(elapsed);

        if (elapsed >= MAX_RECORDING_SEC) {
          stopRecording(false);
          return;
        }

        analyser.getFloatTimeDomainData(timeDomain);
        let sumSq = 0;
        for (let i = 0; i < timeDomain.length; i++) {
          sumSq += timeDomain[i] * timeDomain[i];
        }
        const rms = Math.sqrt(sumSq / timeDomain.length);
        const db = rms > 1e-5 ? Math.max(-60, Math.min(0, 20 * Math.log10(rms))) : -60;
        setInputLevelDb(db);

        rafRef.current = requestAnimationFrame(updateMeter);
      };
      rafRef.current = requestAnimationFrame(updateMeter);
    } catch (err) {
      cleanupNodes();
      isRecordingRef.current = false;
      setIsRecording(false);

      if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
        onError?.(
          'Microphone access was denied. Please allow microphone permission in your browser settings and try again.'
        );
      } else if (err?.name === 'NotFoundError' || err?.name === 'DevicesNotFoundError') {
        onError?.('No microphone was found on this device. Connect a mic or upload an audio file instead.');
      } else {
        onError?.(`Could not start microphone recording: ${err?.message || 'Unknown audio device error'}.`);
      }
    }
  }, [cleanupNodes, getAudioContext, onError, stopRecording]);

  useEffect(() => {
    return () => {
      isRecordingRef.current = false;
      cleanupNodes();
    };
  }, [cleanupNodes]);

  return {
    isRecording,
    elapsedSec,
    inputLevelDb,
    startRecording,
    stopRecording,
  };
}
