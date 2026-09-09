"use client";

import { useCallback, useRef, useState, type RefObject } from "react";
import { formatTimestamp } from "@/lib/format";

export function useRecordingPlayer() {
  const audioRef = useRef<HTMLAudioElement>(null);
  const seek = useCallback((seconds: number) => {
    const audio = audioRef.current;
    if (!audio || !Number.isFinite(seconds)) return;
    audio.currentTime = Math.max(0, Math.min(seconds, Number.isFinite(audio.duration) ? audio.duration : seconds));
    void audio.play().catch(() => undefined);
  }, []);
  return { audioRef, seek };
}

type Source = { label: string; url: string };

export function RecordingPlayer({ audioRef, src, sources, errorMessage }: {
  audioRef: RefObject<HTMLAudioElement | null>;
  src: string;
  sources?: Source[];
  errorMessage?: string;
}) {
  const [selected, setSelected] = useState(src);
  const [position, setPosition] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [error, setError] = useState("");
  const pending = useRef<{ position: number; playing: boolean } | null>(null);
  const activeSrc = sources?.some((source) => source.url === selected) ? selected : src;

  function switchSource(url: string) {
    if (url === activeSrc) return;
    const audio = audioRef.current;
    pending.current = { position: audio?.currentTime || 0, playing: audio ? !audio.paused : false };
    setError("");
    setSelected(url);
  }

  function skip(delta: number) {
    const audio = audioRef.current;
    if (!audio) return;
    const end = Number.isFinite(audio.duration) ? audio.duration : Infinity;
    audio.currentTime = Math.max(0, Math.min(end, audio.currentTime + delta));
    setPosition(audio.currentTime);
  }

  return <>
    {sources && sources.length > 1 ? <div className="page-actions" role="group" aria-label="Listening channel">
      {sources.map((source) => <button key={source.url} type="button"
        className={source.url === activeSrc ? "button" : "button secondary"}
        aria-pressed={source.url === activeSrc} onClick={() => switchSource(source.url)}>{source.label}</button>)}
    </div> : null}
    <audio ref={audioRef} className="audio-player" controls preload="metadata" src={activeSrc}
      onTimeUpdate={(event) => setPosition(event.currentTarget.currentTime)}
      onLoadedMetadata={(event) => {
        const audio = event.currentTarget;
        audio.playbackRate = speed;
        if (pending.current) {
          audio.currentTime = Math.min(pending.current.position, audio.duration);
          if (pending.current.playing) void audio.play().catch(() => undefined);
          pending.current = null;
        }
        setPosition(audio.currentTime);
        setError("");
      }}
      onError={() => setError(errorMessage || "The recording could not be played. Check your session and the local audio file.")}>
      Your browser does not support audio playback.
    </audio>
    <div className="page-actions audio-controls">
      <output aria-label="Player position">{formatTimestamp(position)}</output>
      {[-5, -2, 2, 5].map((delta) => <button type="button" className="button secondary compact"
        key={delta} onClick={() => skip(delta)} aria-label={`Seek ${delta > 0 ? "forward" : "back"} ${Math.abs(delta)} seconds`}>
        {delta > 0 ? "+" : "−"}{Math.abs(delta)}s
      </button>)}
      <label className="field">Playback speed
        <select value={speed} onChange={(event) => {
          const value = Number(event.target.value);
          setSpeed(value);
          if (audioRef.current) audioRef.current.playbackRate = value;
        }}>{[0.75, 1, 1.25, 1.5].map((value) => <option key={value} value={value}>{value}x</option>)}</select>
      </label>
    </div>
    {error ? <div className="form-error" role="alert">{error}</div> : null}
  </>;
}
