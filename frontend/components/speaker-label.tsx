import { formatTimestamp } from "@/lib/format";

export const speakerOptions = ["Operator", "Customer", "Other", "Unknown"] as const;
export function channelLabel(channel: number | null) {
  return channel === 0 ? "Channel A / 0" : channel === 1 ? "Channel B / 1" : "None";
}

export function SpeakerLabel({ label, source, timestamp, onSeek, disabled = false }: {
  label?: string; source?: string; timestamp: number; onSeek: (seconds: number) => void; disabled?: boolean;
}) {
  const anonymous = /^[A-Z]$/.test(label || "");
  const unknown = !label || label === "Unknown" || label === "Unknown speaker";
  const display = unknown ? "Unassigned speaker" : anonymous ? `Speaker ${label}` : label;
  return <div className="speaker" data-speaker={unknown ? "unknown" : label}>
    <span className="speaker-avatar" aria-hidden="true">{unknown ? "?" : anonymous ? label : label?.slice(0, 1)}</span>
    <strong title={source === "openai_diarization" ? "Anonymous voice; identity has not been confirmed" : undefined}>{display}</strong>
    <button type="button" className="text-button" disabled={disabled} onClick={() => onSeek(timestamp)} aria-label={`Play segment from ${formatTimestamp(timestamp)}`}>{formatTimestamp(timestamp)}</button>
  </div>;
}
