import { formatTimestamp, titleCase } from "@/lib/format";

export const speakerOptions = ["Operator", "Customer", "Other", "Unknown"] as const;
export function channelLabel(channel: number | null) {
  return channel === 0 ? "Channel A / 0" : channel === 1 ? "Channel B / 1" : "None";
}

export function SpeakerLabel({ label, source, timestamp, onSeek }: {
  label?: string; source?: string; timestamp: number; onSeek: (seconds: number) => void;
}) {
  return <div className="speaker"><strong>{label || "Unknown speaker"}</strong>
    <span>{titleCase(source)}</span><br />
    <button type="button" className="text-button" onClick={() => onSeek(timestamp)}>{formatTimestamp(timestamp)}</button>
  </div>;
}
