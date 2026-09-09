const DEFAULT_TIMEZONE = "Europe/Athens";
let activeTimezone = DEFAULT_TIMEZONE;

export function setActiveTimezone(value?: string | null): void {
  if (!value) return;
  try {
    new Intl.DateTimeFormat("en-GB", { timeZone: value }).format(new Date());
    activeTimezone = value;
  } catch {
    activeTimezone = DEFAULT_TIMEZONE;
  }
}

function resolvedTimezone(value?: string): string {
  return value || activeTimezone;
}

export function formatDateTime(value?: string | null, timeZone?: string): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: resolvedTimezone(timeZone),
  }).format(date);
}

export function formatDate(value?: string | null, timeZone?: string): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("en-GB", { dateStyle: "medium", timeZone: resolvedTimezone(timeZone) }).format(date);
}

export function formatTime(value?: string | null, timeZone?: string): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("en-GB", { timeStyle: "short", timeZone: resolvedTimezone(timeZone) }).format(date);
}

export function formatDuration(seconds?: number | null): string {
  if (seconds === undefined || seconds === null || !Number.isFinite(seconds)) return "—";
  const value = Math.max(0, Math.round(seconds));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const remaining = value % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`
    : `${minutes}:${String(remaining).padStart(2, "0")}`;
}

export function formatTimestamp(seconds?: number | null): string {
  if (seconds === undefined || seconds === null || !Number.isFinite(seconds)) return "0:00";
  return formatDuration(seconds);
}

export function titleCase(value?: string | null): string {
  if (!value) return "—";
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function toIsoDateTime(localValue: string, timeZone?: string): string {
  timeZone = resolvedTimezone(timeZone);
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?$/.exec(localValue);
  if (!match) throw new Error("Enter a valid date and time.");
  const requested = {
    year: Number(match[1]),
    month: Number(match[2]),
    day: Number(match[3]),
    hour: Number(match[4]),
    minute: Number(match[5]),
    second: Number(match[6] || 0),
  };
  const wallClockUtc = Date.UTC(requested.year, requested.month - 1, requested.day, requested.hour, requested.minute, requested.second);
  let candidate = wallClockUtc;
  try {
    for (let index = 0; index < 4; index += 1) {
      const represented = zonedParts(new Date(candidate), timeZone);
      const representedUtc = Date.UTC(represented.year, represented.month - 1, represented.day, represented.hour, represented.minute, represented.second);
      const next = candidate + (wallClockUtc - representedUtc);
      if (next === candidate) break;
      candidate = next;
    }
    const verified = zonedParts(new Date(candidate), timeZone);
    if (Object.entries(requested).some(([key, value]) => verified[key as keyof typeof verified] !== value)) {
      throw new Error("The selected local time does not exist because of a daylight-saving change. Choose a different time.");
    }
  } catch (error) {
    if (error instanceof Error && error.message.includes("daylight-saving")) throw error;
    throw new Error("The configured timezone is not valid.");
  }
  return new Date(candidate).toISOString();
}

function zonedParts(date: Date, timeZone: string) {
  const parts = new Intl.DateTimeFormat("en-GB", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).formatToParts(date);
  const value = (type: Intl.DateTimeFormatPartTypes) => Number(parts.find((part) => part.type === type)?.value);
  return { year: value("year"), month: value("month"), day: value("day"), hour: value("hour"), minute: value("minute"), second: value("second") };
}

export function statusTone(status?: string): "success" | "warning" | "error" | "neutral" {
  const value = (status || "").toLowerCase().replaceAll("_", " ");
  if (["connected", "ready", "completed", "enabled", "available", "verified"].includes(value)) return "success";
  if (["failed", "unavailable", "error", "cancelled", "disabled"].includes(value)) return "error";
  if (["not configured", "completed with errors", "warning", "in progress"].includes(value)) return "warning";
  return "neutral";
}

export function configurationReady(status?: string): boolean {
  const value = (status || "").toLowerCase().replaceAll("_", " ");
  return value === "connected" || value === "ready" || value === "configured";
}
