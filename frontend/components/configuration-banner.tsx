import type { Configuration } from "@/lib/types";
import { configurationReady } from "@/lib/format";

function normalized(status?: string): string {
  return (status || "").toLowerCase().replaceAll("_", " ");
}

export function ConfigurationBanner({ configuration }: { configuration: Configuration }) {
  const yeastarReady = configurationReady(configuration.yeastar.status);
  const openAiReady = configurationReady(configuration.openai.status);
  if (yeastarReady && openAiReady) return null;

  const yeastarMissing = normalized(configuration.yeastar.status).includes("not configured") || normalized(configuration.yeastar.status).includes("missing");
  const openAiMissing = normalized(configuration.openai.status).includes("not configured") || normalized(configuration.openai.status).includes("missing");
  const phoneMessage = yeastarStatusMessage(configuration.yeastar.status);

  return (
    <div className="configuration-banner" role="status">
      {!yeastarReady ? (
        <div>
          <strong>{phoneMessage}</strong>
          <p>{yeastarMissing ? "Add the connection details in Settings, then use Test connection." : "Resolve the phone-system issue, then use Test connection in Settings before starting an analysis."}</p>
        </div>
      ) : null}
      {!openAiReady ? (
        <div>
          <strong>{openAiMissing ? "OpenAI transcription is not configured." : "The transcription service is unavailable."}</strong>
          <p>{openAiMissing ? "Add an OpenAI API key in Settings, then test the connection." : "Check the OpenAI connection in Settings before starting an analysis."}</p>
        </div>
      ) : null}
    </div>
  );
}

function yeastarStatusMessage(status?: string): string {
  const value = normalized(status);
  if (value.includes("not configured") || value.includes("missing")) return "Phone system not configured";
  if (value === "not tested") return "Connection has not been tested";
  if (value === "auth rejected") return "The connection details were rejected";
  if (value === "token refresh failed" || value === "connection paused") return "Connection attempts have been paused for safety";
  if (value === "ip not allowed") return "This server is not permitted to access the phone system";
  if (value === "ip blocked") return "The phone system has blocked this server";
  if (value === "unsupported api version" || value === "unsupported firmware") return "The installed phone-system version is not supported";
  return "Could not reach the phone system";
}
