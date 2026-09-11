"use client";

import { Fragment, useMemo, useState } from "react";
import { SpeakerLabel } from "@/components/speaker-label";
import { formatTimestamp } from "@/lib/format";
import type { KeywordMatch, TranscriptSegment, WordingReview } from "@/lib/types";

const reviewFlags = new Set([
  "human_review_recommended", "low_confidence", "both_attempts_low_confidence",
  "transcription_disagreement", "speaker_alignment_uncertain", "refinement_failed",
]);
export function segmentNeedsReview(segment: TranscriptSegment) {
  return !segment.speaker_label || segment.speaker_label === "Unknown" ||
    Boolean(segment.quality_flags?.some((flag) => reviewFlags.has(flag)));
}

function fold(text: string) {
  return text.normalize("NFD").replace(/\p{M}/gu, "").toLocaleLowerCase("el-GR").replace(/ς/g, "σ");
}

export function CallTranscript({ segments, matches, reviews, verificationStatus, truncated, position, canSeek, onSeek }: {
  segments: TranscriptSegment[];
  matches: KeywordMatch[];
  reviews: WordingReview[];
  verificationStatus?: string;
  truncated?: boolean;
  position: number;
  canSeek: boolean;
  onSeek: (seconds: number) => void;
}) {
  const [query, setQuery] = useState("");
  const [reviewOnly, setReviewOnly] = useState(false);
  const [copyStatus, setCopyStatus] = useState("");
  const groups = useMemo(() => {
    const rows: Array<{ segments: TranscriptSegment[]; indexes: number[] }> = [];
    segments.forEach((segment, index) => {
      const previous = rows.at(-1);
      const last = previous?.segments.at(-1);
      if (last && last.speaker_label === segment.speaker_label && last.speaker_source === segment.speaker_source &&
        segment.start_timestamp - last.end_timestamp <= 1 && segment.start_timestamp >= last.start_timestamp) {
        previous!.segments.push(segment);
        previous!.indexes.push(index);
      } else rows.push({ segments: [segment], indexes: [index] });
    });
    return rows;
  }, [segments]);
  const reviewCount = groups.filter((group) => group.segments.some(segmentNeedsReview)).length;
  const visible = groups.filter((group) => (!reviewOnly || group.segments.some(segmentNeedsReview)) &&
    (!query.trim() || fold(group.segments.map((segment) => segment.original_text).join(" ")).includes(fold(query.trim()))));
  const approximate = segments.some((segment) => segment.quality_flags?.includes("approximate_timestamps"));

  async function copyTranscript() {
    try {
      await navigator.clipboard.writeText(segments.map((segment) => segment.original_text).join(" "));
      setCopyStatus("Transcript copied");
    } catch { setCopyStatus("Copy failed. Select the transcript text and copy it manually."); }
  }

  return <section className="conversation-panel" aria-labelledby="transcript-title">
    <div className="conversation-heading">
      <div><span className="eyebrow">CALL REVIEW</span><h2 id="transcript-title">Transcript</h2></div>
      <button className="button secondary compact" type="button" onClick={() => void copyTranscript()} disabled={!segments.length}>Copy transcript</button>
    </div>
    <div className="conversation-toolbar">
      <div className="conversation-tabs" role="group" aria-label="Transcript view">
        <button type="button" aria-pressed={!reviewOnly} onClick={() => setReviewOnly(false)}>All conversation <span>{groups.length}</span></button>
        <button type="button" aria-pressed={reviewOnly} onClick={() => setReviewOnly(true)}>Needs review <span>{reviewCount}</span></button>
      </div>
      <label className="transcript-search"><span className="sr-only">Search transcript</span>
        <input type="search" placeholder="Find a word or phrase…" value={query} onChange={(event) => setQuery(event.target.value)} />
      </label>
    </div>
    <div className="conversation-guidance">
      <p>{approximate ? "Speaker timestamps are approximate. Listen from a little before the passage to check the wording." : "Select a timestamp to listen to that passage."}</p>
      {verificationStatus === "complete" ? <p>Wording differences are highlighted below. Agreement between readings does not guarantee accuracy.</p> :
        verificationStatus ? <p>The second wording check was unavailable. Review names and numbers against the recording.</p> : null}
      {truncated ? <p>The first 32 wording differences are shown. Review the full recording for additional differences.</p> : null}
      {!canSeek ? <p>Playback is unavailable. The transcript remains readable.</p> : null}
    </div>
    {copyStatus ? <p className="copy-status" role="status">{copyStatus}</p> : null}
    <div className="conversation-list">
      {visible.map((group) => {
        const first = group.segments[0];
        const flagged = group.segments.some(segmentNeedsReview);
        const unknown = !first.speaker_label || first.speaker_label === "Unknown";
        const disputed = reviews.filter((review) => group.indexes.includes(review.segment_indexes[0]));
        const active = group.segments.some((segment) => position >= segment.start_timestamp && position < segment.end_timestamp);
        const terms = matches.filter((match) => group.segments.some((segment) => String(segment.id) === String(match.transcript_segment_id)))
          .map((match) => match.original_matched_text || match.keyword_phrase || match.keyword || "");
        return <article key={String(first.id)} className={`conversation-turn${active ? " is-playing" : ""}${flagged ? " is-review" : ""}`}
          aria-label={`${unknown ? "Unassigned speaker" : first.speaker_label}, ${formatTimestamp(first.start_timestamp)}`} aria-current={active ? "true" : undefined}>
          <div className="turn-heading">
            <SpeakerLabel label={first.speaker_label} source={first.speaker_source} timestamp={first.start_timestamp} onSeek={onSeek} disabled={!canSeek} />
            {unknown ? <span className="review-chip">Speaker unclear</span> : disputed.length ? <span className="review-chip">Check wording</span> : flagged ? <span className="review-chip">Check audio</span> : null}
          </div>
          <p className="transcript-text">{group.segments.map((segment, index) => <Fragment key={String(segment.id)}>
            {index > 0 ? " " : ""}{highlight(segment.original_text, [...terms, query])}
          </Fragment>)}</p>
          {unknown ? <p className="turn-note">These words could not be assigned to a speaker reliably.</p> : null}
          {disputed.length ? <details className="wording-review">
            <summary>{disputed.length === 1 ? "Compare wording" : `Compare ${disputed.length} wording differences`}</summary>
            <p>Both readings come from the audio. Listen before deciding which is correct.</p>
            {disputed.map((review, index) => <div className="wording-comparison" key={index}>
              <div><span>Current reading</span><p>{review.original_text || "No words in this reading"}</p></div>
              <div><span>Second reading</span><p>{review.alternative_text || "No words in this reading"}</p></div>
              <button className="button secondary compact" type="button" disabled={!canSeek}
                onClick={() => onSeek(Math.max(0, review.start_seconds - 1))}>Listen from {formatTimestamp(Math.max(0, review.start_seconds - 1))}</button>
            </div>)}
          </details> : flagged ? <button className="text-button review-listen" type="button" disabled={!canSeek}
            onClick={() => onSeek(Math.max(0, first.start_timestamp - 1))}>Listen with context</button> : null}
        </article>;
      })}
      {!visible.length ? <div className="empty-state"><h3>{!segments.length ? "No transcript is available" : query ? "No matching passages" : "No passages flagged for review"}</h3>
        <p className="muted">{!segments.length ? "Processing may still be underway." : "Try another search or return to all conversation."}</p>
        {segments.length ? <button className="button secondary" type="button" onClick={() => { setQuery(""); setReviewOnly(false); }}>Show all conversation</button> : null}
      </div> : null}
    </div>
    <p className="conversation-footer">{query || reviewOnly ? `${visible.length} of ${groups.length} passages` : `${groups.length} passages`} · Original transcript preserved</p>
  </section>;
}

function highlight(text: string, terms: string[]) {
  const search = fold(text);
  const positions: Array<{ start: number; end: number }> = [];
  // Keep offsets into the original Unicode text while matching accent-free Greek.
  const offsets: Array<{ start: number; end: number }> = [];
  let offset = 0;
  for (const char of text) {
    for (let index = 0; index < fold(char).length; index++) offsets.push({ start: offset, end: offset + char.length });
    offset += char.length;
  }
  for (const term of new Set(terms.map((value) => fold(value.trim())).filter(Boolean))) {
    let from = 0;
    while (from < search.length) {
      const found = search.indexOf(term, from);
      if (found < 0) break;
      positions.push({ start: offsets[found].start, end: offsets[found + term.length - 1].end });
      from = found + term.length;
    }
  }
  const merged: typeof positions = [];
  for (const range of positions.sort((a, b) => a.start - b.start)) {
    const last = merged.at(-1);
    if (last && range.start <= last.end) last.end = Math.max(last.end, range.end);
    else merged.push({ ...range });
  }
  const result = [];
  let cursor = 0;
  for (const range of merged) {
    result.push(text.slice(cursor, range.start), <mark key={range.start}>{text.slice(range.start, range.end)}</mark>);
    cursor = range.end;
  }
  result.push(text.slice(cursor));
  return result;
}
