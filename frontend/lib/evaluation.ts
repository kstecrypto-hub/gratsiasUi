import type { EvaluationDetail, HumanReference, ReferenceDraft } from "@/lib/types";

export const qualityDefinitions = {
  clean: "Clear speech with little or no background noise.",
  normal: "Everyday call audio with minor noise or compression.",
  noisy: "Noticeable noise or distortion; most speech remains understandable.",
  very_noisy: "Heavy noise or distortion; substantial speech is hard to understand.",
} as const;

export function editableReference(reference: HumanReference): ReferenceDraft {
  return {
    quality: reference.quality,
    operator_channel: reference.operator_channel,
    operator_channel_answered: reference.operator_channel_answered,
    expected_keywords: reference.expected_keywords,
    segments: reference.segments,
  };
}

export function verificationErrors(draft: ReferenceDraft, call: EvaluationDetail): string[] {
  const errors: string[] = [];
  if (!draft.quality) errors.push("Select a quality label.");
  if (!draft.segments.length) errors.push("Add at least one reference segment.");
  draft.segments.forEach((segment, index) => {
    if (!Number.isFinite(segment.start) || !Number.isFinite(segment.end) ||
      segment.start < 0 || segment.end <= segment.start || segment.end > call.duration_seconds) {
      errors.push(`Segment ${index + 1}: enter valid start/end times within the recording.`);
    }
    if (!segment.exclude_from_wer && !segment.text.trim()) {
      errors.push(`Segment ${index + 1}: enter audible text or exclude an unintelligible region.`);
    }
    if (call.mode === "mono" && segment.channel !== null) {
      errors.push(`Segment ${index + 1}: choose None for a mono recording.`);
    }
  });
  if (call.mode === "stereo" && !draft.operator_channel_answered) {
    errors.push("Answer the operator channel question.");
  }
  return errors;
}
