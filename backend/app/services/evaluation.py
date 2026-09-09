"""Local evaluation files only; deliberately independent of ORM models and ASR services."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import wave
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from fastapi import HTTPException
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.evaluation import ReferenceDraft
from app.services.export import mask_phone_number


IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
AUDIO_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
               ".ogg": "audio/ogg", ".flac": "audio/flac", ".opus": "audio/ogg",
               ".aac": "audio/aac"}
EDITABLE = set(ReferenceDraft.model_fields)


def unavailable(message: str = "Evaluation data is unavailable.", code: int = 404):
    return HTTPException(status_code=code, detail=message)


def contained_path(root: Path, value: str, *, folder: str | None = None) -> Path:
    """Reject cross-platform traversal, absolute paths, symlinks and junctions.

    The root is administrator configuration; every child is checked before access.
    The browser supplies an opaque ID, never any of these paths.
    """
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise unavailable("Invalid evaluation path.", 400)
    relative = PurePosixPath(value)
    if (relative.is_absolute() or PureWindowsPath(value).is_absolute()
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise unavailable("Invalid evaluation path.", 400)
    if folder and (len(relative.parts) < 2 or relative.parts[0] != folder):
        raise unavailable("Invalid evaluation directory.", 400)
    root = root.resolve()
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink() or candidate.is_junction():
            raise unavailable("Linked evaluation paths are not allowed.", 400)
    if not candidate.resolve().is_relative_to(root):
        raise unavailable("Invalid evaluation path.", 400)
    return candidate


@contextmanager
def reference_lock(path: Path):
    """OS locks release on process exit; the empty lock file can safely remain."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                if handle.read(1) == b"":
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise unavailable("Another reference action is running. Try again.", 409) from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class EvaluationStore:
    def __init__(self, settings: Settings):
        self.root = settings.EVALUATION_ROOT.resolve()
        self.manifest_name = settings.EVALUATION_MANIFEST

    def path(self, value: str, folder: str | None = None) -> Path:
        return contained_path(self.root, value, folder=folder)

    def records(self) -> list[dict]:
        path = self.path(self.manifest_name)
        if not path.exists():
            return []
        try:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
            seen_ids, seen_references = set(), set()
            for record in records:
                record_id = record.get("evaluation_id", record.get("id", ""))
                if not isinstance(record_id, str) or not IDENTIFIER.fullmatch(record_id):
                    raise ValueError
                if record_id in seen_ids or record.get("split") not in {"dev", "test"}:
                    raise ValueError
                if record.get("mode") not in {"mono", "stereo"}:
                    raise ValueError
                audio = self.path(record["audio"], "audio")
                reference = self.path(record["reference"], "references")
                if audio.suffix.lower() not in AUDIO_TYPES or reference.suffix != ".json":
                    raise ValueError
                if reference in seen_references:
                    raise ValueError
                if record.get("context"):
                    self.path(record["context"], "context")
                seen_ids.add(record_id)
                seen_references.add(reference)
                record["evaluation_id"] = record_id
            return records
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
            raise unavailable("The evaluation manifest is invalid.", 422) from exc

    def record(self, record_id: str) -> dict:
        if not IDENTIFIER.fullmatch(record_id):
            raise unavailable("Invalid evaluation ID.", 400)
        for record in self.records():
            if record["evaluation_id"] == record_id:
                return record
        raise unavailable("Evaluation recording not found.")

    def audio_path(self, record: dict) -> Path:
        path = self.path(record["audio"], "audio")
        if not path.is_file():
            raise unavailable("The frozen evaluation audio is missing.")
        return path

    def audio_info(self, record: dict) -> tuple[float, int]:
        path = self.audio_path(record)
        try:
            if path.suffix.lower() == ".wav":
                try:
                    with wave.open(str(path), "rb") as source:
                        duration = source.getnframes() / source.getframerate()
                        channels = source.getnchannels()
                except (wave.Error, EOFError):
                    duration, channels = self._probe(path)
            else:
                duration, channels = self._probe(path)
            if not math.isfinite(duration) or duration <= 0 or channels not in {1, 2}:
                raise ValueError
            if channels != (2 if record["mode"] == "stereo" else 1):
                raise unavailable("Audio channels do not match the frozen manifest.", 422)
            return duration, channels
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            raise unavailable("Cannot inspect evaluation audio. Check the file and FFmpeg installation.", 422) from exc

    @staticmethod
    def _probe(path: Path) -> tuple[float, int]:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=channels:format=duration", "-of", "json", str(path)],
            check=True, capture_output=True, timeout=30,
        )
        payload = json.loads(result.stdout)
        return float(payload["format"]["duration"]), int(payload["streams"][0]["channels"])

    def metadata(self, record: dict) -> dict:
        # The frozen PBX context is allowlisted. Never return raw provider payload,
        # storage paths, production call IDs, transcripts, matches or ASR fields.
        raw = record.get("metadata", {})
        if record.get("context"):
            try:
                raw = json.loads(self.path(record["context"], "context").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise unavailable("The frozen PBX context is unavailable.", 422) from exc
        if not isinstance(raw, dict):
            raise unavailable("Invalid frozen PBX context.", 422)
        def text(key):
            value = raw.get(key)
            return value[:255] if isinstance(value, str) else None
        operators = raw.get("operators", [])
        return {
            "direction": text("direction"),
            "queue": text("queue"),
            "is_queue": raw.get("is_queue") if isinstance(raw.get("is_queue"), bool) else None,
            "transfer_state": text("transfer_state"),
            "audio_topology": text("audio_topology"),
            "occurred_at": text("occurred_at"),
            "caller": mask_phone_number(text("caller")),
            "callee": mask_phone_number(text("callee")),
            "operators": [
                {"name": item.get("name", "")[:255], "extension": item.get("extension", "")[:40]}
                for item in operators if isinstance(item, dict)
                and isinstance(item.get("name", ""), str)
                and isinstance(item.get("extension", ""), str)
            ] if isinstance(operators, list) else [],
        }

    def read_reference(self, record: dict) -> dict:
        path = self.path(record["reference"], "references")
        try:
            content = path.read_bytes() if path.exists() else b""
            raw = json.loads(content) if content else {}
            if not isinstance(raw, dict):
                raise ValueError
            editable = {key: raw[key] for key in EDITABLE if key in raw}
            # Historical cohort names are not controlled human quality labels.
            legacy_quality = editable.get("quality") not in {None, "clean", "normal", "noisy", "very_noisy"}
            if legacy_quality:
                editable["quality"] = None
            draft = ReferenceDraft.model_validate(editable)
            status = raw.get("verification_status", "in_progress" if content else "not_started")
            if status not in {"not_started", "in_progress", "verified"}:
                raise ValueError
            if legacy_quality:
                status = "in_progress"
            verified_at = raw.get("verified_at") if status == "verified" else None
            if verified_at is not None:
                verified_at = datetime.fromisoformat(verified_at).isoformat()
            return {
                **draft.model_dump(), "verification_status": status, "verified_at": verified_at,
                "revision": hashlib.sha256(content).hexdigest(),
            }
        except (OSError, ValueError, TypeError, ValidationError) as exc:
            raise unavailable("The local human reference is invalid.", 422) from exc

    def detail(self, record_id: str) -> dict:
        record = self.record(record_id)
        return self._detail(record)

    def _detail(self, record: dict) -> dict:
        duration, _ = self.audio_info(record)
        return {
            "evaluation_id": record["evaluation_id"], "split": record["split"], "mode": record["mode"],
            "duration_seconds": duration, "metadata": self.metadata(record),
            "reference": self.read_reference(record),
        }

    def listing(self, filter_by: str) -> dict:
        rows = []
        for record in self.records():
            detail = self._detail(record)
            rows.append({
                "evaluation_id": detail["evaluation_id"], "split": detail["split"],
                "mode": detail["mode"], "duration_seconds": detail["duration_seconds"],
                "direction": detail["metadata"]["direction"],
                "quality": detail["reference"]["quality"],
                "verification_status": detail["reference"]["verification_status"],
            })
        progress = {}
        for split in ("all", "dev", "test"):
            cohort = [r for r in rows if split == "all" or r["split"] == split]
            progress[split] = {"total": len(cohort), "verified": sum(
                r["verification_status"] == "verified" for r in cohort)}
        filtered = [r for r in rows if filter_by == "all"
                    or r["split"] == filter_by
                    or (filter_by == "verified" and r["verification_status"] == "verified")
                    or (filter_by == "unverified" and r["verification_status"] != "verified")]
        return {"items": filtered, "progress": progress}

    @staticmethod
    def verification_errors(draft: ReferenceDraft, duration: float, mode: str) -> list[str]:
        errors = []
        if draft.quality is None:
            errors.append("Select a quality label.")
        if not draft.segments:
            errors.append("Add at least one reference segment.")
        for index, segment in enumerate(draft.segments, 1):
            if segment.end <= segment.start or segment.end > duration or segment.start >= duration:
                errors.append(f"Segment {index}: start and end must be ordered and within the recording.")
            if not segment.exclude_from_wer and not segment.text.strip():
                errors.append(f"Segment {index}: enter audible text or exclude an unintelligible region.")
            if mode == "mono" and segment.channel is not None:
                errors.append(f"Segment {index}: choose None for a mono recording.")
        if mode == "stereo" and not draft.operator_channel_answered:
            errors.append("Answer the operator channel question, including Cannot establish when appropriate.")
        if mode == "mono" and draft.operator_channel is not None:
            errors.append("A mono recording cannot have an operator channel.")
        return errors

    def write_reference(self, record_id: str, revision: str,
                        draft: ReferenceDraft | None = None, *, verify: bool = False) -> dict:
        record = self.record(record_id)
        path = self.path(record["reference"], "references")
        lock = self.path(record["reference"] + ".lock", "references")
        with reference_lock(lock):
            current = self.read_reference(record)
            if current["revision"] != revision:
                raise unavailable("This reference changed in another tab. Reload before saving or verifying.", 409)
            if verify:
                draft = ReferenceDraft.model_validate({k: current[k] for k in EDITABLE})
                duration, _ = self.audio_info(record)
                errors = self.verification_errors(draft, duration, record["mode"])
                if errors:
                    raise unavailable(" ".join(errors), 422)
            assert draft is not None
            payload = {
                "id": record_id, "evaluation_id": record_id, "split": record["split"],
                "mode": record["mode"], **draft.model_dump(),
                "verification_status": "verified" if verify else "in_progress",
                "verified_at": datetime.now(UTC).isoformat() if verify else None,
            }
            # Only a reference is written. Atomic replacement avoids partial JSON
            # and never modifies an existing hard-linked target in place.
            path = self.path(record["reference"], "references")
            fd, temporary = tempfile.mkstemp(prefix=".reference-", suffix=".json", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            return self.read_reference(record)

    def channel_preview(self, record_id: str, channel: int) -> Path:
        record = self.record(record_id)
        if channel not in {0, 1} or record["mode"] != "stereo":
            raise unavailable("This recording does not have that stereo channel.")
        source = self.audio_path(record)
        self.audio_info(record)
        fingerprint = hashlib.sha256(
            (record["audio"] + str(source.stat().st_mtime_ns) + str(source.stat().st_size)).encode()
        ).hexdigest()[:20]
        name = f"previews/{record_id}-{fingerprint}-{channel}.wav"
        destination = self.path(name, "previews")
        if destination.is_file():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".channel-", suffix=".wav", dir=destination.parent)
        os.close(fd)
        try:
            if source.suffix.lower() == ".wav":
                try:
                    self._extract_wave(source, Path(temp_name), channel)
                except (wave.Error, EOFError):
                    self._extract_ffmpeg(source, Path(temp_name), channel)
            else:
                self._extract_ffmpeg(source, Path(temp_name), channel)
            os.replace(temp_name, self.path(name, "previews"))
        except (OSError, subprocess.SubprocessError, wave.Error, EOFError) as exc:
            raise unavailable("Channel preview could not be generated. Check the audio and FFmpeg installation.", 422) from exc
        finally:
            Path(temp_name).unlink(missing_ok=True)
        return destination

    @staticmethod
    def _extract_wave(source: Path, destination: Path, channel: int):
        with wave.open(str(source), "rb") as audio, wave.open(str(destination), "wb") as preview:
            width = audio.getsampwidth()
            preview.setnchannels(1)
            preview.setsampwidth(width)
            preview.setframerate(audio.getframerate())
            while frames := audio.readframes(65536):
                samples = b"".join(frames[offset:offset + width]
                                   for offset in range(channel * width, len(frames), width * 2))
                preview.writeframesraw(samples)

    @staticmethod
    def _extract_ffmpeg(source: Path, destination: Path, channel: int):
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
             "-map", "0:a:0", "-af", f"pan=mono|c0=c{channel}", "-c:a", "pcm_s16le",
             "-map_metadata", "-1", str(destination)],
            check=True, capture_output=True, timeout=120,
        )
