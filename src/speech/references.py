"""Human-reference contract and local persistence; generated rows remain drafts."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from threading import RLock

CLASSES = {"apple", "cup", "bottle", "orange", "cell phone", "book"}
CUES = {"PRESENTATION_LANGUAGE", "ENDING_LANGUAGE", "NEGATED_LANGUAGE", "TOPIC_CHANGE"}
RECORD_FIELDS = {"schema_version", "case_id", "source_sha256", "review_status", "reviewer",
                 "human_confirmed", "coverage_complete", "reference_method", "notes", "segments"}
SEGMENT_FIELDS = {"segment_id", "start_ms", "end_ms", "kind", "text", "mentioned_classes",
                  "cue_types", "timing_verified", "semantic_verified", "notes"}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def make_draft(clip):
    return {"schema_version": 1, "case_id": clip["case_id"], "source_sha256": clip["source_sha256"],
            "review_status": "draft", "reviewer": "", "human_confirmed": False,
            "coverage_complete": False, "reference_method": "independent_listening", "notes": "",
            "segments": [{"segment_id": clip["case_id"] + "-s001", "start_ms": 0,
                          "end_ms": clip["duration_ms"], "kind": "uncertain", "text": "",
                          "mentioned_classes": [], "cue_types": [], "timing_verified": False,
                          "semantic_verified": False, "notes": ""}]}


def _string(value, name, maximum=10000):
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{name} must be text of at most {maximum} characters")


def _labels(values, allowed, name):
    if not isinstance(values, list) or any(not isinstance(v, str) or v not in allowed for v in values):
        raise ValueError(f"{name} contains an unsupported label")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


def validate_reference(record, clip, require_verified=False):
    """Validate human provenance, complete coverage assertion and interval bounds.

    Human confirmations are explicit assertions, not inferred from playback or
    generated predictions. Draft overlaps are allowed while a user edits a split.
    """
    if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
        raise ValueError("reference fields do not match schema_version 1")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("reference schema_version must be 1")
    if record["case_id"] != clip["case_id"] or record["source_sha256"] != clip["source_sha256"]:
        raise ValueError("reference source identity differs from the prepared recording")
    if record["review_status"] not in {"draft", "verified", "unusable"}:
        raise ValueError("invalid review_status")
    if record["reference_method"] not in {"independent_listening", "asr_assisted"}:
        raise ValueError("invalid reference_method")
    for name in ("human_confirmed", "coverage_complete"):
        if type(record[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    _string(record["reviewer"], "reviewer", 120)
    _string(record["notes"], "notes")
    rows = record["segments"]
    if not isinstance(rows, list) or not rows or len(rows) > 1000:
        raise ValueError("segments must contain 1 to 1000 rows")
    verified = record["review_status"] == "verified"
    if require_verified and not verified:
        raise ValueError("reference is not human-verified")
    if verified and not (record["human_confirmed"] and record["coverage_complete"] and record["reviewer"].strip()):
        raise ValueError("确认前请填写校对者，并明确确认人工审听和完整覆盖")
    ids = set()
    previous_end = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != SEGMENT_FIELDS:
            raise ValueError("segment fields do not match schema_version 1")
        sid = row["segment_id"]
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", sid) or sid in ids:
            raise ValueError("segment_id must be unique and contain only letters/numbers/_/-")
        ids.add(sid)
        start, end = row["start_ms"], row["end_ms"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= clip["duration_ms"]:
            raise ValueError(f"{sid}: 时间必须满足 0 <= 开始 < 结束 <= 音频时长（毫秒）")
        if row["kind"] not in {"speech", "no_speech", "uncertain"}:
            raise ValueError("invalid segment kind")
        _string(row["text"], "text")
        _string(row["notes"], "segment notes")
        _labels(row["mentioned_classes"], CLASSES, "mentioned_classes")
        _labels(row["cue_types"], CUES, "cue_types")
        for flag in ("timing_verified", "semantic_verified"):
            if type(row[flag]) is not bool:
                raise ValueError(f"{flag} must be boolean")
        if row["kind"] == "no_speech" and (row["text"].strip() or row["mentioned_classes"] or row["cue_types"]):
            raise ValueError(f"{sid}: 无语音行应清空文字、类别和语言标签")
        if verified:
            if start < previous_end:
                raise ValueError("确认前请按时间排序，并消除标注区间重叠")
            previous_end = end
            if row["kind"] == "uncertain":
                raise ValueError("仍有听不清的行，请保留草稿或将片段标为不可用")
            if row["kind"] == "speech" and not row["text"].strip():
                raise ValueError(f"{sid}: 有语音行必须填写实际听到的原话")
            if not row["timing_verified"] or not row["semantic_verified"]:
                raise ValueError(f"{sid}: 请逐行确认时间和语义标签")
    return deepcopy(record)


class ReferenceStore:
    def __init__(self, session_path):
        self.session_path = Path(session_path).resolve()
        self.folder = self.session_path.parent
        self.session = json.loads(self.session_path.read_text(encoding="utf-8"))
        self.clips = {clip["case_id"]: clip for clip in self.session["clips"]}
        if len(self.clips) != len(self.session["clips"]):
            raise ValueError("duplicate case identifiers")
        for case_id in self.clips:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", case_id):
                raise ValueError("invalid case identifier")
        self.lock = RLock()
        self.ref_dir = self.folder / "references"
        self.ref_dir.mkdir(exist_ok=True)
        self.exposure_path = self.folder / "draft_exposure.json"
        self.exposure = (json.loads(self.exposure_path.read_text(encoding="utf-8"))
                         if self.exposure_path.exists() else {})

    def _clip(self, case_id):
        if case_id not in self.clips:
            raise KeyError(case_id)
        return self.clips[case_id]

    def get(self, case_id):
        clip = self._clip(case_id)
        with self.lock:
            path = self.ref_dir / (case_id + ".json")
            value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else make_draft(clip)
            value = validate_reference(value, clip)
            if self.exposure.get(case_id):
                value["reference_method"] = "asr_assisted"
            return value

    def save(self, case_id, value):
        with self.lock:
            clip = self._clip(case_id)
            value = validate_reference(value, clip)
            if self.exposure.get(case_id):
                value["reference_method"] = "asr_assisted"
            atomic_json(self.ref_dir / (case_id + ".json"), value)
            with (self.folder / "review_audit.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                                         "case_id": case_id, "review_status": value["review_status"],
                                         "reference_method": value["reference_method"]}) + "\n")
            return value

    def mark_draft_exposed(self, case_id):
        self._clip(case_id)
        with self.lock:
            self.exposure[case_id] = True
            atomic_json(self.exposure_path, self.exposure)
            # Persist provenance even if the browser closes before the next save.
            current = self.get(case_id)
            atomic_json(self.ref_dir / (case_id + ".json"), current)

    def progress(self):
        values = [self.get(case_id) for case_id in self.clips]
        return {"verified": sum(v["review_status"] == "verified" for v in values),
                "total": len(values), "unusable": sum(v["review_status"] == "unusable" for v in values)}
