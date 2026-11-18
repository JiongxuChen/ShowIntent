"""Speech-only evaluation gated by independently reviewed local references.

Whole-clip text scores do not depend on ASR timestamps. Lexical/cue and latency
diagnostics do: their maximum-overlap assignment is deliberately reported as an
approximation, never as visual showing/target/fusion accuracy.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from .events import SpeechEvent, require_ms
from .evaluation import score_transcript


MODES = ("offline", "causal", "agreement")
SEMANTIC_SOURCE = "current_rules_on_published_asr_text"
ACCURACY_SETTINGS = ("model_revision", "requested_language", "beam_size", "quality_policy")
PERFORMANCE_SETTINGS = ("device", "compute_type", "cpu_threads")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_snapshot(path: Path):
    payload = path.read_bytes()
    return json.loads(payload.decode("utf-8-sig")), hashlib.sha256(payload).hexdigest()


def _event_snapshot(path: Path):
    payload = path.read_bytes()
    values = []
    for number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                values.append(SpeechEvent.from_dict(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"event line {number}: {exc}") from exc
    return _validated_events(values), hashlib.sha256(payload).hexdigest()


def _rates(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def _set_counts(truth: set, predicted: set) -> tuple[int, int, int]:
    return len(truth & predicted), len(predicted - truth), len(truth - predicted)


def _latency_summary(values) -> dict:
    ordered = sorted(values)
    return {"count": len(ordered), "min": ordered[0] if ordered else None,
            "median": statistics.median(ordered) if ordered else None,
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None,
            "max": ordered[-1] if ordered else None,
            "negative_count": sum(value < 0 for value in ordered)}


def _overlap(event, segment, offset):
    return max(0, min(event.end_ms, segment["end_ms"] + offset)
               - max(event.start_ms, segment["start_ms"] + offset))


def _validated_events(values):
    events = [SpeechEvent.from_dict(value) if isinstance(value, dict) else value for value in values]
    if any(not isinstance(event, SpeechEvent) for event in events):
        raise ValueError("events must contain SpeechEvent instances or complete wire dictionaries")
    for event in events:
        event.to_dict()
    if any(a.available_at_ms > b.available_at_ms for a, b in zip(events, events[1:])):
        raise ValueError("events must be in published availability order")
    return events


def score_verified_clip(reference, clip, events, *, language="zh", media_offset_ms=0,
                        lexicon_path=None) -> dict:
    """Score one complete verified reference; refuse drafts and unusable rows.

    Reference times are audio-relative; add the recorded audio-stream offset and
    an explicit run offset for common-clock diagnostics. All published event text
    contributes to CER/WER, including text outside manually marked speech rows.
    """
    from .references import validate_reference
    from .language import DEFAULT_LEXICON, enrich_event

    validate_reference(reference, clip, require_verified=True)
    require_ms(media_offset_ms, "media_offset_ms")
    offset = require_ms(clip.get("audio_offset_ms", 0), "audio_offset_ms") + media_offset_ms
    events = _validated_events(events)
    published = events
    events = [enrich_event(event, lexicon_path=lexicon_path) for event in published]
    lexicon = Path(lexicon_path) if lexicon_path else DEFAULT_LEXICON
    segments = reference["segments"]
    speech = [segment for segment in segments if segment["kind"] == "speech"]
    no_speech = [segment for segment in segments if segment["kind"] == "no_speech"]
    reference_text = " ".join(segment["text"] for segment in speech)
    hypothesis_text = " ".join(event.text for event in published)
    transcript_score = score_transcript(reference_text, hypothesis_text, language)
    assigned = [[] for _ in speech]
    unmatched = []
    ambiguous = 0
    no_speech_hits = set()
    assigned_to_no_speech = 0
    overlaps_no_speech = 0
    for event in events:
        overlaps = [_overlap(event, segment, offset) for segment in speech]
        candidates = [index for index, amount in enumerate(overlaps) if amount > 0]
        ambiguous += len(candidates) > 1
        if candidates:
            index = max(candidates, key=lambda index: (overlaps[index], -index))
            assigned[index].append(event)
        else:
            unmatched.append(event)
        quiet_overlaps = [_overlap(event, segment, offset) for segment in no_speech]
        if any(quiet_overlaps):
            overlaps_no_speech += 1
            # A boundary-crossing speech event is not automatically a silent-
            # region hallucination. Require no-speech to dominate its overlap.
            quiet_index = max(range(len(quiet_overlaps)), key=lambda index: (quiet_overlaps[index], -index))
            if quiet_overlaps[quiet_index] > max(overlaps, default=0):
                no_speech_hits.add(quiet_index)
                assigned_to_no_speech += 1
    counts = {"mentioned_classes": [0, 0, 0], "cue_types": [0, 0, 0]}
    latency_samples = []
    rows = []
    for segment, matching in zip(speech, assigned):
        row = {"segment_id": segment["segment_id"], "matched_event_count": len(matching)}
        for field in counts:
            truth = set(segment[field])
            prediction = {label for event in matching for label in getattr(event, field)}
            delta = _set_counts(truth, prediction)
            counts[field] = [old + extra for old, extra in zip(counts[field], delta)]
            row[field] = {"expected": sorted(truth), "predicted": sorted(prediction),
                          **_rates(*delta)}
            for label in sorted(truth & prediction):
                earliest = min(event.available_at_ms for event in matching if label in getattr(event, field))
                latency_samples.append({"segment_id": segment["segment_id"], "field": field,
                                        "label": label, "delay_ms": earliest - (segment["end_ms"] + offset)})
        rows.append(row)
    for event in unmatched:
        for field in counts:
            counts[field][1] += len(set(getattr(event, field)))
    return {
        "case_id": clip["case_id"], "source_sha256": clip["source_sha256"],
        "reference_method": reference["reference_method"],
        "coverage_complete": True, "transcript_score": transcript_score,
        "hypothesis_order": "published_event_order; no ASR timestamp filtering of full-clip text",
        "semantic_source": SEMANTIC_SOURCE, "lexicon_sha256": _hash(lexicon),
        "semantic_metrics": {field: _rates(*values) for field, values in counts.items()},
        "temporal_assignment": {
            "policy": "each event to maximum-positive-overlap human speech row; ties choose earlier row",
            "time_origin": "human audio-relative interval + audio_offset_ms + explicit run media_offset_ms",
            "audio_offset_ms": clip.get("audio_offset_ms", 0), "media_offset_ms": media_offset_ms,
            "unmatched_predicted_events": len(unmatched), "ambiguous_multi_speech_overlap_events": ambiguous,
            "human_speech_intervals": len(speech),
            "human_speech_intervals_without_matched_event": sum(not values for values in assigned),
            "limitation": "ASR intervals are approximate; poor alignment can affect semantic assignment independently of transcript quality",
        },
        "no_speech_diagnostics": {
            "human_no_speech_intervals": len(no_speech),
            "intervals_with_predicted_speech": len(no_speech_hits),
            "events_assigned_to_no_speech": assigned_to_no_speech,
            "predicted_events_overlapping_no_speech": overlaps_no_speech,
            "false_speech_interval_rate": len(no_speech_hits) / len(no_speech) if no_speech else None,
            "policy": "event overlap with a no-speech row must exceed its best speech-row overlap; simple boundary overlap is diagnostic only",
        },
        "human_boundary_latency_ms": {
            "definition": "earliest available correct class/cue label minus manually verified utterance end; negative values retained",
            "limitation": "utterance-end lag, not keyword time or calibrated end-to-end visual response; negative values can reflect early mentions or alignment error",
            "overall": _latency_summary(row["delay_ms"] for row in latency_samples),
            "mentioned_classes": _latency_summary(row["delay_ms"] for row in latency_samples if row["field"] == "mentioned_classes"),
            "cue_types": _latency_summary(row["delay_ms"] for row in latency_samples if row["field"] == "cue_types"),
            "samples": latency_samples,
        },
        "interval_semantic_diagnostics": rows,
        "published_event_count": len(events),
    }


def _run_settings(metadata):
    keys = ("mode", "model_revision", "requested_language", "device", "compute_type",
            "cpu_threads", "beam_size", "window_ms", "step_ms", "buffer_ms", "paced",
            "quality_policy", "media_offset_ms", "audio_offset_ms")
    return {key: metadata.get(key) for key in keys if key in metadata}


def _paired_configurations(results, common_ids, modes):
    """Keep diagnostic scores, but pair only documented comparable settings."""
    if len(modes) == 1:
        return common_ids, []
    comparable, exclusions = set(), []
    fields = ACCURACY_SETTINGS + PERFORMANCE_SETTINGS
    for case_id in sorted(common_ids):
        settings = {row["mode"]: row["run_settings"] for row in results
                    if row["case_id"] == case_id and row["mode"] in modes and row["status"] == "SCORED"}
        differences = {}
        unverified = {}
        for field in fields:
            values = {mode: settings[mode].get(field) for mode in modes}
            missing = []
            for mode in modes:
                value = values[mode]
                known = field in settings[mode]
                if field == "requested_language":
                    known = known and (value is None or isinstance(value, str) and bool(value.strip()))
                elif field in ("beam_size", "cpu_threads"):
                    known = known and type(value) is int and value > 0
                else:
                    known = known and isinstance(value, str) and bool(value.strip())
                if not known:
                    missing.append(mode)
            if missing:
                unverified[field] = missing
            if len({json.dumps(value, sort_keys=True) for value in values.values()}) > 1:
                differences[field] = values
        if differences or unverified:
            exclusions.append({
                "case_id": case_id,
                "reason": "CONFIGURATION_MISMATCH" if differences else "UNVERIFIED_CONFIGURATION",
                "differing_settings": differences, "missing_or_unverified_settings": unverified,
                "accuracy_setting_differences": [field for field in ACCURACY_SETTINGS if field in differences],
                "performance_setting_differences": [field for field in PERFORMANCE_SETTINGS if field in differences],
            })
        else:
            comparable.add(case_id)
    return comparable, exclusions


def _aggregate(results, common_ids, modes):
    summaries = {}
    for mode in modes:
        selected = [row for row in results if row["mode"] == mode and row["case_id"] in common_ids
                    and row["status"] == "SCORED"]
        if not selected:
            continue
        lexical = {}
        for field in ("mentioned_classes", "cue_types"):
            sums = [sum(row["semantic_metrics"][field][key] for row in selected) for key in ("tp", "fp", "fn")]
            lexical[field] = _rates(*sums)
        by_language = {}
        for language in sorted({row["language"] for row in selected}):
            language_rows = [row for row in selected if row["language"] == language]
            scores = [row["transcript_score"] for row in language_rows]
            metric = "cer" if language == "zh" else "wer"
            totals = {key: sum(score[key] for score in scores) for key in
                      ("reference_units", "hypothesis_units", "hits", "substitutions", "deletions", "insertions")}
            denominator = totals["reference_units"]
            by_language[language] = {
                "metric": metric, **totals,
                metric: (totals["substitutions"] + totals["deletions"] + totals["insertions"]) / denominator if denominator else None,
                "empty_reference_note": "zero reference units: report insertion counts without a percentage",
            }
            if language == "zh":
                normalized = {key: sum(score["t2s"][key] for score in scores) for key in
                              ("reference_units", "hypothesis_units", "hits", "substitutions", "deletions", "insertions")}
                denominator = normalized["reference_units"]
                by_language[language]["t2s"] = {
                    **normalized,
                    "cer": (normalized["substitutions"] + normalized["deletions"] + normalized["insertions"]) / denominator if denominator else None,
                }
        summaries[mode] = {"case_ids": sorted(common_ids), "semantic_metrics": lexical,
                           "transcript_scores_by_language": by_language}
    return summaries


def evaluate_reference_session(session, references_dir, events_dir, *, modes=MODES,
                               language="zh", lexicon_path=None) -> dict:
    """Evaluate eligible clips only; missing/draft/unusable refs produce no score.

    Results are private summaries without dialogue or image content. Nonexistent
    run modes remain N/A. Cross-mode aggregates use only the shared case set.
    """
    from .references import validate_reference

    if (not isinstance(session, dict) or type(session.get("schema_version")) is not int
            or session["schema_version"] != 1 or not isinstance(session.get("clips"), list)):
        raise ValueError("session must use schema_version1 and contain clips")
    if language not in ("zh", "en"):
        raise ValueError("evaluation language must be zh or en")
    if session.get("scope") != "development":
        raise ValueError("this workflow accepts development references only")
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("modes must be distinct supported ASR modes")
    references_dir, events_dir = Path(references_dir), Path(events_dir)
    exposure_path = references_dir.parent / "draft_exposure.json"
    exposure = _json(exposure_path) if exposure_path.is_file() else {}
    if not isinstance(exposure, dict):
        raise ValueError("draft_exposure must be an object")
    pending, invalid, excluded, eligible, results = [], [], [], [], []
    eligible_methods = {}
    seen = set()
    for clip in session["clips"]:
        case_id = clip.get("case_id") if isinstance(clip, dict) else None
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id) or case_id in seen:
            raise ValueError("session case IDs must be unique safe identifiers")
        seen.add(case_id)
        path = references_dir / f"{case_id}.json"
        if not path.is_file():
            pending.append({"case_id": case_id, "reason": "MISSING_REFERENCE"})
            continue
        try:
            reference, reference_hash = _json_snapshot(path)
            validate_reference(reference, clip)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            invalid.append({"case_id": case_id, "reason": str(exc)})
            continue
        if reference["review_status"] == "unusable":
            excluded.append({"case_id": case_id, "reason": "HUMAN_MARKED_UNUSABLE"})
            continue
        try:
            validate_reference(reference, clip, require_verified=True)
        except (ValueError, TypeError, KeyError) as exc:
            pending.append({"case_id": case_id, "reason": "NOT_READY_FOR_SCORING", "validation": str(exc)})
            continue
        eligible.append(case_id)
        if exposure.get(case_id):
            reference["reference_method"] = "asr_assisted"
        eligible_methods[case_id] = reference["reference_method"]
        metric_language = clip.get("language", language)
        if metric_language not in ("zh", "en"):
            raise ValueError(f"{case_id}: manual evaluation language must be zh or en")
        for mode in modes:
            event_path = events_dir / f"{case_id}_{mode}.jsonl"
            meta_path = event_path.with_suffix(".meta.json")
            row = {"case_id": case_id, "mode": mode, "reference_method": reference["reference_method"]}
            if not event_path.is_file() or not meta_path.is_file():
                results.append({**row, "status": "N/A", "reason": "MISSING_RUN_ARTIFACTS"})
                continue
            try:
                metadata, metadata_hash = _json_snapshot(meta_path)
                if not isinstance(metadata, dict):
                    raise ValueError("ASR run metadata must be an object")
                if metadata.get("source_sha256") != clip["source_sha256"]:
                    raise ValueError("ASR run source SHA256 differs from reviewed clip")
                if metadata.get("mode") != mode:
                    raise ValueError("ASR run metadata mode differs from requested mode")
                if require_ms(metadata.get("audio_offset_ms", 0), "run audio_offset_ms") != clip.get("audio_offset_ms", 0):
                    raise ValueError("ASR run audio offset differs from reviewed clip clock")
                events, events_hash = _event_snapshot(event_path)
                score = score_verified_clip(reference, clip, events, language=metric_language,
                                            media_offset_ms=metadata.get("media_offset_ms", 0), lexicon_path=lexicon_path)
                results.append({**row, **score, "status": "SCORED", "language": metric_language,
                                "reference_sha256": reference_hash, "events_sha256": events_hash,
                                "run_metadata_sha256": metadata_hash, "run_settings": _run_settings(metadata),
                                "asr_draft_exposed": bool(exposure.get(case_id))})
            except (OSError, ValueError, TypeError, KeyError) as exc:
                results.append({**row, "status": "INVALID_RUN", "reason": str(exc)})
    scored = [row for row in results if row["status"] == "SCORED"]
    valid_sets = [{row["case_id"] for row in scored if row["mode"] == mode} for mode in modes]
    common_valid_ids = set.intersection(*valid_sets) if valid_sets else set()
    common_ids, config_exclusions = _paired_configurations(scored, common_valid_ids, modes)
    status = ("PENDING_HUMAN_REVIEW" if not eligible else
              "NO_VALID_RUNS" if not scored else "PARTIAL" if pending or invalid or any(row["status"] != "SCORED" for row in results) else "SCORED")
    return {
        "schema_version": 1, "session_id": session.get("session_id"), "scope": session.get("scope", "development"),
        "status": status, "eligible_verified_cases": eligible, "pending_references": pending,
        "invalid_references": invalid, "excluded_unusable_cases": excluded, "results": results,
        "common_case_ids_for_cross_mode_aggregate": sorted(common_ids),
        "common_case_ids_with_valid_runs": sorted(common_valid_ids),
        "cross_mode_comparison_status": ("NOT_REQUESTED" if len(modes) == 1 else
                                          "NO_COMPARABLE_CASES" if not common_ids else
                                          "PARTIAL_COMPARABLE_CASES" if config_exclusions else
                                          "COMPARABLE_CASES_AVAILABLE"),
        "cross_mode_configuration_exclusions": config_exclusions,
        "cross_mode_matching_policy": {
            "accuracy_settings": list(ACCURACY_SETTINGS),
            "performance_settings": list(PERFORMANCE_SETTINGS),
            "allowed_mode_specific_settings": ["mode", "window_ms", "step_ms", "buffer_ms"],
            "policy": "omit a case from cross-mode aggregates if required settings differ or are unverified; retain individual diagnostic scores",
        },
        "cross_mode_aggregate": _aggregate(scored, common_ids, modes) if common_ids else {},
        "reference_method_counts": dict(Counter(eligible_methods.values())),
        "limitations": ["development references only; no held-out or visual intent/fusion accuracy claim",
                        "drafts, unconfirmed, incomplete and unusable references are never scored",
                        "full-clip text scores include missed speech and all published text; ASR drafts are not ground truth",
                        "temporal lexical/cue assignment is approximate and sensitive to ASR timestamp error",
                        "human-boundary latency uses human utterance end; negative values remain visible",
                        "individual diagnostic runs may use different model/language/decoding settings; inspect each run_settings",
                        "cross-mode aggregates require valid runs for every requested mode and matching documented accuracy/performance settings"],
    }
