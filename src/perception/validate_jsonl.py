"""Validate the minimum A -> B/C JSONL interface without extra dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_box(box: object, location: str) -> list[str]:
    if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in box):
        return [f"{location}: bbox_norm must contain four numbers in [0, 1]"]
    if box[0] >= box[2] or box[1] >= box[3]:
        return [f"{location}: bbox_norm must have x1 < x2 and y1 < y2"]
    return []


def validate_packet(value: object, line: int) -> list[str]:
    location = f"line {line}"
    if not isinstance(value, dict):
        return [f"{location}: expected JSON object"]
    errors = []
    for key in ("timestamp_ms", "frame_id", "objects", "hands", "relations"):
        if key not in value:
            errors.append(f"{location}: missing {key}")
    if errors:
        return errors
    if not isinstance(value["timestamp_ms"], int) or value["timestamp_ms"] < 0:
        errors.append(f"{location}: timestamp_ms must be a non-negative integer")
    for collection, id_field in (("objects", "track_id"), ("hands", "hand_id")):
        if not isinstance(value[collection], list):
            errors.append(f"{location}: {collection} must be a list")
            continue
        for index, item in enumerate(value[collection]):
            item_location = f"{location} {collection}[{index}]"
            if not isinstance(item, dict) or not isinstance(item.get(id_field), str):
                errors.append(f"{item_location}: missing string {id_field}")
                continue
            errors.extend(validate_box(item.get("bbox_norm"), item_location))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PerceptionPacket JSONL")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    last_timestamp = -1
    for line_number, line in enumerate(args.path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
            continue
        errors.extend(validate_packet(value, line_number))
        if isinstance(value, dict) and isinstance(value.get("timestamp_ms"), int):
            if value["timestamp_ms"] < last_timestamp:
                errors.append(f"line {line_number}: timestamp_ms moved backwards")
            last_timestamp = value["timestamp_ms"]
    if errors:
        raise SystemExit("Invalid PerceptionPacket JSONL:\n- " + "\n- ".join(errors))
    print(f"Valid PerceptionPacket JSONL: {args.path}")


if __name__ == "__main__":
    main()
