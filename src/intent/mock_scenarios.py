"""B-owned, deterministic synthetic inputs; no detector or video dependency.

Expected behavior is stored separately and is never passed to an intent engine.
These fixtures diagnose engineering behavior, not accuracy on real videos.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Scenario:
    name: str
    description: str
    packets: list[dict]
    events: list[dict] = field(default_factory=list)
    forbidden: list[tuple[int, int]] = field(default_factory=list)
    holds: list[dict] = field(default_factory=list)
    releases: list[dict] = field(default_factory=list)
    final_none: bool = True

    def expectations(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "source": "synthetic_engineering_fixture_not_real_video_ground_truth",
            "events": self.events,
            "forbidden_showing_ranges_ms": self.forbidden,
            "required_hold_ranges": self.holds,
            "release_deadlines": self.releases,
            "require_final_none": self.final_none,
        }


def _object(track_id: str, class_name: str, x: float, y: float) -> dict:
    return {"track_id": track_id, "class_name": class_name,
            "bbox_norm": [round(x - .06, 6), round(y - .07, 6),
                          round(x + .06, 6), round(y + .07, 6)], "score": .95}


def _show_y(t: int, start: int, end: int) -> float:
    if t < start or t >= end + 600:
        return .80
    if t < start + 1000:
        return .80 - .35 * (t - start) / 1000
    if t < end:
        return .45
    return .45 + .35 * (t - end) / 600


def _event(target: str, start: int, end: int) -> dict:
    # Confirmation is allowed up to 1.5 seconds after the lift finishes.
    return {"target_id": target, "start_ms": start, "end_ms": end,
            "latest_start_ms": start + 1500}


def make_scenarios(fps: int = 30) -> list[Scenario]:
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 120:
        raise ValueError("fps must be an integer between 1 and 120")
    definitions = [
        ("idle", "桌面同时存在不同类别候选，无手物关系", 6000),
        ("contact_only", "手靠近桌面物体但未举起", 6000),
        ("table_transport", "有手物关联的桌面横向搬运", 8000),
        ("raised_transport", "举离桌面后持续横向搬运，没有停留", 8000),
        ("initial_hold", "开场已经持物，未观察到展示动作；采用保守不触发策略", 8000),
        ("lift_hold_putdown", "举起、展示性停留、放下、回到空闲", 10000),
        ("two_targets", "杯子展示后搬运苹果，再展示苹果；桌面杯子持续关联造成干扰", 16000),
        ("brief_occlusion", "展示中短暂漏检 400 毫秒，目标 ID 恢复不变", 11000),
        ("long_loss", "展示目标长期消失，必须有限时长后退出", 11000),
    ]
    result = []
    for name, description, duration in definitions:
        packets = []
        for frame in range(duration * fps // 1000 + 1):
            t = round(frame * 1000 / fps)
            cup_x, cup_y, apple_x, apple_y = .30, .80, .70, .80
            associations: list[tuple[str, str]] = []
            missing: set[str] = set()
            if name == "contact_only":
                associations = [("hand_right", "track_cup")]
            elif name in ("table_transport", "raised_transport") and 2000 <= t < 6000:
                cup_x = .25 + .10 * (t - 2000) / 1000
                if name == "raised_transport":
                    cup_y = .80 - .22 * min(1., (t - 2000) / 500)
                associations = [("hand_right", "track_cup")]
            elif name == "initial_hold":
                cup_y = .60 if t < 5000 else .80
                if t < 5000:
                    associations = [("hand_right", "track_cup")]
            elif name in ("lift_hold_putdown", "two_targets", "brief_occlusion", "long_loss"):
                end = 6000 if name in ("lift_hold_putdown", "two_targets") else 8000
                cup_y = _show_y(t, 2000, end)
                if 2000 <= t < end + 600:
                    associations.append(("hand_right", "track_cup"))
                if name == "two_targets":
                    if 6000 <= t < 10500:
                        associations = [("hand_left", "track_cup")]
                    if 7000 <= t < 9000:
                        apple_x = .70 - .10 * (t - 7000) / 1000
                        associations.append(("hand_right", "track_apple"))
                    apple_y = _show_y(t, 9000, 13000)
                    if 9000 <= t < 13600:
                        apple_x = .50
                        associations.append(("hand_right", "track_apple"))
                if name == "brief_occlusion" and 4000 <= t < 4400:
                    missing.add("track_cup")
                if name == "long_loss" and 5000 <= t < 9000:
                    missing.add("track_cup")
            objects = [_object("track_cup", "cup", cup_x, cup_y),
                       _object("track_apple", "apple", apple_x, apple_y),
                       _object("track_bottle", "bottle", .88, .80)]
            by_id = {o["track_id"]: o for o in objects}
            hands = []
            relations = []
            for hand_id, target in associations:
                box = by_id[target]["bbox_norm"]
                hands.append({"hand_id": hand_id, "bbox_norm": list(box), "score": .90})
                if target not in missing:
                    relations.append({"hand_id": hand_id, "object_id": target,
                                      "type": "near_or_moving_together", "score": .85})
            packets.append({"timestamp_ms": t, "frame_id": frame,
                            "objects": [o for o in objects if o["track_id"] not in missing],
                            "hands": hands, "relations": relations})
        scenario = Scenario(name, description, packets)
        if name in ("idle", "contact_only", "table_transport", "raised_transport", "initial_hold"):
            scenario.forbidden = [(0, duration + 1)]
            scenario.final_none = name != "contact_only"
        else:
            end = 5000 if name == "long_loss" else 8000 if name == "brief_occlusion" else 6000
            scenario.events = [_event("track_cup", 3000, end)]
            if name == "two_targets":
                scenario.events.append(_event("track_apple", 10000, 13000))
                scenario.forbidden = [(7000, 9000)]
            if name == "brief_occlusion":
                scenario.holds = [{"start_ms": 4000, "end_ms": 4400, "target_id": "track_cup"}]
            if name == "long_loss":
                scenario.releases = [{"target_id": "track_cup", "lost_at_ms": 5000,
                                      "release_by_ms": 6500, "until_ms": 9000}]
        result.append(scenario)
    return result


def write_scenario(scenario: Scenario, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "perception.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in scenario.packets), encoding="utf-8")
    (out_dir / "expectations.json").write_text(
        json.dumps(scenario.expectations(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 B 的合成诊断场景，不依赖 A 模型")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    try:
        scenarios = make_scenarios(args.fps)
    except ValueError as exc:
        parser.error(str(exc))
    for scenario in scenarios:
        write_scenario(scenario, args.out / scenario.name)
    print(f"Wrote {len(scenarios)} synthetic scenarios at {args.fps} FPS -> {args.out}")


if __name__ == "__main__":
    main()
