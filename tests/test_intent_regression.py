"""Optional regression tests using locally supplied perception outputs."""

def _run_real_clip(name):
    """Run the state machine over a real perception JSONL (skipped when the
    local evaluation outputs are absent, e.g. in CI)."""
    import json
    import os
    from pathlib import Path
    directory = Path(os.environ.get("SHOWINTENT_PERCEPTION_DIR", Path(__file__).resolve().parent.parent / "outputs"))
    path = directory / f"perception_{name}.jsonl"
    if not path.exists():
        import pytest
        pytest.skip(f"real perception output not available: {path}")
    from src.intent.temporal import TemporalIntentStateMachine
    sm = TemporalIntentStateMachine()
    with path.open(encoding="utf-8") as f:
        return [sm.update(json.loads(line)) for line in f if line.strip()]


def test_regression_A02_ends_and_returns_to_none():
    out = _run_real_clip("A02")
    assert any(o["state"] == "SHOWING" for o in out)
    assert out[-1]["state"] in ("NONE", "CANDIDATE")


def test_regression_C01_C03_detect_showing():
    for name in ("C01", "C03"):
        out = _run_real_clip(name)
        assert any(o["state"] == "SHOWING" for o in out), f"{name}: SHOWING missed"


def test_regression_B02_B03_no_false_trigger():
    for name in ("B02", "B03"):
        out = _run_real_clip(name)
        assert not any(o["state"] == "SHOWING" for o in out), f"{name}: false trigger"
