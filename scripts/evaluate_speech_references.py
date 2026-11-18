"""Score manually verified speech references, or write a pending-review report."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_speech import write_jsonl
from src.speech.reference_evaluation import MODES, evaluate_reference_session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=ROOT / "data/private/speech_review/session.json")
    parser.add_argument("--references-dir", type=Path, help="defaults to session sibling references/")
    parser.add_argument("--events-dir", type=Path, default=ROOT / "outputs/recorded_tests")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reference_evaluation")
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="manual metric language fallback; clip.language takes precedence")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--lexicon", type=Path, help="optional current rule configuration")
    args = parser.parse_args(argv)
    reference_dir = args.references_dir or args.session.parent / "references"
    destinations = [args.output_dir / "report.json", args.output_dir / "results.jsonl"]
    if any(path.resolve() == args.session.resolve() or
           path.resolve().is_relative_to(reference_dir.resolve()) or
           path.resolve().is_relative_to(args.events_dir.resolve()) or
           args.lexicon is not None and path.resolve() == args.lexicon.resolve()
           for path in destinations):
        parser.error("evaluation outputs must not overwrite session, reference, event or lexicon inputs")
    try:
        if not args.session.is_file():
            report = {"schema_version": 1, "status": "PENDING_HUMAN_REVIEW",
                      "eligible_verified_cases": [], "results": [], "cross_mode_aggregate": {},
                      "pending_references": [{"reason": "MISSING_REVIEW_SESSION"}],
                      "note": "Prepare and independently review local references before scoring; no accuracy computed."}
        else:
            session = json.loads(args.session.read_text(encoding="utf-8-sig"))
            report = evaluate_reference_session(session, reference_dir, args.events_dir,
                                                modes=args.modes, language=args.language, lexicon_path=args.lexicon)
        write_jsonl(destinations[0], [report])
        write_jsonl(destinations[1], report["results"])
        scored = sum(row.get("status") == "SCORED" for row in report["results"])
        print(f"{report['status']}: {len(report['eligible_verified_cases'])} verified clip(s), {scored} scored run(s); {destinations[0]}")
        return 0 if report["status"] not in {"NO_VALID_RUNS"} else 2
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
