"""Export numeric development diagnostics, without dialogue or reviewer identity."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_speech import write_jsonl
from src.speech.subtitles import read_events


def build_summary(report, events_dir):
    rows = []
    for row in report['results']:
        if row['status'] != 'SCORED':
            continue
        case, mode = row['case_id'], row['mode']
        path = events_dir / f'{case}_{mode}.jsonl'
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['events_sha256']:
            raise ValueError('events changed since evaluation')
        events = read_events(path)
        meta_path = path.with_suffix('.meta.json')
        if hashlib.sha256(meta_path.read_bytes()).hexdigest() != row['run_metadata_sha256']:
            raise ValueError('metadata changed since evaluation')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        category_events = [event for event in events if event.mentioned_classes]
        rows.append(dict(case_id=case, mode=mode, reference_method=row['reference_method'],
                         transcript_score=row['transcript_score'], semantic_metrics=row['semantic_metrics'],
                         published_event_count=len(events), category_event_count=len(category_events),
                         category_events_fresh_at_publication_3s=sum(event.available_at_ms - event.end_ms <= 3000 for event in category_events),
                         compute_rtf=meta['compute_rtf'],
                         human_boundary_latency_ms={name: value for name, value in row['human_boundary_latency_ms'].items() if name in ('overall', 'mentioned_classes', 'cue_types')},
                         temporal_assignment={name: row['temporal_assignment'][name] for name in ('human_speech_intervals', 'human_speech_intervals_without_matched_event', 'ambiguous_multi_speech_overlap_events')},
                         no_speech_diagnostics=row['no_speech_diagnostics'],
                         run_settings=row['run_settings'], reference_sha256=row['reference_sha256'],
                         events_sha256=row['events_sha256'], run_metadata_sha256=row['run_metadata_sha256']))
    return dict(schema_version=1, scope='development', status=report['status'],
                verified_case_count=len(report['eligible_verified_cases']), scored_run_count=len(rows),
                missing_run_count=sum(row['status'] != 'SCORED' for row in report['results']),
                reference_method_run_counts=dict(Counter(row['reference_method'] for row in rows)),
                cross_mode_comparison_status=report['cross_mode_comparison_status'],
                common_case_ids=report['common_case_ids_for_cross_mode_aggregate'],
                cross_mode_aggregate=report['cross_mode_aggregate'], results=rows,
                limitations=report['limitations'] + ['Fresh-at-publication counts measure category-bearing ASR events, not correct visual matches or final B fusion.',
                    'Human-boundary latency samples with raw text and segment IDs are excluded from this export.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events-dir', type=Path, default=ROOT / 'outputs/verified_benchmark')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/experiments/verified_speech_results.json')
    args = parser.parse_args()
    report = json.loads((args.events_dir.with_name(args.events_dir.name + '_evaluation') / 'report.json').read_text(encoding='utf-8'))
    write_jsonl(args.output, [build_summary(report, args.events_dir)])
    print(f'Exported numeric diagnostics: {args.output}')


if __name__ == '__main__':
    main()
