"""Public summaries must omit private fields and reject changed artifacts."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_verified_speech_summary import build_summary


class SummaryExportTests(unittest.TestCase):
    def test_private_fields_are_omitted_and_hash_changes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            event_bytes = b''
            metadata = json.dumps(dict(compute_rtf=0.1, source='PRIVATE_PATH')).encode()
            (path / 'A01_offline.jsonl').write_bytes(event_bytes)
            (path / 'A01_offline.meta.json').write_bytes(metadata)
            row = dict(status='SCORED', case_id='A01', mode='offline', reference_method='independent_listening',
                       transcript_score={}, semantic_metrics={}, run_settings={}, reference_sha256='reference_hash',
                       events_sha256=hashlib.sha256(event_bytes).hexdigest(),
                       run_metadata_sha256=hashlib.sha256(metadata).hexdigest(),
                       human_boundary_latency_ms=dict(overall={}, samples=[dict(text='PRIVATE_SPEECH')]),
                       temporal_assignment=dict(human_speech_intervals=0, human_speech_intervals_without_matched_event=0,
                                                ambiguous_multi_speech_overlap_events=0), no_speech_diagnostics={}, reviewer='PRIVATE_NAME')
            report = dict(status='SCORED', eligible_verified_cases=['A01'], results=[row],
                          cross_mode_comparison_status='NOT_REQUESTED', common_case_ids_for_cross_mode_aggregate=[],
                          cross_mode_aggregate={}, limitations=[])
            encoded = json.dumps(build_summary(report, path))
            for secret in ('PRIVATE_PATH', 'PRIVATE_SPEECH', 'PRIVATE_NAME'):
                self.assertNotIn(secret, encoded)
            (path / 'A01_offline.jsonl').write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'events changed'):
                build_summary(report, path)


if __name__ == '__main__':
    unittest.main()
