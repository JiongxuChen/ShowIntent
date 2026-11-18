"""Run all three ASR modes on verified development clips, preserving old runs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.speech.references import validate_reference
from src.speech.quality import POLICY


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=ROOT / 'data/private/speech_review/session.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/verified_benchmark')
    args = parser.parse_args()
    session = json.loads(args.session.read_text(encoding='utf-8-sig'))
    pinned = json.loads((ROOT / 'config/asr_model.json').read_text(encoding='utf-8'))
    lexicon_hash = digest(ROOT / 'config/speech_lexicon.json')
    # Validate every human reference before any inference. References never enter ASR.
    for clip in session['clips']:
        reference = json.loads((args.session.parent / 'references' / (clip['case_id'] + '.json')).read_text(encoding='utf-8'))
        validate_reference(reference, clip, require_verified=True)
        if digest(ROOT / clip['source_path']) != clip['source_sha256']:
            raise ValueError('reviewed source bytes changed')
    for clip in session['clips']:
        for mode in ('offline', 'causal', 'agreement'):
            output = args.output_dir / f"{clip['case_id']}_{mode}.jsonl"
            expected = dict(mode=mode, source_sha256=clip['source_sha256'],
                            requested_language=clip.get('language', 'zh'), model_revision=pinned['revision'],
                            lexicon_sha256=lexicon_hash, device='cpu', compute_type='int8', cpu_threads=4,
                            beam_size=3, media_offset_ms=0, audio_offset_ms=clip.get('audio_offset_ms', 0))
            expected.update(quality_policy=POLICY, semantic_annotations='rule_extractor')
            expected.update(dict(step_ms=2000, buffer_ms=15000, paced=False) if mode == 'agreement'
                            else dict(window_ms=4000 if mode == 'causal' else None))
            meta = output.with_suffix('.meta.json')
            sidecars = [output, meta, output.with_suffix('.windows.jsonl'), output.with_suffix('.transcript.jsonl')]
            if any(path.exists() for path in sidecars):
                if not all(path.is_file() for path in sidecars):
                    raise ValueError(f'incomplete existing run: {output}; choose a new output directory')
                saved = json.loads(meta.read_text(encoding='utf-8'))
                if any(saved.get(k) != v for k, v in expected.items()):
                    raise ValueError(f'existing settings differ: {output}; choose a new output directory')
                print(f'Reuse {clip["case_id"]} {mode}', flush=True)
                continue
            print(f'Run {clip["case_id"]} {mode}', flush=True)
            subprocess.run([sys.executable, str(ROOT / 'run_speech.py'), '--backend', 'faster-whisper',
                            '--audio', str(ROOT / clip['source_path']), '--language', expected['requested_language'],
                            '--mode', mode, '--output', str(output)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_speech_references.py'),
                    '--session', str(args.session), '--events-dir', str(args.output_dir),
                    '--output-dir', str(args.output_dir.with_name(args.output_dir.name + '_evaluation'))], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
