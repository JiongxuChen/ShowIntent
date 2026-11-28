import json
from pathlib import Path
from src.evaluation.collection import check_annotations

def test_all_scenes_share_the_same_complete_evaluation_layout():
    expected={'annotations.csv','dataset_manifest.csv','annotation_record.json','protocol.json','split_manifest.csv','track_mapping.csv'}
    fields={'schema_version','scene','label_source','annotations_sha256','video_sha256','review_status','clip_notes','provenance'}
    for scene in ['fruit','books','cups','drinks']:
        folder=Path('evaluation')/scene
        assert {p.name for p in folder.iterdir() if p.is_file()}==expected
        record=json.loads((folder/'annotation_record.json').read_text())
        assert set(record)==fields and record['scene']==scene
        rows,labels=check_annotations(folder,verify_media=False)
        assert set(record['review_status'])=={r['clip_id'] for r in rows}
        assert set(record['video_sha256'])=={r['clip_id'] for r in rows}
