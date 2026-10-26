"""Aggregate counts of an autolabel export (CPU, stdlib only). Added for the public write-up, 2026-10-26.

Reads frames.jsonl, intervals.jsonl, events.jsonl, strikes.jsonl and manifest.json and writes counts only: no frame,
source or interval identifiers, so the summary can be published without the labels.

  python src/export_summary.py --export <workspace>/exports/autolabel-v1 --out docs/export_summary.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cohort(trial_id):
    return 'trial' if trial_id.startswith('trial') else {'a2s': 'breadth', 'a3m': 'matches', 'a4g': 'gi'}.get(trial_id[:3], 'other')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    manifest = json.loads((a.export/'manifest.json').read_text())
    frames = jsonl(a.export/'frames.jsonl')
    intervals = jsonl(a.export/'intervals.jsonl')
    events = jsonl(a.export/'events.jsonl')
    strikes = jsonl(a.export/'strikes.jsonl')
    summary = {
        'run': manifest['run'], 'created_at': manifest['created_at'], 'taxonomy_sha256': manifest['taxonomy_sha256'],
        'intervals': manifest['intervals'], 'images': manifest['images'], 'instances': manifest['instances'],
        'images_with_review_flags': manifest['images_with_review_flags'],
        'sources': len({i['source_key'] for i in intervals}),
        'intervals_by_cohort': dict(Counter(cohort(i['trial_id']) for i in intervals).most_common()),
        'sources_by_cohort': dict(Counter(cohort(t) for t in {i['trial_id'] for i in intervals}).most_common()),
        'intervals_by_clip_context': dict(Counter(i['clip_context'] for i in intervals).most_common()),
        'frames_by_quality_tier': dict(Counter(f['quality_tier'] for f in frames).most_common()),
        'frames_by_position': dict(Counter(f['position'] for f in frames).most_common()),
        'events': len(events), 'events_by_type': dict(Counter(e['type'] for e in events).most_common()),
        'events_crossing_shot_cut': sum(bool(e.get('crosses_shot_cut')) for e in events),
        'strikes': len(strikes), 'strikes_by_type': dict(Counter(s['type'] for s in strikes).most_common()),
        'strikes_by_result': dict(Counter(s['result'] for s in strikes).most_common()),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
