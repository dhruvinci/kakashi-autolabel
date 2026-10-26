"""Export autolabel labels.json files as training views (runs in .venv-sam31 for pycocotools RLE compression).

  instances.coco.json   COCO instance segmentation: category 'person'; actor/role/posture/source as attributes;
                        every visible detected person is an instance (spectators included, never silent background)
  frames.jsonl          frame classification: shot_type, pair_configuration (+ review flags)
  intervals.jsonl       interval classification: clip_context, actors
  manifest.json         counts, flagged-frame policy, provenance hashes
Frames with review flags are kept and marked (image['review_flags']); filter them for a high-precision subset.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from autolabel_common import OUT, ROOT, TAXONOMY, rle_decode, sha256, write_json


def compress(rle):
    encoded = mask_utils.encode(np.asfortranarray(rle_decode(rle).astype(np.uint8)))
    return {'size': encoded['size'], 'counts': encoded['counts'].decode('ascii')}


def export(run, output):
    tax = json.loads(TAXONOMY.read_text())
    label_files = sorted((OUT/run).glob('*/*/labels.json'))
    images, annotations, frame_rows, interval_rows, hashes = [], [], [], [], {}
    for path in label_files:
        labels = json.loads(path.read_text())
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        interval_rows.append({'trial_id': labels['trial_id'], 'packet_id': labels['packet_id'],
                              'source_key': labels['source_key'], 'time_range_s': labels['time_range_s'],
                              'clip_context': labels['clip_context'], 'actors': labels['actors']})
        primary = {a['actor_id'] for a in labels['actors'] if a.get('primary')}
        base = path.parent
        verify = {r['frame_index']: r for r in json.loads((base/'verify.json').read_text()).get('frames', [])} \
            if (base/'verify.json').exists() else {}
        smooth = json.loads((base/'smooth.json').read_text()) if (base/'smooth.json').exists() else {}
        signals = json.loads((base/'mask_signals.json').read_text())['signals'] if (base/'mask_signals.json').exists() else {}
        fight = json.loads((base/'fight.json').read_text())['result'] if (base/'fight.json').exists() else None
        for frame in labels['frames']:
            image_id = len(images) + 1
            v = verify.get(frame['frame_index'])
            # Quality tier (rules calibrated by a blind Claude adjudication of 60 disputed frames, 2026-10-04):
            #   clean: no review/risk flags; astra_verified: flagged but Astra's second opinion says both masks are
            #   correct (Claude sided with Astra on 20/30 such disputes); rejected: Astra confirms a mask problem.
            if not frame['review_flags'] and not frame.get('risk_flags'):
                tier = 'clean'
            elif v is None:
                tier = 'flagged_unverified'
            else:
                tier = 'astra_verified' if v['masks_ok'] else 'rejected'
            # mask_fix.py (2026-10-11) swapped a primary fallback mask for the dense video mask: Astra's verdict judged
            # the old mask, so it no longer applies. Speckle-only cleanups keep the old verdict (ownership unchanged).
            if v is not None and any(i.get('maskfix', {}).get('action') == 'dense' and i['actor_id'] in primary
                                     for i in frame['instances']):
                tier = 'dense_replaced_unverified'
            k = frame['frame_index']
            position = smooth.get('position', {}).get('smoothed', [None]*(k + 1))[k] if smooth else None
            if v is not None and not v['position_ok']:
                position = 'transition_or_uncertain'  # Astra vs pipeline position disputes were mostly genuinely ambiguous
            fight_frame = dict(fight['frames'][k]) if fight and k < len(fight['frames']) else {}
            sp = smooth.get('position', {})
            if fight_frame and 'detail' in sp and k < len(sp['detail']):  # detail/top consistent with the smoothed position
                fight_frame['position_detail'], fight_frame['top_actor'] = sp['detail'][k], sp['top'][k]
            images.append({'id': image_id, 'file_name': str(Path(frame['path']).relative_to(ROOT)),
                           'width': frame['width'], 'height': frame['height'],
                           'frame_id': f"{labels['trial_id']}/{labels['packet_id']}/{frame['frame_id']}",
                           'source_key': labels['source_key'], 'time_s': frame['time_s'],
                           'is_keyframe': frame['is_keyframe'], 'review_flags': frame['review_flags'],
                           'risk_flags': frame.get('risk_flags', []), 'info_flags': frame.get('info_flags', [])})
            images[-1]['quality_tier'] = tier
            frame_rows.append({'image_id': image_id, 'frame_id': images[-1]['frame_id'], 'file_name': images[-1]['file_name'],
                               'shot_type': frame['shot_type'], 'pair_configuration': frame['pair_configuration'],
                               'clip_context': labels['clip_context'], 'quality_tier': tier,
                               'position': position, 'position_detail': fight_frame.get('position_detail'),
                               'engagement': fight_frame.get('engagement'), 'top_actor': fight_frame.get('top_actor'),
                               'dominant_actor': fight_frame.get('dominant_actor'),
                               'review_flags': frame['review_flags']})
            for inst in frame['instances']:
                if not inst['area']:
                    continue
                x1, y1, x2, y2 = inst['box_xyxy']
                annotations.append({
                    'id': len(annotations) + 1, 'image_id': image_id, 'category_id': 1, 'iscrowd': 0,
                    'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': inst['area'], 'segmentation': compress(inst['mask_rle']),
                    'attributes': {'actor_id': None if str(inst['actor_id']).startswith('crowd') else inst['actor_id'],
                                   'track_id': None if str(inst['actor_id']).startswith('crowd')
                                   else f"{labels['trial_id']}/{labels['packet_id']}/{inst['actor_id']}",
                                   'role': inst['role'], 'posture': inst['posture'], 'source': inst['source'],
                                   'primary': inst['actor_id'] in primary,
                                   'largest_component_share': inst.get('largest_component_share'),
                                   'maskfix': (inst.get('maskfix') or {}).get('action'),
                                   # uncalibrated mask-quality signals (src/mask_signals.py); thresholds to be set from the gold set
                                   'mask_signals': {k: v for k, v in signals.get(f"{frame['frame_index']}:{inst['actor_id']}", {}).items()
                                                    if k in ('off_joints', 'partner_joints', 'conf_joints', 'lcs', 'contact')} or None}})
    output.mkdir(parents=True, exist_ok=True)
    coco = {'info': {'description': 'Kakashi autolabel (gpt-6-astra set-of-marks + local SAM3.1)', 'run': run,
                     'created': datetime.now(timezone.utc).isoformat(), 'taxonomy': tax['version']},
            'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}],
            'attribute_vocab': {'role': tax['person']['role']['values'], 'posture': tax['person']['posture']['values']},
            'images': images, 'annotations': annotations}
    (output/'instances.coco.json').write_text(json.dumps(coco))
    (output/'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in frame_rows))
    (output/'intervals.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in interval_rows))
    events = []
    for path in label_files:
        fp = path.parent/'fight.json'
        if fp.exists():
            fd = json.loads(fp.read_text())
            times = {f['frame']: f['time_s'] for f in fd['frames']}
            sm = json.loads((path.parent/'smooth.json').read_text()) if (path.parent/'smooth.json').exists() else {}
            cut_frames = {f"F{c:02d}" for c in sm.get('shot_cuts', [])}
            order = [f['frame'] for f in fd['frames']]
            for e in fd['result']['events']:
                row = e | {'trial_id': fd['trial_id'], 'packet_id': fd['packet_id'],
                           'start_s': times.get(e['start_frame']), 'end_s': times.get(e['end_frame'])}
                if e['start_frame'] in order and e['end_frame'] in order:
                    span = order[order.index(e['start_frame']) + 1:order.index(e['end_frame']) + 1]
                    row['crosses_shot_cut'] = bool(cut_frames & set(span))
                    # takedown/sweep/back_take/mount_achieve without an actor: whoever is on top at the end did it
                    if not e.get('actor') and e['type'] in ('takedown', 'sweep', 'back_take', 'mount_achieve', 'pass') \
                            and sm.get('position', {}).get('top'):
                        k = order.index(e['end_frame'])
                        row['actor_inferred'] = sm['position']['top'][k] if k < len(sm['position']['top']) else None
                events.append(row)
    (output/'events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in events))
    strikes = []
    for path in label_files:
        sp = path.parent/'strikes.json'
        if sp.exists():
            sd = json.loads(sp.read_text())
            for st in sd.get('strikes', []):
                strikes.append(st | {'trial_id': path.parent.parts[-2], 'packet_id': path.parent.parts[-1]})
    (output/'strikes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in strikes))
    write_json(output/'manifest.json', {
        'run': run, 'created_at': datetime.now(timezone.utc).isoformat(), 'taxonomy_sha256': sha256(TAXONOMY),
        'intervals': len(interval_rows), 'images': len(images), 'instances': len(annotations),
        'images_with_review_flags': sum(bool(i['review_flags']) for i in images),
        'flag_policy': 'flagged frames are exported and marked; drop them for a high-precision subset',
        'split': 'dev only (trial100 calibration sources); not a held-out test set',
        'label_files_sha256': hashes})
    print(json.dumps({'intervals': len(interval_rows), 'images': len(images), 'instances': len(annotations)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v1')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    export(a.run, a.output)
