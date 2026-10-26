"""Stage 3: fuse SAM3.1 detections with Astra's per-frame identities into final per-frame labels.

Phases (each loops over all targets so at most one model is resident on the 6 GiB GPU):
  instances  CPU: Astra mark verdicts over SAM3.1 detections for every frame; Astra boxes (missed / merged people)
             converted to frame coordinates and kept mask-less
  resolve    GPU (.venv-sam31): Astra box -> single-frame SAM3.1 tracker-decoder mask; per-frame pixel exclusivity;
             review flags; writes labels.json
Outputs: autolabel/<run>/<trial>/<packet>/{frame_instances.json, labels.json}

SAM3.1 video propagation is not used: through the tracker-only adapter, propagated masks collapse onto one region for
all objects from the first non-prompt frame. Identity across frames comes from Astra (see docs/autolabel/CALIBRATION.md).
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from autolabel_common import SAM31_CKPT, box_iou, interval_dir, mask_box, rle_decode, rle_encode, sha256, write_json

# The tracker-only SAM 3.1 adapter (`openai_sam_smoke.sam31_runner.build_tracker_only_sam31`) comes from a separate,
# unpublished project and is not included here. Point KAKASHI_SAM31_ADAPTER at the directory that contains it.
QWEN = Path(os.environ.get('KAKASHI_SAM31_ADAPTER', Path.home()/'Code/kakashi-research/Qwen3.8-VL-SEG/phase1-grounding-audit'))
TRACKED_ROLES = {'athlete', 'referee', 'instructor', 'coach_or_corner'}
RISK_FLAGS = {'merged_detection_split_by_astra_boxes'}  # elevated error risk (~2/3 had ownership errors in audit)


def load_interval(run_name, trial_id, packet_id):
    base = interval_dir(run_name, trial_id, packet_id)
    return base, json.loads((base/'detections.json').read_text()), json.loads((base/'astra.json').read_text())


def clip_box(box, w, h):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    box = [max(0, min(x1, x2)), max(0, min(y1, y2)), min(w, max(x1, x2)), min(h, max(y1, y2))]
    return box if box[2] - box[0] >= 4 and box[3] - box[1] >= 4 else None


def sheet_to_frame(coords, sheet, frame_label):
    """Convert flat [x, y, x, y, ...] sheet coordinates to frame coordinates."""
    slot = sheet['frames'].index(frame_label)
    ox, oy = sheet['tile_origins_xy'][slot]
    s = 1/sheet['scale']
    return [(v - (ox if k % 2 == 0 else oy))*s for k, v in enumerate(coords)]


def frame_points(points, w, h, sheet=None, frame_label=None):
    out = []
    for pt in points or []:
        if len(pt) != 2:
            continue
        x, y = sheet_to_frame(pt, sheet, frame_label) if sheet else pt
        if 0 <= x < w and 0 <= y < h:
            out.append([int(round(x)), int(round(y))])
    return out


# ---------------------------------------------------------------- instances

def frame_instances(det, astra):
    result = astra['result']
    roles = {a['actor_id']: a['role'] for a in result['actors']}
    labelled = {f['frame']: f for f in result['keyframes'] + result['frames']}
    sheet_of = {fr: s for s in astra['tile_sheets'] for fr in s['frames']}
    out = []
    for meta in astra['frames']:
        frame = det['frames'][meta['frame_index']]
        w, h = frame['width'], frame['height']
        labels = labelled.get(meta['frame'])
        record = {'frame': meta['frame'], 'frame_index': meta['frame_index'], 'frame_id': meta['frame_id'],
                  'is_keyframe': meta['is_keyframe'], 'instances': [], 'merged_marks': [], 'flags': []}
        if labels is None:
            record |= {'shot_type': 'unknown', 'pair_configuration': 'unknown'}
            record['flags'].append('astra_returned_no_labels_for_frame')
            out.append(record)
            continue
        record |= {'shot_type': labels['shot_type'], 'pair_configuration': labels['pair_configuration']}
        mark_to_det = meta['mark_to_detection_index']
        seen = set()
        for m in labels['marks']:
            if not 1 <= m['mark'] <= len(mark_to_det) or m['mark'] in seen:
                record['flags'].append(f"invalid_or_repeated_mark_{m['mark']}")
                continue
            seen.add(m['mark'])
            d = frame['detections'][mark_to_det[m['mark'] - 1]]
            if m['verdict'] == 'merged':
                record['merged_marks'].append({'mark': m['mark'], 'actor_ids': m['merged_actor_ids']})
            if m['verdict'] != 'person':
                continue
            actor = m['actor_id'] or 'crowd'
            record['instances'].append({
                'actor_id': actor if actor != 'crowd' else f"crowd-{meta['frame']}-m{m['mark']}",
                'role': roles.get(actor, 'spectator_or_crew' if actor == 'crowd' else 'unknown'),
                'posture': m['posture'], 'source': 'sam31_text_detection', 'mark': m['mark'],
                'score': d['score'], 'mask_rle': d['mask_rle']})
        unlabelled = sorted(set(range(1, len(mark_to_det) + 1)) - seen)
        if unlabelled:
            record['flags'].append(f'marks_without_verdict={unlabelled}')
        for u in labels['unmarked_people']:
            if 'box_xyxy_sheet' in u:
                sheet = sheet_of[meta['frame']]
                box = clip_box(sheet_to_frame(u['box_xyxy_sheet'], sheet, meta['frame']), w, h) if len(u['box_xyxy_sheet']) == 4 else None
                points = frame_points(u.get('positive_points'), w, h, sheet, meta['frame'])
            else:
                box = clip_box(u['box_xyxy'], w, h) if len(u['box_xyxy']) == 4 else None
                points = frame_points(u.get('positive_points'), w, h)
            if box is None:
                record['flags'].append(f"{u['actor_id']}:unusable_astra_box")
                continue
            actor = u['actor_id']
            record['instances'].append({
                'actor_id': actor if actor != 'crowd' else f"crowd-{meta['frame']}-u{len(record['instances'])}",
                'role': roles.get(actor, 'spectator_or_crew' if actor == 'crowd' else 'unknown'),
                'posture': u['posture'], 'source': 'astra_box_to_sam31_mask', 'astra_box_xyxy': box,
                'astra_positive_points': points, 'score': 0.0, 'mask_rle': None})
        if record['merged_marks']:
            record['flags'].append('merged_detection_split_by_astra_boxes')
        ids = [i['actor_id'] for i in record['instances'] if not i['actor_id'].startswith('crowd')]
        for actor in sorted({a for a in ids if ids.count(a) > 1}):
            record['flags'].append(f'{actor}:assigned_twice')
        out.append(record)
    return out


def run_instances(run_name, targets, force):
    for trial_id, packet_id in targets:
        base, det, astra = load_interval(run_name, trial_id, packet_id)
        path = base/'frame_instances.json'
        if path.exists() and not force:
            continue
        frames = frame_instances(det, astra)
        write_json(path, {'version': 2, 'stage': 'frame_instances', 'trial_id': trial_id, 'packet_id': packet_id,
                          'astra_sha256': sha256(base/'astra.json'), 'detections_sha256': sha256(base/'detections.json'),
                          'frames': frames})
        print(f'{trial_id}/{packet_id}: instances', sum(len(f['instances']) for f in frames),
              'astra boxes', sum(i['mask_rle'] is None for f in frames for i in f['instances']), flush=True)


# ---------------------------------------------------------------- resolve

def exclusive(instances, shape):
    """Each pixel belongs to at most one instance: clean detections, then Astra box/point masks, then detections
    that contain another actor's point; ties by score."""
    def rank(inst):
        if inst['source'] != 'sam31_text_detection':
            return 1
        return 2 if inst.get('contains_other_actor_point') else 0
    order = sorted(range(len(instances)), key=lambda i: (rank(instances[i]), -instances[i]['score']))
    taken = np.zeros(shape, dtype=bool)
    for i in order:
        mask = instances[i]['mask']
        instances[i]['pixels_removed_by_overlap'] = int((mask & taken).sum())
        instances[i]['mask'] = mask & ~taken
        taken |= instances[i]['mask']


def box_masks(predictor, frame_dir, pending, shape):
    """Single-frame SAM3.1 tracker-decoder masks: box corners (2/3) + own positive points (1) + other people's
    positive points in the same frame as negatives (0). pending: [(frame_index, box, positives, negatives)]."""
    h, w = shape
    masks = []
    session = predictor.handle_request({'type': 'start_session', 'resource_path': str(frame_dir)})['session_id']
    try:
        for obj_id, (frame_index, box, positives, negatives) in enumerate(pending, start=1):
            x1, y1, x2, y2 = box
            points = [[x1/w, y1/h], [x2/w, y2/h]] + [[x/w, y/h] for x, y in positives + negatives]
            labels = [2, 3] + [1]*len(positives) + [0]*len(negatives)
            o = predictor.handle_request({
                'type': 'add_prompt', 'session_id': session, 'frame_index': frame_index, 'obj_id': obj_id,
                'points': points, 'point_labels': labels,
                'clear_old_points': True, 'rel_coordinates': True})['outputs']
            ids = [int(i) for i in o['out_obj_ids']]
            masks.append(np.asarray(o['out_binary_masks'][ids.index(obj_id)], dtype=bool))
    finally:
        predictor.handle_request({'type': 'close_session', 'session_id': session})
    return masks


def astra_mark_det(astra, frame_label, mark):
    meta = next(m for m in astra['frames'] if m['frame'] == frame_label)
    return meta['mark_to_detection_index'][mark - 1]


def dilate(mask, steps):
    out = mask.copy()
    for _ in range(steps):
        grown = out.copy()
        grown[1:] |= out[:-1]; grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]; grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def largest_component_share(mask):
    import cv2
    if not mask.any():
        return 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return round(float(areas.max()/areas.sum()), 3)


def speckle_share(mask):
    """Share of mask pixels in tiny components (<2% of the mask): speckle, as opposed to a body split by occlusion."""
    import cv2
    total = int(mask.sum())
    if not total:
        return 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return round(float(areas[areas < 0.02*total].sum()/total), 3)


def primary_actors(frames, actors):
    """The (up to) two athletes/instructors with the largest median visible area: the pair the clip is about."""
    roles = {a['actor_id']: a['role'] for a in actors}
    areas = {}
    for f in frames:
        for i in f['instances']:
            if roles.get(i['actor_id']) in ('athlete', 'instructor'):
                areas.setdefault(i['actor_id'], []).append(i['area'])
    ranked = sorted(areas, key=lambda a: -float(np.median(areas[a])))
    return set(ranked[:2])


def quality_flags(frames, primary):
    """Mask-quality checks for the primary pair: fragmentation, sudden area drop, oversized (partner-swallowing) mask.
    Flags about other actors are moved to info_flags."""
    series = {a: [next((i['area'] for i in f['instances'] if i['actor_id'] == a), None) for f in frames] for a in primary}
    for k, f in enumerate(frames):
        boxes = {i['actor_id']: i['box_xyxy'] for i in f['instances'] if i['actor_id'] in primary and i['box_xyxy']}
        for i in f['instances']:
            a = i['actor_id']
            if a not in primary or not i['area']:
                continue
            if i['speckle_share'] > 0.1:
                f['review_flags'].append(f'{a}:speckled_mask={i["speckle_share"]}')
            same_shot = [j for j in range(max(0, k - 3), min(len(frames), k + 4))
                         if j != k and frames[j]['shot_type'] == f['shot_type'] and series[a][j]]
            if len(same_shot) < 3:
                continue
            ref = float(np.median([series[a][j] for j in same_shot]))
            occluded = any(box_iou(boxes[a], b) > 0.2 for o, b in boxes.items() if o != a) if a in boxes else False
            if i['area'] < 0.5*ref:
                target = f['info_flags'] if occluded else f['review_flags']
                target.append(f'{a}:area_drop={round(i["area"]/ref, 2)}' + ('_with_partner_overlap' if occluded else ''))
            if i['area'] > 1.6*ref and i['source'] == 'sam31_text_detection' and occluded:
                f['review_flags'].append(f'{a}:oversized_mask_may_include_partner={round(i["area"]/ref, 2)}')
    for f in frames:
        keep, info = [], f['info_flags']
        for flag in f['review_flags']:
            actor = flag.split(':')[0]
            if actor.startswith('P') and actor not in primary and ':' in flag:
                info.append(flag)
            else:
                keep.append(flag)
        f['review_flags'] = keep


def colour_hist(path, small_mask, step=4):
    """RGB 64-bin histogram of the pixels under a mask already downsampled by `step`."""
    from PIL import Image
    with Image.open(path) as im:
        rgb = np.asarray(im.convert('RGB'))[::step, ::step]
    px = rgb[small_mask[:rgb.shape[0], :rgb.shape[1]]]
    if len(px) < 30:
        return None
    q = (px // 64).astype(int)
    h = np.bincount(q[:, 0]*16 + q[:, 1]*4 + q[:, 2], minlength=64).astype(float)
    return h/h.sum()


def appearance_flags(frames, masks, actors, primary):
    """Identity check: an athlete/instructor whose clothing colours match another actor's interval appearance better
    than their own. Returns the share of checked actor-frames that look swapped."""
    roles = {a['actor_id']: a['role'] for a in actors}
    hists = {}
    for f in frames:
        for i in f['instances']:
            if i['actor_id'] in primary and i['area'] > 2000:
                h = colour_hist(f['path'], masks[(f['frame_index'], id(i))])
                if h is not None:
                    hists.setdefault(i['actor_id'], []).append((f, h))
    means = {a: np.mean([h for _, h in v], axis=0) for a, v in hists.items()}
    checked = suspect = 0
    for a, items in hists.items():
        for f, h in items:
            own = np.minimum(h, means[a]).sum()
            other, who = max(((np.minimum(h, means[b]).sum(), b) for b in means if b != a), default=(0, None))
            checked += 1
            if other > own + 0.1:
                suspect += 1
                f['review_flags'].append(f'{a}:appearance_matches_{who}')
    return round(suspect/checked, 3) if checked else 0.0


def continuity_flags(frames):
    """Flag tracked actors that vanish for a single frame between two frames where they are present."""
    present = [{i['actor_id'] for i in f['instances'] if i['role'] in TRACKED_ROLES} for f in frames]
    for k in range(1, len(frames) - 1):
        for actor in (present[k - 1] & present[k + 1]) - present[k]:
            frames[k]['review_flags'].append(f'{actor}:missing_between_neighbours')


def run_resolve(run_name, targets, force):
    sys.path.insert(0, str(QWEN))
    from openai_sam_smoke.sam31_runner import build_tracker_only_sam31
    predictor = None
    for trial_id, packet_id in targets:
        base, det, astra = load_interval(run_name, trial_id, packet_id)
        if (base/'labels.json').exists() and not force:
            continue
        if predictor is None:
            predictor = build_tracker_only_sam31(SAM31_CKPT, device='cuda')
        fdoc = json.loads((base/'frame_instances.json').read_text())
        frame_dir = Path(det['frames'][0]['path']).parent
        names = sorted(p.name for p in frame_dir.glob('*.jpg'))
        if names != [Path(f['path']).name for f in det['frames']]:
            # match chunks share one frames folder; give the tracker exactly this packet's frames, in order
            frame_dir = base/'frames-link'
            if frame_dir.exists():
                shutil.rmtree(frame_dir)
            frame_dir.mkdir()
            for j, f in enumerate(det['frames']):
                (frame_dir/f'{j:05d}.jpg').symlink_to(f['path'])
        h, w = det['frames'][0]['height'], det['frames'][0]['width']
        pending, prompts = [], []
        for f in fdoc['frames']:
            boxed = [i for i in f['instances'] if i['mask_rle'] is None]
            for inst in boxed:
                own = inst.get('astra_positive_points', [])
                others = [pt for o in boxed if o is not inst and not o['actor_id'].startswith('crowd')
                          for pt in o.get('astra_positive_points', [])]
                pending.append((f, inst))
                prompts.append((f['frame_index'], inst['astra_box_xyxy'], own, others))
        masks = box_masks(predictor, frame_dir, prompts, (h, w)) if pending else []
        merged_regions = {}
        for f in fdoc['frames']:
            frame = det['frames'][f['frame_index']]
            region = np.zeros((h, w), dtype=bool)
            for m in f['merged_marks']:
                region |= rle_decode(frame['detections'][astra_mark_det(astra, f['frame'], m['mark'])]['mask_rle'])
            merged_regions[f['frame']] = (region, {a for m in f['merged_marks'] for a in m['actor_ids']})
        for (f, inst), mask in zip(pending, masks):
            box = inst['astra_box_xyxy']
            region, merged_actors = merged_regions[f['frame']]
            if inst['actor_id'] in merged_actors and region.any():
                clipped = mask & dilate(region, 6)
                inst['clipped_to_merged_region_px'] = int(mask.sum() - clipped.sum())
                mask = clipped
            pts = inst.get('astra_positive_points', [])
            inst['positive_points_inside_mask'] = sum(bool(mask[y, x]) for x, y in pts)
            inst['mask'] = mask
            inst['mask_inside_astra_box'] = round(float(mask[box[1]:box[3], box[0]:box[2]].sum()/max(1, mask.sum())), 3)
            inst['mask_box_iou_with_astra_box'] = round(box_iou(mask_box(mask) or [0, 0, 0, 0], box), 3)
        frames_out, kept_masks = [], {}
        for f in fdoc['frames']:
            frame = det['frames'][f['frame_index']]
            flags = list(f['flags'])
            for inst in f['instances']:
                if 'mask' not in inst:
                    inst['mask'] = rle_decode(inst['mask_rle'])
            for inst in f['instances']:
                if inst['source'] != 'sam31_text_detection' or inst['role'] not in TRACKED_ROLES:
                    continue
                foreign = [pt for o in f['instances'] if o['actor_id'] != inst['actor_id']
                           for pt in o.get('astra_positive_points', [])]
                if any(inst['mask'][y, x] for x, y in foreign):
                    inst['contains_other_actor_point'] = True
                    flags.append(f"{inst['actor_id']}:detection_contains_other_actor_point")
            exclusive(f['instances'], (h, w))
            for inst in f['instances']:
                mask = inst.pop('mask')
                kept_masks[(f['frame_index'], id(inst))] = mask[::4, ::4]
                inst.pop('mark', None)
                inst['largest_component_share'] = largest_component_share(mask)
                inst['speckle_share'] = speckle_share(mask)
                inst['box_xyxy'] = mask_box(mask)
                inst['area'] = int(mask.sum())
                inst['mask_rle'] = rle_encode(mask)
                tracked = inst['role'] in TRACKED_ROLES
                if tracked and inst.get('mask_box_iou_with_astra_box', 1) < 0.5:
                    flags.append(f"{inst['actor_id']}:astra_box_vs_mask_iou={inst['mask_box_iou_with_astra_box']}")
                if tracked and inst.get('astra_positive_points') and inst['positive_points_inside_mask'] < len(inst['astra_positive_points']):
                    flags.append(f"{inst['actor_id']}:own_point_outside_mask")
                if tracked and inst['pixels_removed_by_overlap'] > inst['area']:
                    flags.append(f"{inst['actor_id']}:mask_mostly_lost_to_overlap")
            frames_out.append({'frame_id': f['frame_id'], 'frame_index': f['frame_index'], 'time_s': frame['time_s'],
                               'path': frame['path'], 'width': w, 'height': h, 'is_keyframe': f['is_keyframe'],
                               'shot_type': f['shot_type'], 'pair_configuration': f['pair_configuration'],
                               'instances': f['instances'],
                               'review_flags': [x for x in flags if x not in RISK_FLAGS],
                               'risk_flags': [x for x in flags if x in RISK_FLAGS], 'info_flags': []})
        continuity_flags(frames_out)
        result = astra['result']
        primary = primary_actors(frames_out, result['actors'])
        identity_suspect_share = appearance_flags(frames_out, kept_masks, result['actors'], primary)
        quality_flags(frames_out, primary)
        for a in result['actors']:
            a['primary'] = a['actor_id'] in primary
        tracked_instances = [i for f in frames_out for i in f['instances'] if i['role'] in TRACKED_ROLES]
        summary = {'frames': len(frames_out), 'actors': [a['actor_id'] for a in result['actors']],
                   'instances': sum(len(f['instances']) for f in frames_out),
                   'tracked_actor_instances': len(tracked_instances),
                   'from_detection': sum(i['source'] == 'sam31_text_detection' for i in tracked_instances),
                   'from_astra_box': sum(i['source'] == 'astra_box_to_sam31_mask' for i in tracked_instances),
                   'frames_with_flags': sum(bool(f['review_flags']) for f in frames_out),
                   'frames_with_risk_flags': sum(bool(f['risk_flags']) for f in frames_out),
                   'identity_suspect_share': identity_suspect_share,
                   'identity_unstable': identity_suspect_share > 0.15}
        write_json(base/'labels.json', {
            'version': 2, 'stage': 'autolabel_labels', 'trial_id': trial_id, 'packet_id': packet_id,
            'source_key': det['source_key'], 'time_range_s': det['time_range_s'],
            'taxonomy': 'autolabel-taxonomy-v1', 'clip_context': result['clip_context'],
            'actors': result['actors'], 'astra_notes': result['notes'],
            'provenance': {'detections': 'sam3.1 image detector, text prompt "person"',
                           'identities_labels': 'gpt-6-astra set-of-marks over every frame (see astra-receipt.json)',
                           'box_masks': 'sam3.1 tracker decoder, single frame, box corners + own positive points + other people negative points; clipped to merged detection region',
                           'astra_sha256': sha256(base/'astra.json'), 'detections_sha256': sha256(base/'detections.json')},
            'created_at': datetime.now(timezone.utc).isoformat(), 'summary': summary, 'frames': frames_out})
        print(f'{trial_id}/{packet_id}: {summary}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['instances', 'resolve'])
    ap.add_argument('--run', default='v1')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('targets', nargs='+', help='trialNNN/intervalNN')
    a = ap.parse_args()
    {'instances': run_instances, 'resolve': run_resolve}[a.phase](a.run, [tuple(t.split('/')) for t in a.targets], a.force)
