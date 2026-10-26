"""Corpus-level quality review of autolabel labels.json files (CPU, .venv).

Measures coverage, flag rates, label distributions, temporal stability, role sanity, identity-swap suspicion
(appearance), completeness against unlabelled high-confidence detections, Astra box/mask agreement, agreement with
the earlier GPT-6 trial100 geometry, and Astra cost. Writes reviews/autolabel-v1/metrics.json.
"""
import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from autolabel_astra import cohort_source
from autolabel_common import OUT, ROOT, box_iou, rle_decode, write_json

TRACKED = {'athlete', 'referee', 'instructor', 'coach_or_corner'}
STEP = 4


def hist(path, mask):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert('RGB'))[::STEP, ::STEP]
    px = rgb[mask[::STEP, ::STEP][:rgb.shape[0], :rgb.shape[1]]]
    if len(px) < 30:
        return None
    q = (px // 64).astype(int)
    h = np.bincount(q[:, 0]*16 + q[:, 1]*4 + q[:, 2], minlength=64).astype(float)
    return h/h.sum()


def flicker(seq):
    """Changes between consecutive values, and A-B-A single-frame blips."""
    changes = sum(a != b for a, b in zip(seq, seq[1:]))
    blips = sum(seq[i - 1] == seq[i + 1] != seq[i] for i in range(1, len(seq) - 1))
    return changes, blips


def pct(a, b):
    return round(100*a/b, 1) if b else None


def review(run='v1'):
    m = collections.defaultdict(collections.Counter)
    per_interval, ious_prior, box_mask_ious, swap_cases, big_unlabelled_examples = [], [], [], [], []
    files = sorted((OUT/run).glob('trial*/interval*/labels.json'))
    for path in files:
        L = json.loads(path.read_text())
        base = path.parent
        src = cohort_source(L['trial_id'])
        cell = f"{src.get('sport_hypothesis')}/{src.get('context_hypothesis')}"
        det = json.loads((base/'detections.json').read_text())
        astra = json.loads((base/'astra.json').read_text())
        receipt = json.loads((base/'astra-receipt.json').read_text())
        frames = L['frames']
        roles = collections.Counter(a['role'] for a in L['actors'])
        row = {'interval': f"{L['trial_id']}/{L['packet_id']}", 'cell': cell, 'clip_context': L['clip_context'],
               'frames': len(frames), 'roles': dict(roles),
               'flagged_frames': sum(bool(f['review_flags']) for f in frames),
               'astra_s': receipt['elapsed_s'], 'tokens_in': (receipt['usage'] or {}).get('input_tokens'),
               'tokens_out': (receipt['usage'] or {}).get('output_tokens')}
        m['cells'][cell] += 1
        m['clip_context'][f"{cell} -> {L['clip_context']}"] += 1
        m['totals']['intervals'] += 1
        m['totals']['frames'] += len(frames)
        m['totals']['flagged_frames'] += row['flagged_frames']
        unstable = L['summary'].get('identity_unstable', False)
        m['totals']['identity_unstable_intervals'] += unstable
        for f in frames:
            risk = bool(f.get('risk_flags'))
            m['totals']['risk_frames'] += risk
            hp = not unstable and not f['review_flags']
            m['totals']['high_precision_frames'] += hp
            m['totals']['strict_frames'] += hp and not risk
            m['totals']['high_precision_instances'] += len(f['instances']) if hp else 0
        m['flagged_by_cell'][cell] += row['flagged_frames']
        m['frames_by_cell'][cell] += len(frames)
        for f in frames:
            for flag in f['review_flags']:
                m['flag_types'][flag.split(':')[-1].split('=')[0].split('[')[0]] += 1
            m['shot_type'][f['shot_type']] += 1
            m['pair_configuration'][f['pair_configuration']] += 1
            for i in f['instances']:
                m['totals']['instances'] += 1
                m['role'][i['role']] += 1
                if i['role'] in TRACKED:
                    m['totals']['tracked_instances'] += 1
                    m['posture'][i['posture']] += 1
                    m['source'][i['source']] += 1
                    if i.get('mask_box_iou_with_astra_box') is not None:
                        box_mask_ious.append(i['mask_box_iou_with_astra_box'])
                        m['points'][f"{i.get('positive_points_inside_mask', 0)}/{len(i.get('astra_positive_points', []))}"] += 1
        # frames Astra did not label
        labelled = {x['frame'] for x in astra['result']['keyframes'] + astra['result']['frames']}
        m['totals']['frames_missing_astra_labels'] += sum(fr['frame'] not in labelled for fr in astra['frames'])

        # temporal stability
        for key in ('pair_configuration', 'shot_type'):
            ch, bl = flicker([f[key] for f in frames])
            m[f'{key}_stability']['changes'] += ch
            m[f'{key}_stability']['single_frame_blips'] += bl
            m[f'{key}_stability']['transitions_possible'] += len(frames) - 1
        actors = sorted({i['actor_id'] for f in frames for i in f['instances'] if i['role'] in TRACKED})
        for actor in actors:
            seq = [next((i['posture'] for i in f['instances'] if i['actor_id'] == actor), None) for f in frames]
            seq = [s for s in seq if s is not None]
            ch, bl = flicker(seq)
            m['posture_stability']['changes'] += ch
            m['posture_stability']['single_frame_blips'] += bl
            m['posture_stability']['transitions_possible'] += max(0, len(seq) - 1)
            m['actor_presence']['actor_frames_present'] += len(seq)
            m['actor_presence']['actor_frames_possible'] += len(frames)

        # role sanity
        athletes = roles.get('athlete', 0)
        m['athletes_per_interval'][athletes] += 1
        if src.get('context_hypothesis') == 'competition' and athletes > 2:
            m['role_sanity']['competition_with_gt2_athletes'] += 1
        if src.get('context_hypothesis') != 'competition' and roles.get('referee'):
            m['role_sanity']['noncompetition_with_referee'] += 1
        if src.get('context_hypothesis') == 'tutorial' and not roles.get('instructor'):
            m['role_sanity']['tutorial_without_instructor'] += 1

        # identity swap suspicion among athletes/instructors (appearance vs own mean vs other actors' mean)
        main = [a for a in actors if any(x['actor_id'] == a and x['role'] in ('athlete', 'instructor') for x in L['actors'])]
        if len(main) >= 2:
            hists = collections.defaultdict(list)
            for f in frames[::2]:
                for i in f['instances']:
                    if i['actor_id'] in main and i['area'] > 2000:
                        h = hist(f['path'], rle_decode(i['mask_rle']))
                        if h is not None:
                            hists[i['actor_id']].append((f['frame_index'], h))
            means = {a: np.mean([h for _, h in v], axis=0) for a, v in hists.items() if v}
            for a, v in hists.items():
                for idx, h in v:
                    own = np.minimum(h, means[a]).sum()
                    other = max((np.minimum(h, means[b]).sum() for b in means if b != a), default=0)
                    m['identity']['checked'] += 1
                    if other > own + 0.1:
                        m['identity']['closer_to_other_actor'] += 1
                        swap_cases.append({'interval': row['interval'], 'frame_index': idx, 'actor': a,
                                           'own': round(float(own), 2), 'other': round(float(other), 2)})

        # completeness: high-confidence large detections not covered by any instance
        for f, df in zip(frames, det['frames']):
            covered = np.zeros((f['height'], f['width']), dtype=bool)
            for i in f['instances']:
                covered |= rle_decode(i['mask_rle'])
            for d in df['detections']:
                if d['score'] < 0.7 or d['area'] < 0.01*f['height']*f['width']:
                    continue
                m['completeness']['confident_large_detections'] += 1
                if (rle_decode(d['mask_rle']) & covered).sum() < 0.3*d['area']:
                    m['completeness']['not_covered_by_any_instance'] += 1
                    if len(big_unlabelled_examples) < 40:
                        big_unlabelled_examples.append({'interval': row['interval'], 'frame_index': f['frame_index'],
                                                        'score': d['score'], 'box': d['box_xyxy']})

        # agreement with the earlier GPT-6 trial100 geometry (person entities only)
        prior = ROOT/'annotations/trial100-v1'/L['trial_id']/f"{L['packet_id']}.json"
        if prior.exists():
            P = json.loads(prior.read_text())
            kinds = {a['actor_id']: a.get('entity_type') for a in P.get('actors', [])}
            by_id = {f['frame_id']: f for f in frames}
            for g in P.get('geometry', []):
                f, b = by_id.get(g['frame_id']), g.get('bbox_xyxy_normalized')
                if not f or not b or kinds.get(g['actor_id']) != 'person':
                    continue
                ob = [b[0]*f['width'], b[1]*f['height'], b[2]*f['width'], b[3]*f['height']]
                ious_prior.append(max([box_iou(ob, i['box_xyxy']) for i in f['instances'] if i['box_xyxy']] + [0]))
        m['cost']['tokens_in'] += row['tokens_in'] or 0
        m['cost']['tokens_out'] += row['tokens_out'] or 0
        m['cost']['astra_seconds'] += row['astra_s']
        per_interval.append(row)

    t = m['totals']
    summary = {
        'intervals': t['intervals'], 'frames': t['frames'], 'instances': t['instances'],
        'tracked_instances': t['tracked_instances'],
        'flagged_frames_pct': pct(t['flagged_frames'], t['frames']),
        'flagged_pct_by_cell': {c: pct(m['flagged_by_cell'][c], m['frames_by_cell'][c]) for c in sorted(m['frames_by_cell'])},
        'frames_missing_astra_labels': t['frames_missing_astra_labels'],
        'risk_frames_pct': pct(t['risk_frames'], t['frames']),
        'identity_unstable_intervals': t['identity_unstable_intervals'],
        'high_precision_frames': t['high_precision_frames'], 'high_precision_instances': t['high_precision_instances'],
        'strict_frames': t['strict_frames'],
        'prior_gpt6_person_boxes': len(ious_prior),
        'prior_match_iou50_pct': pct(sum(x >= 0.5 for x in ious_prior), len(ious_prior)),
        'prior_match_iou70_pct': pct(sum(x >= 0.7 for x in ious_prior), len(ious_prior)),
        'prior_median_iou': round(statistics.median(ious_prior), 3) if ious_prior else None,
        'astra_box_vs_mask_iou_median': round(statistics.median(box_mask_ious), 3) if box_mask_ious else None,
        'astra_box_vs_mask_iou_lt50_pct': pct(sum(x < 0.5 for x in box_mask_ious), len(box_mask_ious)),
        'pair_config_change_pct': pct(m['pair_configuration_stability']['changes'], m['pair_configuration_stability']['transitions_possible']),
        'pair_config_blip_pct': pct(m['pair_configuration_stability']['single_frame_blips'], m['pair_configuration_stability']['transitions_possible']),
        'posture_change_pct': pct(m['posture_stability']['changes'], m['posture_stability']['transitions_possible']),
        'posture_blip_pct': pct(m['posture_stability']['single_frame_blips'], m['posture_stability']['transitions_possible']),
        'identity_closer_to_other_pct': pct(m['identity']['closer_to_other_actor'], m['identity']['checked']),
        'confident_large_dets_uncovered_pct': pct(m['completeness']['not_covered_by_any_instance'], m['completeness']['confident_large_detections']),
        'actor_presence_pct': pct(m['actor_presence']['actor_frames_present'], m['actor_presence']['actor_frames_possible']),
        'astra_hours': round(m['cost']['astra_seconds']/3600, 1),
        'tokens_in_M': round(m['cost']['tokens_in']/1e6, 2), 'tokens_out_M': round(m['cost']['tokens_out']/1e6, 2)}
    out = {'summary': summary, 'counters': {k: dict(v) for k, v in m.items()},
           'identity_suspects': swap_cases[:200], 'uncovered_detection_examples': big_unlabelled_examples,
           'per_interval': per_interval}
    write_json(ROOT/'reviews/autolabel-v1/metrics.json', out)
    print(json.dumps(summary, indent=1))
    return out


if __name__ == '__main__':
    review(sys.argv[1] if len(sys.argv) > 1 else 'v1')
