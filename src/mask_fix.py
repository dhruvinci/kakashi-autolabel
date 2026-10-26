"""Mask fix 1: replace the speckled Astra-box fallback masks with the SAM 3.1 video (dense) mask at the same moment.

The `astra_box_to_sam31_mask` instances (single-frame SAM decoder from an Astra box, used when the detector merged or missed
a person) were 87% bad in the gold post-mortem (reviews/gold-v1/mask_postmortem). The dense 12.5 fps track is propagated
from clean anchors and was better on most faulted frames. For each fallback instance of a tracked actor:
  - dense mask coherent (non-empty, largest component >= LCS_MIN, little speckle, sane area vs the old mask, little of it
    owned by clean detections) -> replace it (source = 'dense_propagated', replaced_source kept);
  - otherwise -> speckle cleanup of the old mask (drop components < 2% of the mask, fill small holes), cleaned = true.
Pixel exclusivity follows autolabel_fuse.exclusive: clean detections keep their pixels; a replaced mask takes pixels from
fallback masks and from detections that contain another actor's point. Frame review/risk flags are left as they are.

  ../.venv-pose/bin/python mask_fix.py --dry-run            # survey only, no writes
  ../.venv-pose/bin/python mask_fix.py                      # apply (backs up labels.json -> labels.pre-maskfix.json.gz once)
  ../.venv-pose/bin/python mask_fix.py --revert             # restore every backup
Summary: reviews/autolabel-v1/maskfix/summary.json
"""
import argparse
import collections
import gzip
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from autolabel_common import ROOT, mask_box, rle_decode, rle_encode, write_json
from autolabel_fuse import largest_component_share, speckle_share

RUN = ROOT/'autolabel/v1'
OUT = ROOT/'reviews/autolabel-v1/maskfix'
FALLBACK = 'astra_box_to_sam31_mask'
LCS_MIN = 0.85          # dense mask: share of its largest connected component
SPECKLE_MAX = 0.05      # dense mask: share of pixels in components < 2% of the mask
AREA_RATIO = (0.35, 3.0)  # dense area / old fallback area
LOST_MAX = 0.3          # share of the dense mask already owned by clean detections
PARTNER_MAX = 0.35      # share of the other primary actor's existing mask the candidate may cover (merge guard)
MIN_AREA = 400
MAX_DT = 0.06           # label time vs dense frame time (s)


def components(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n - 1, lab, stats[1:, cv2.CC_STAT_AREA]


def clean_speckle(mask, keep_frac=0.02, hole_frac=0.01):
    """Drop components smaller than keep_frac of the mask; fill holes smaller than hole_frac of the mask."""
    total = int(mask.sum())
    if not total:
        return mask
    n, lab, areas = components(mask)
    keep = np.flatnonzero(areas >= keep_frac*total) + 1
    out = np.isin(lab, keep) if len(keep) else mask.copy()
    inv = (~out).astype(np.uint8)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])).tolist())
    small = [j for j in range(1, n2) if j not in border and stats2[j, cv2.CC_STAT_AREA] < hole_frac*max(1, out.sum())]
    if small:
        out = out | np.isin(lab2, small)
    return out


def dense_lookup(base):
    d = json.loads((base/'dense/dense.json').read_text())
    z = np.load(base/'dense/masks.npz')
    times = np.array([f['time_s'] for f in d['frames']])
    return d, z, times


def dense_mask(d, z, times, t, actor, shape):
    k = int(np.argmin(np.abs(times - t)))
    if abs(times[k] - t) > MAX_DT:
        return None, k
    key = f'{k}_{actor}'
    if key not in z.files:
        return None, k
    h, w = d['shape']
    bits = np.unpackbits(z[key])
    if bits.size < h*w or (h, w) != tuple(shape):
        return None, k
    return bits[:h*w].reshape(h, w).astype(bool), k


def coherent(dm, old, owned_clean, partner=None):
    """(ok, metrics) for a candidate (dense or tracked) mask; partner = the other primary actor's existing mask."""
    area = int(dm.sum())
    if area < MIN_AREA:
        return False, {'reason': 'empty'}
    lcs = largest_component_share(dm)
    sp = speckle_share(dm)
    ratio = area/max(1, int(old.sum()))
    lost = float((dm & owned_clean).sum()/area)
    cover = float((dm & partner).sum()/max(1, partner.sum())) if partner is not None and partner.any() else 0.0
    m = {'dense_lcs': lcs, 'dense_speckle': sp, 'area_ratio': round(ratio, 3), 'lost_to_clean': round(lost, 3),
         'partner_covered': round(cover, 3)}
    for cond, why in ((cover <= PARTNER_MAX, 'swallows_partner'), (lcs >= LCS_MIN, 'fragmented'), (sp <= SPECKLE_MAX, 'speckled'),
                      (AREA_RATIO[0] <= ratio <= AREA_RATIO[1], 'area_ratio'), (lost <= LOST_MAX, 'owned_by_clean_detection')):
        if not cond:
            return False, m | {'reason': why}
    return True, m


def fix_interval(base, write):
    L = json.loads((base/'labels.json').read_text())
    if not any(i['source'] == FALLBACK and 'maskfix' not in i for f in L['frames'] for i in f['instances']):
        return collections.Counter(), [], L
    has_dense = (base/'dense/dense.json').exists() and (base/'dense/masks.npz').exists()
    d = z = times = None
    if has_dense:
        d, z, times = dense_lookup(base)
    tracked = set(d['tracked_actors']) if has_dense else set()
    primary = {a['actor_id'] for a in L['actors'] if a.get('primary')}
    counts, records = collections.Counter(), []
    for f in L['frames']:
        fb = [i for i in f['instances'] if i['source'] == FALLBACK and i.get('mask_rle') and i['area'] and 'maskfix' not in i]
        if not fb:
            continue
        shape = (f['height'], f['width'])
        masks = {id(i): rle_decode(i['mask_rle']) for i in f['instances'] if i.get('mask_rle')}
        clean = np.zeros(shape, bool)
        for i in f['instances']:
            if i['source'] == 'sam31_text_detection' and not i.get('contains_other_actor_point') and id(i) in masks:
                clean |= masks[id(i)]
        new = {}
        for inst in fb:
            old = masks[id(inst)]
            rec = {'iv': f"{base.parts[-2]}/{base.parts[-1]}", 'fi': f['frame_index'], 'actor': inst['actor_id'],
                   'primary': inst['actor_id'] in primary, 'old_area': int(old.sum()), 'old_lcs': inst.get('largest_component_share'),
                   'old_speckle': inst.get('speckle_share')}
            dm = None
            if inst['actor_id'] in tracked:
                dm, k = dense_mask(d, z, times, f['time_s'], inst['actor_id'], shape)
                rec['dense_k'] = k
            if dm is not None:
                partner = np.zeros(shape, bool)
                for o in f['instances']:
                    if o['actor_id'] in primary and o['actor_id'] != inst['actor_id'] and id(o) in masks:
                        partner |= masks[id(o)]
                ok, m = coherent(dm, old, clean, partner)
                rec.update(m)
                if ok:
                    new[id(inst)] = ('dense', dm & ~clean)
                    rec['action'] = 'dense'
                else:
                    rec['action'] = 'cleaned'
            else:
                rec['action'] = 'cleaned'
                rec['reason'] = 'no_dense_track' if inst['actor_id'] not in tracked else 'no_dense_frame'
            if rec['action'] == 'cleaned':
                new[id(inst)] = ('cleaned', clean_speckle(old) & ~clean)
            counts[rec['action'] + ('_primary' if rec['primary'] else '_other')] += 1
            records.append(rec)
        # exclusivity: replaced dense masks first (by actor id), then cleaned masks; take pixels from non-clean instances
        order = sorted(new, key=lambda x: (new[x][0] != 'dense', next(i['actor_id'] for i in fb if id(i) == x)))
        taken = clean.copy()
        final = {}
        for x in order:
            m = new[x][1] & ~taken
            final[x] = m
            taken |= m
        for inst in f['instances']:
            if id(inst) in final:
                continue
            if id(inst) in masks and inst['source'] != 'sam31_text_detection' or inst.get('contains_other_actor_point'):
                m = masks.get(id(inst))
                if m is not None and (m & taken).any():
                    m = m & ~taken
                    inst['pixels_lost_to_maskfix'] = int((masks[id(inst)] & taken).sum())
                    set_mask(inst, m)
        for inst in fb:
            m = final[id(inst)]
            action = new[id(inst)][0]
            old_area = inst['area']
            fix = {'action': action, 'old_area': old_area, 'old_largest_component_share': inst.get('largest_component_share'),
                   'old_speckle_share': inst.get('speckle_share'),
                   'iou_with_old': round(float((m & masks[id(inst)]).sum()/max(1, (m | masks[id(inst)]).sum())), 3)}
            rec = next(r for r in records if r['fi'] == f['frame_index'] and r['actor'] == inst['actor_id'])
            fix.update({k: rec[k] for k in ('dense_k', 'dense_lcs', 'dense_speckle', 'area_ratio', 'lost_to_clean', 'reason') if k in rec})
            set_mask(inst, m)
            inst['maskfix'] = fix
            if action == 'dense':
                inst['replaced_source'] = inst['source']
                inst['source'] = 'dense_propagated'
            else:
                inst['cleaned'] = True
            rec.update(new_area=inst['area'], new_lcs=inst['largest_component_share'], new_speckle=inst['speckle_share'],
                       iou_with_old=fix['iou_with_old'])
    if write and records:
        backup = base/'labels.pre-maskfix.json.gz'
        if not backup.exists():
            with open(base/'labels.json', 'rb') as src, gzip.open(backup, 'wb', compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
        L.setdefault('provenance', {})['maskfix'] = ('fallback astra-box masks replaced by the SAM 3.1 dense video mask when coherent, '
                                                     'else speckle-cleaned; see src/mask_fix.py; backup labels.pre-maskfix.json.gz')
        write_json(base/'labels.json', L)
    return counts, records, L


def set_mask(inst, m):
    inst['mask_rle'] = rle_encode(m)
    inst['area'] = int(m.sum())
    inst['box_xyxy'] = mask_box(m)
    inst['largest_component_share'] = largest_component_share(m)
    inst['speckle_share'] = speckle_share(m)


def original_labels(base):
    """labels.json as it was before mask_fix (the gzip backup when present)."""
    b = base/'labels.pre-maskfix.json.gz'
    if b.exists():
        with gzip.open(b, 'rt') as fh:
            return json.load(fh)
    return json.loads((base/'labels.json').read_text())


def revert():
    n = 0
    for b in sorted(RUN.glob('*/*/labels.pre-maskfix.json.gz')):
        with gzip.open(b, 'rb') as src, open(b.with_name('labels.json.tmp'), 'wb') as dst:
            shutil.copyfileobj(src, dst)
        b.with_name('labels.json.tmp').replace(b.with_name('labels.json'))
        b.unlink()
        n += 1
    print('restored', n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--only', nargs='*', help='trial/packet subset')
    ap.add_argument('--out', type=Path, default=OUT, help='summary/records directory (keep the full-run summary when fixing a subset)')
    a = ap.parse_args()
    if a.revert:
        return revert()
    bases = [RUN/t for t in a.only] if a.only else sorted(p.parent for p in RUN.glob('*/*/labels.json'))
    total, rows = collections.Counter(), []
    for n, base in enumerate(bases):
        try:
            c, r, _ = fix_interval(base, write=not a.dry_run)
        except Exception as error:
            print('failed', base, repr(error)[:200], flush=True)
            total['interval_failed'] += 1
            continue
        total.update(c)
        rows += r
        if n % 100 == 0:
            print(n, len(bases), dict(total), flush=True)
    reasons = collections.Counter(r.get('reason') for r in rows if r['action'] == 'cleaned')
    summary = {'dry_run': a.dry_run, 'intervals': len(bases), 'counts': dict(total), 'cleaned_reasons': dict(reasons),
               'thresholds': {'partner_max': PARTNER_MAX, 'lcs_min': LCS_MIN, 'speckle_max': SPECKLE_MAX, 'area_ratio': AREA_RATIO, 'lost_max': LOST_MAX},
               'new_lcs_median': {act: float(np.median([r['new_lcs'] for r in rows if r['action'] == act] or [0])) for act in ('dense', 'cleaned')},
               'old_lcs_median': {act: float(np.median([r['old_lcs'] or 0 for r in rows if r['action'] == act] or [0])) for act in ('dense', 'cleaned')}}
    a.out.mkdir(parents=True, exist_ok=True)
    write_json(a.out/('summary.dry-run.json' if a.dry_run else 'summary.json'), summary)
    write_json(a.out/('records.dry-run.json' if a.dry_run else 'records.json'), rows)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
