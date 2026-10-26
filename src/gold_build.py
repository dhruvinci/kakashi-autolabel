"""Build the gold-set review pack (human accuracy measurement of autolabel-v1).

Stratified items (fixed seed):
  mask      300  primary-pair masks on one frame; stratified by quality tier (clean / astra_verified / rejected)
  position  300  smoothed fight position + detail + top actor on one frame; stratified by position class
  event     100  short clip around an Astra event; stratified by event type
  identity  100  one actor id shown at 4 moments of its interval
  strike     60  6-frame 12.5 fps strip: 45 Astra-confirmed strikes (by type) + 15 rejected motion candidates
Output: reviews/gold-v1/{items.json, assets/*.jpg|*.mp4}. Served by src/gold_serve.py.
"""
import collections
import json
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from autolabel_astra import font, outline
from autolabel_common import ROOT, packet, rle_decode

RUN = ROOT/'autolabel/v1'
OUT = ROOT/'reviews/gold-v1'
ASSETS = OUT/'assets'
RED, BLUE = (230, 25, 75), (0, 130, 200)
rng = random.Random(20261011)


def intervals():
    out = []
    for lab in sorted(RUN.glob('*/*/labels.json')):
        base = lab.parent
        if (base/'fight.json').exists() and (base/'smooth.json').exists():
            out.append(base)
    return out


def primary(L):
    return sorted(a['actor_id'] for a in L['actors'] if a.get('primary'))


def tier(f, verify):
    if not f['review_flags'] and not f.get('risk_flags'):
        return 'clean'
    v = verify.get(f['frame_index'])
    if v is None:
        return 'flagged_unverified'
    return 'astra_verified' if v['masks_ok'] else 'rejected'


def crop_box(frame, pair, min_frac=0.4, pad=0.35):
    """16:9 crop around the pair's union box (padded), at least min_frac of the frame width, kept inside the frame."""
    H, W = frame['height'], frame['width']
    boxes = [i['box_xyxy'] for i in frame['instances'] if i['actor_id'] in pair and i['area']]
    if not boxes:
        return 0, 0, W, H
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    cx, cy = (x1 + x2)/2, (y1 + y2)/2
    w = max((x2 - x1)*(1 + 2*pad), (y2 - y1)*(1 + 2*pad)*W/H, min_frac*W)
    w = min(w, W)
    h = w*H/W
    x0 = min(max(0, cx - w/2), W - w)
    y0 = min(max(0, cy - h/2), H - h)
    return round(x0), round(y0), round(x0 + w), round(y0 + h)


def draw_pair(frame, pair, path, width=1280, fill=True, caption=None, raw=None, min_frac=0.4):
    with Image.open(frame['path']) as im:
        px = np.asarray(im.convert('RGB')).astype(np.float32)
    box = crop_box(frame, [a for a in pair if a != '__none__'], min_frac)
    if raw:  # unannotated copy so the reviewer can toggle and see what the overlay hides
        r = Image.fromarray(px.astype(np.uint8)).crop(box)
        r.resize((width, round(r.height*width/r.width)), Image.LANCZOS).save(raw, quality=85)
    for actor, col in zip(pair, (RED, BLUE)):
        inst = next((i for i in frame['instances'] if i['actor_id'] == actor), None)
        if inst is None or not inst['area']:
            continue
        m = rle_decode(inst['mask_rle'])
        if fill:
            px[m] = px[m]*0.6 + np.array(col)*0.4
        px[outline(m, 3)] = col
    img = Image.fromarray(px.clip(0, 255).astype(np.uint8)).crop(box)
    img = img.resize((width, round(img.height*width/img.width)), Image.LANCZOS)
    if caption:
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, d.textlength(caption, font=font(20)) + 12, 30], fill='black')
        d.text((6, 4), caption, fill='white', font=font(20))
    img.save(path, quality=88)


def master_of(base):
    trial, pid = base.parts[-2], base.parts[-1]
    pkt, pkt_path = packet(trial, pid)
    return json.loads((pkt_path.parents[1]/'index.json').read_text())['master_path']


def clip(master, t0, t1, path):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-ss', f'{max(0, t0):.2f}', '-t', f'{t1 - t0:.2f}',
                    '-i', master, '-vf', 'scale=640:-2,fps=12', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
                    '-movflags', '+faststart', '-pix_fmt', 'yuv420p', str(path)], check=True, timeout=120)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    bases = intervals()
    items = []
    # ---- mask + position + identity pools
    mask_pool, pos_pool, id_pool = collections.defaultdict(list), collections.defaultdict(list), []
    for base in bases:
        L = json.loads((base/'labels.json').read_text())
        pair = primary(L)
        if len(pair) != 2:
            continue
        verify = {r['frame_index']: r for r in json.loads((base/'verify.json').read_text()).get('frames', [])} \
            if (base/'verify.json').exists() else {}
        F = json.loads((base/'fight.json').read_text())['result']['frames']
        sp = json.loads((base/'smooth.json').read_text()).get('position', {})
        sm = sp.get('smoothed', [])
        for k, f in enumerate(L['frames']):
            present = [a for a in pair if any(i['actor_id'] == a and i['area'] > 1500 for i in f['instances'])]
            if len(present) == 2:
                mask_pool[tier(f, verify)].append((base, k))
            if k < len(F) and k < len(sm):
                pos_pool[sm[k]].append((base, k))
        id_pool.append((base, pair[0]))
    # masks: 300 = clean 120, astra_verified 90, rejected 90
    for t, n in (('clean', 120), ('astra_verified', 90), ('rejected', 90)):
        for base, k in rng.sample(mask_pool[t], min(n, len(mask_pool[t]))):
            L = json.loads((base/'labels.json').read_text())
            name = f"mask_{len(items):04d}.jpg"
            draw_pair(L['frames'][k], primary(L), ASSETS/name, raw=ASSETS/name.replace('.jpg', '_raw.jpg'))
            items.append({'type': 'mask', 'asset': name, 'raw': name.replace('.jpg', '_raw.jpg'), 'stratum': t, 'src': f'{base.parts[-2]}/{base.parts[-1]}#{k}'})
    # positions: 300 balanced over classes (cap per class), excluding not_applicable/unknown
    classes = [c for c in pos_pool if c not in ('not_applicable', 'unknown') and pos_pool[c]]
    per = max(1, 300//len(classes))
    for c in classes:
        for base, k in rng.sample(pos_pool[c], min(per, len(pos_pool[c]))):
            L = json.loads((base/'labels.json').read_text())
            sp = json.loads((base/'smooth.json').read_text())['position']
            pair = primary(L)
            name = f"pos_{len(items):04d}.jpg"
            draw_pair(L['frames'][k], pair, ASSETS/name, fill=False, raw=ASSETS/name.replace('.jpg', '_raw.jpg'))
            top = sp['top'][k]
            top_col = {pair[0]: 'RED', pair[1]: 'BLUE'}.get(top, top or 'none')
            items.append({'type': 'position', 'asset': name, 'raw': name.replace('.jpg', '_raw.jpg'), 'stratum': c, 'pred': {'position': c, 'detail': sp['detail'][k],
                          'top': top_col}, 'src': f'{base.parts[-2]}/{base.parts[-1]}#{k}'})
    # events: 100 stratified by type
    ev_pool = collections.defaultdict(list)
    for base in bases:
        fd = json.loads((base/'fight.json').read_text())
        times = {f['frame']: f['time_s'] for f in fd['frames']}
        L = json.loads((base/'labels.json').read_text())
        descs = {a['actor_id']: a['description'][:70] for a in L['actors']}
        for e in fd['result']['events']:
            if e['start_frame'] in times and e['end_frame'] in times:
                ev_pool[e['type']].append((base, e, times[e['start_frame']], times[e['end_frame']], descs))
    types = sorted(ev_pool)
    per = max(1, 100//len(types))
    for ty in types:
        for base, e, t0, t1, descs in rng.sample(ev_pool[ty], min(per, len(ev_pool[ty]))):
            name = f"event_{len(items):04d}.mp4"
            try:
                clip(master_of(base), t0 - 1.5, max(t1, t0 + 1) + 1.5, ASSETS/name)
            except Exception as error:
                print('clip failed', base, error)
                continue
            who = f"{e['actor']} ({descs.get(e['actor'], '?')})" if e['actor'] else 'n/a'
            items.append({'type': 'event', 'asset': name, 'stratum': ty, 'pred': {'event': ty, 'actor': who,
                          'outcome': e['outcome'], 'detail': e['detail'][:160]}, 'src': f'{base.parts[-2]}/{base.parts[-1]}'})
    # identity: 100 intervals, 4 moments of one actor
    for base, actor in rng.sample(id_pool, min(100, len(id_pool))):
        L = json.loads((base/'labels.json').read_text())
        ks = [k for k, f in enumerate(L['frames']) if any(i['actor_id'] == actor and i['area'] > 1500 for i in f['instances'])]
        if len(ks) < 4:
            continue
        pick = [ks[round(j*(len(ks) - 1)/3)] for j in range(4)]
        tiles = []
        for k in pick:
            tmp = ASSETS/'_tmp.jpg'
            draw_pair(L['frames'][k], [actor, '__none__'], tmp, width=640, min_frac=0.6)
            tiles.append(Image.open(tmp).copy())
        th = max(t.height for t in tiles)
        sheet = Image.new('RGB', (640*2 + 6, th*2 + 6), 'black')
        for j, t in enumerate(tiles):
            sheet.paste(t, (646*(j % 2), (th + 6)*(j//2)))
        name = f"id_{len(items):04d}.jpg"
        sheet.save(ASSETS/name, quality=88)
        items.append({'type': 'identity', 'asset': name, 'stratum': base.parts[-2][:3], 'src': f'{base.parts[-2]}/{base.parts[-1]}:{actor}'})
    # strikes: 45 Astra-confirmed strikes (stratified by type) + 15 rejected candidates (recall check)
    yes, no = collections.defaultdict(list), []
    for sp in sorted(RUN.glob('*/*/strikes.json')):
        sd = json.loads(sp.read_text())
        for c in sd.get('candidates', []):
            a = c.get('answer')
            if not a:
                continue
            if a['strike'] == 'yes' and a['striker'] in ('red', 'blue'):
                yes[a['type']].append((sp.parent, c))
            elif a['strike'] == 'no':
                no.append((sp.parent, c))
    picked = []
    while len(picked) < 45 and any(yes.values()):
        for ty in sorted(yes):
            if yes[ty] and len(picked) < 45:
                picked.append(yes[ty].pop(rng.randrange(len(yes[ty]))))
    picked += rng.sample(no, min(15, len(no)))
    for base, c in picked:
        a = c['answer']
        name = f"strike_{len(items):04d}.jpg"
        img = Image.open(base/c['image'])
        img.save(ASSETS/name, quality=88)
        items.append({'type': 'strike', 'asset': name, 'stratum': a['type'] if a['strike'] == 'yes' else 'rejected_candidate',
                      'pred': {'strike': a['strike'], 'striker': a['striker'].upper(), 'type': a['type'], 'target': a['target'],
                               'result': a['result'], 'peak_tile': a['peak_tile']},
                      'src': f"{base.parts[-2]}/{base.parts[-1]}@{c['time_s']:.2f}"})
    (ASSETS/'_tmp.jpg').unlink(missing_ok=True)
    rng.shuffle(items)
    # keep types interleaved but grouped in blocks of 10 so the reviewer stays in one mode for a while
    blocks = collections.defaultdict(list)
    for it in items:
        blocks[it['type']].append(it)
    ordered = []
    while any(blocks.values()):
        for ty in ('mask', 'position', 'identity', 'event', 'strike'):
            ordered += blocks[ty][:10]
            blocks[ty] = blocks[ty][10:]
    for n, it in enumerate(ordered):
        it['id'] = n
    (OUT/'items.json').write_text(json.dumps(ordered, indent=1))
    print(json.dumps(collections.Counter(i['type'] for i in ordered)))


if __name__ == '__main__':
    main()
