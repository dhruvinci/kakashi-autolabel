"""VLM saturation bench: how good can Astra / Claude get on hard grappling positions, and what input makes the difference?

Cases are gold-v1 items whose ground truth comes from Dhruva's review notes (reviews/gold-v1/interpreted.json).
Each input variant renders the same cases differently; each model answers every case blind (no notes, no ids).

  ../.venv/bin/python vlm_bench.py render                       # bench/<variant>/cNN*.jpg + prompt.txt
  ../.venv/bin/python vlm_bench.py astra --variant frame --effort high
  ../.venv/bin/python vlm_bench.py score                        # table over bench/answers/*.json
Output: reviews/vlm-bench/
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from autolabel_astra import anchor, call_astra, enum, font, obj, outline, read_usage
from autolabel_common import ROOT, rle_decode
from autolabel_fight import L1A, L1B
from gold_build import BLUE, RED, primary

OUT = ROOT/'reviews/vlm-bench'
GOLD = ROOT/'reviews/gold-v1'
HL = (255, 215, 0)

# Bench cases live next to the gold pack, not in the code (they are labels):
# reviews/vlm-bench/cases.json = [[gold_item_id, position, detail or null, top "RED"/"BLUE" or null, hard], ...]
# detail / top null = not scored; hard = occlusion or fine detail. Ground truth comes from Dhruva's review notes.
CASES_FILE = OUT/'cases.json'

DETAILS = sorted({d for v in L1B.values() for d in v}) + ['none', 'unknown']

PROMPT = """You are an expert BJJ / MMA analyst labelling the grappling position of ONE moment in combat-sports video. Answer only from the attached images. Do not run commands or read files.

{inputs}
The two athletes of interest are {marking}:
- RED: {red}
- BLUE: {blue}
Other people are background.

Return (write the observations FIRST, then classify from them):
- observations: orientation of each athlete, contact points, where each athlete's legs and arms are, grips, hooks; say what is hidden.
- position: one of {l1a}. "transition" only when there is no settled position (scramble, roll or inversion in progress); a takedown attempt, single leg, double leg, front headlock or clinch while at least one athlete is standing is "standing" with the matching detail.
- detail: the most specific term for that position from: {l1b}; else "none" or "unknown".
- top: red | blue | none (the athlete on top / pinning; none when standing or neither).
- confidence: high | medium | low; alt_position: the most plausible other position.
- reasoning: 1-3 sentences naming the visual evidence (whose legs are where, hooks, grips, chest-to-back, ankles crossed...).

Definitions to check:
- Guard: bottom athlete's legs control the top athlete. Closed = ankles crossed behind the top athlete's back (still closed while the bottom athlete is just starting to open it). Half = bottom athlete's legs trap ONE of the top athlete's legs. Open seated / supine = legs used without crossing or trapping.
- Side control: top athlete chest-down across the bottom athlete with BOTH bottom legs free (guard passed). Kesa gatame = top athlete sits hip-to-hip facing the head, arm around the neck; north-south = head-to-head reversed; a leg-drag pin counts as side control once the legs no longer trap.
- Mount: top athlete straddles the bottom athlete's torso, knees on the mat, bottom legs not trapping. If heads are level and the bottom athlete traps a leg, it is half guard, not mount.
- Back: chest to the opponent's back; two_hooks = both heels inside the thighs; body_triangle = legs locked in a figure-four around the waist.
- Leg entanglement: legs intertwined to attack/control a leg (50/50 = mirrored, each athlete's same-side leg trapped).
- Turtle: one athlete on knees and elbows/hands, back exposed."""


def cases():
    items = {i['id']: i for i in json.loads((GOLD/'items.json').read_text())}
    rows = json.loads(CASES_FILE.read_text())
    return [dict(n=n, item=items[i], gt=dict(position=p, detail=d, top=t), hard=h) for n, (i, p, d, t, h) in enumerate(rows)]


def load(src):
    path, k = src.split('#')
    base = ROOT/'autolabel/v1'/path
    L = json.loads((base/'labels.json').read_text())
    return L, int(k), primary(L)


def union_box(frames, pair, pad=0.3, min_frac=0.35):
    H, W = frames[0]['height'], frames[0]['width']
    bs = [i['box_xyxy'] for f in frames for i in f['instances'] if i['actor_id'] in pair and i['area']]
    if not bs:
        return 0, 0, W, H
    x1, y1, x2, y2 = min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)
    cx, cy = (x1 + x2)/2, (y1 + y2)/2
    w = min(W, max((x2 - x1)*(1 + 2*pad), (y2 - y1)*(1 + 2*pad)*W/H, min_frac*W))
    h = w*H/W
    x0, y0 = min(max(0, cx - w/2), W - w), min(max(0, cy - h/2), H - h)
    return round(x0), round(y0), round(x0 + w), round(y0 + h)


def frame_img(f, pair, box, width, outlines=True, caption=None, border=None, tags=False):
    with Image.open(f['path']) as im:
        px = np.asarray(im.convert('RGB')).copy()
    marks = []
    for actor, col, letter in zip(pair, (RED, BLUE), 'RB'):
        inst = next((i for i in f['instances'] if i['actor_id'] == actor and i['area']), None)
        if inst and outlines:
            px[outline(rle_decode(inst['mask_rle']), 3)] = col
        if inst and tags:
            marks.append((letter, col, anchor(rle_decode(inst['mask_rle']))))
    img = Image.fromarray(px).crop(box)
    sc = width/img.width
    img = img.resize((width, round(img.height*sc)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    for letter, col, (cx, cy) in marks:  # small tag at the mask anchor: identifies the athlete without painting over the body
        x, y = (cx - box[0])*sc, (cy - box[1])*sc
        d.ellipse([x - 17, y - 17, x + 17, y + 17], fill=col, outline='white', width=3)
        d.text((x - 8, y - 13), letter, fill='white', font=font(22))
    if caption:
        d.rectangle([0, 0, d.textlength(caption, font=font(22)) + 12, 32], fill='black')
        d.text((6, 4), caption, fill='white', font=font(22))
    if border:
        d.rectangle([0, 0, img.width - 1, img.height - 1], outline=border, width=8)
    return img


def sheet(imgs, cols):
    w, h = imgs[0].width, max(i.height for i in imgs)
    out = Image.new('RGB', (w*cols, h*((len(imgs) + cols - 1)//cols)), 'gray')
    for n, im in enumerate(imgs):
        out.paste(im, ((n % cols)*w, (n//cols)*h))
    return out


def render_case(variant, c, d):
    """Writes the case images; returns (paths, input description for the prompt)."""
    L, k, pair = load(c['item']['src'])
    F = L['frames']
    f = F[k]
    stem = d/f"c{c['n']:02d}"
    if variant == 'frame':  # baseline: the target frame outlined + the same frame unannotated
        box = union_box([f], pair)
        a, b = stem.with_name(stem.name + '_a.jpg'), stem.with_name(stem.name + '_b.jpg')
        frame_img(f, pair, box, 1600).save(a, quality=90)
        frame_img(f, pair, box, 1600, outlines=False).save(b, quality=90)
        return [a, b], 'Image 1: the moment, athletes outlined. Image 2: the same moment without outlines (to see hidden detail).'
    if variant in ('strip', 'strip_long', 'strip_tags'):  # temporal context from the 2 fps label frames, common crop
        before, after, step = (16, 4, 2) if variant == 'strip_long' else (6, 2, 1)
        ol, tg = variant != 'strip_tags', variant == 'strip_tags'
        ks = [j for j in range(k - before, k + after + 1, step) if 0 <= j < len(F)]
        box = union_box([F[j] for j in ks], pair)
        tiles = [frame_img(F[j], pair, box, 800, caption=f"{F[j]['time_s'] - f['time_s']:+.1f}s",
                           border=HL if j == k else None, outlines=ol, tags=tg) for j in ks]
        a, b = stem.with_name(stem.name + '_a.jpg'), stem.with_name(stem.name + '_b.jpg')
        sheet(tiles, 3).save(a, quality=88)
        frame_img(f, pair, box, 1600, outlines=False, caption='TARGET (no outlines)').save(b, quality=90)
        span = F[ks[-1]]['time_s'] - F[ks[0]]['time_s']
        return [a, b], (f'Image 1: {len(ks)} frames spanning {span:.1f} s (left-to-right, top-to-bottom; time relative to the target '
                        f'in each tile). The TARGET moment is the tile with the yellow border (+0.0s); label THAT moment, using the '
                        f'other frames to understand how the athletes got there and what is hidden. Image 2: the target frame '
                        f'enlarged without outlines.')
    raise ValueError(variant)


def schema():
    return obj({'observations': {'type': 'string'}, 'position': enum(L1A), 'detail': enum(DETAILS), 'top': enum(['red', 'blue', 'none']),
                'confidence': enum(['high', 'medium', 'low']), 'alt_position': enum(L1A), 'reasoning': {'type': 'string'}})


def prompt_for(inputs, variant, c):
    L, _, pair = load(c['item']['src'])
    desc = {a['actor_id']: a['description'] for a in L['actors']}
    marking = 'marked with small R / B tags (bodies are not painted)' if variant == 'strip_tags' else 'outlined in red and blue'
    return PROMPT.format(inputs=inputs, marking=marking, red=desc[pair[0]], blue=desc[pair[1]], l1a=', '.join(L1A), l1b='; '.join(f"{k}: {', '.join(v)}" for k, v in L1B.items()))


def render(variants):
    for v in variants:
        d = OUT/'inputs'/v
        d.mkdir(parents=True, exist_ok=True)
        manifest = []
        for c in cases():
            paths, inputs = render_case(v, c, d)
            (d/f"c{c['n']:02d}.prompt.txt").write_text(prompt_for(inputs, v, c))
            manifest.append({'case': c['n'], 'images': [p.name for p in paths]})
        (d/'manifest.json').write_text(json.dumps(manifest, indent=1))
        print(v, len(manifest), 'cases')


def astra(variant, effort, workers=3):
    d = OUT/'inputs'/variant
    out = OUT/'answers'/f'astra-{effort}-{variant}.json'
    done = json.loads(out.read_text()) if out.exists() else {}
    man = [m for m in json.loads((d/'manifest.json').read_text()) if str(m['case']) not in done]
    used = max((u or {}).get('usedPercent', 0) for u in (read_usage()['primary'], read_usage()['secondary']))
    if used >= 98:
        raise SystemExit(f'usage {used}%')

    def one(m):
        res, rec = call_astra((d/f"c{m['case']:02d}.prompt.txt").read_text(), [d/p for p in m['images']], schema(), effort)
        print(f"astra {effort} {variant} c{m['case']:02d}: {res['position']}/{res['detail']}/{res['top']} {rec['elapsed_s']}s", flush=True)
        return m['case'], res
    with ThreadPoolExecutor(workers) as ex:
        for n, res in ex.map(one, man):
            done[str(n)] = res
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(done, indent=1))


def score():
    cs = cases()
    rows = []
    for f in sorted((OUT/'answers').glob('*.json')):
        ans = json.loads(f.read_text())
        s = dict(cls=0, det=0, det_n=0, top=0, top_n=0, full=0, hard_full=0, n=0, hard_n=0)
        for c in cs:
            a = ans.get(str(c['n']))
            if not a:
                continue
            g = c['gt']
            ok_c = a['position'] == g['position']
            ok_d = g['detail'] is None or a['detail'] == g['detail']
            ok_t = g['top'] is None or a['top'].upper() == g['top']
            s['n'] += 1; s['cls'] += ok_c
            if g['detail']:
                s['det_n'] += 1; s['det'] += ok_c and ok_d
            if g['top']:
                s['top_n'] += 1; s['top'] += ok_t
            full = ok_c and ok_d and ok_t
            s['full'] += full
            if c['hard']:
                s['hard_n'] += 1; s['hard_full'] += full
        rows.append((f.stem, s))
    print(f"{'run':34} {'class':>7} {'detail':>7} {'top':>7} {'full':>7} {'hard full':>10}")
    for name, s in rows:
        print(f"{name:34} {s['cls']:>3}/{s['n']:<3} {s['det']:>3}/{s['det_n']:<3} {s['top']:>3}/{s['top_n']:<3} "
              f"{s['full']:>3}/{s['n']:<3} {s['hard_full']:>5}/{s['hard_n']}")


VERIFY = """{base}

A first-pass labeller (same model) answered this case:
{first}

Act as the senior verifier. Re-examine the images and check the known confusion pairs one by one, stating the evidence for each:
1. Half guard vs side control vs mount: is ANY bottom-athlete leg still trapping a top-athlete leg (half guard)? Are both bottom legs free and the top athlete chest-down beside the torso (side control)? Is the top athlete straddling with a knee on EACH side of the torso (mount)? A knee crossing the belly while the other leg is still on the near side is side control / knee-cut, not mount.
2. Closed vs open guard: are the ankles crossed, or only just uncrossing? An opening closed guard is still closed until the legs separate.
3. Standing vs turtle vs transition: is either athlete on their feet driving a takedown (standing + single/double leg / front headlock), or is one athlete stably on knees and hands (turtle)?
4. Back details: hooks inside the thighs vs legs locked in a figure-four (body triangle).
5. Leg entanglement variant: mirrored same-side legs (50/50) vs ashi variants.
Then give the FINAL answer in the same fields (observations = your verification notes) and set changed = true if it differs from the first pass."""


def verify(variant, effort, first, workers=3):
    d = OUT/'inputs'/variant
    src = json.loads((OUT/'answers'/f'{first}.json').read_text())
    out = OUT/'answers'/f'{first}+verify-{effort}.json'
    done = json.loads(out.read_text()) if out.exists() else {}
    man = [m for m in json.loads((d/'manifest.json').read_text()) if str(m['case']) not in done]
    sch = schema()
    sch['properties']['changed'] = {'type': 'boolean'}
    sch['required'].append('changed')

    def one(m):
        a = src[str(m['case'])]
        first_txt = json.dumps({k: a[k] for k in ('observations', 'position', 'detail', 'top', 'confidence', 'reasoning') if k in a}, indent=1)
        prompt = VERIFY.format(base=(d/f"c{m['case']:02d}.prompt.txt").read_text(), first=first_txt)
        res, rec = call_astra(prompt, [d/p for p in m['images']], sch, effort)
        print(f"verify {first} c{m['case']:02d}: {res['position']}/{res['detail']}/{res['top']} changed={res['changed']}", flush=True)
        return m['case'], res
    with ThreadPoolExecutor(workers) as ex:
        for n, res in ex.map(one, man):
            done[str(n)] = res
            out.write_text(json.dumps(done, indent=1))


def cases_table():
    runs = {f.stem: json.loads(f.read_text()) for f in sorted((OUT/'answers').glob('*.json'))}
    for c in cases():
        g = c['gt']
        print(f"c{c['n']:02d} #{c['item']['id']:<3} {'H' if c['hard'] else ' '} GT {g['position']}/{g['detail'] or '*'}/{g['top'] or '*'}")
        for name, ans in runs.items():
            a = ans.get(str(c['n']))
            if a:
                ok = a['position'] == g['position'] and (g['detail'] is None or a['detail'] == g['detail']) and (g['top'] is None or a['top'].upper() == g['top'])
                print(f"      {'+' if ok else '-'} {name:28} {a['position']}/{a['detail']}/{a['top']} ({a['confidence']})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['render', 'astra', 'score', 'cases', 'verify'])
    ap.add_argument('--variant', action='append')
    ap.add_argument('--effort', default='high')
    ap.add_argument('--first')
    a = ap.parse_args()
    if a.cmd == 'render':
        render(a.variant or ['frame', 'strip', 'strip_long', 'strip_tags'])
    elif a.cmd == 'astra':
        for v in a.variant:
            astra(v, a.effort)
    elif a.cmd == 'verify':
        verify(a.variant[0], a.effort, a.first)
    elif a.cmd == 'cases':
        cases_table()
    else:
        score()
