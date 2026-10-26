"""VLM saturation bench for EVENTS (sibling of vlm_bench.py): can Astra / Claude get event type, actor and outcome right,
and does denser / more focused temporal input help?

Ground truth = gold-v1 event items, with Dhruva's notes (interpreted.json) overriding the original Astra label.
  ../.venv/bin/python vlm_bench_events.py render
  ../.venv/bin/python vlm_bench_events.py astra --variant interval2 --effort high
  ../.venv/bin/python vlm_bench_events.py score
Output: reviews/vlm-bench-events/
"""
import argparse
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from autolabel_astra import anchor, call_astra, enum, font, obj, outline, read_usage
from autolabel_common import ROOT, rle_decode
from autolabel_densify import decode_span
from autolabel_fight import EVENTS
from gold_build import BLUE, RED, master_of, primary
from vlm_bench import union_box

OUT = ROOT/'reviews/vlm-bench-events'
GOLD = ROOT/'reviews/gold-v1'
SUBS = ['rear_naked_choke', 'guillotine', 'darce', 'anaconda', 'arm_triangle', 'triangle', 'armbar', 'kimura', 'americana',
        'omoplata', 'heel_hook', 'kneebar', 'straight_ankle_lock', 'toe_hold', 'calf_slicer', 'other']

# Ground truth lives next to the gold pack, not in the code (it is labels):
# reviews/vlm-bench-events/cases.json = {"<gold item id>": {"accept": [[type, actor or null, [outcomes...]], ...],
#                                                          "hard": bool}, ...}
# actor null = not scored; hard = the original label was wrong per Dhruva's review notes.
CASES_FILE = OUT/'cases.json'


def load_gt():
    raw = json.loads(CASES_FILE.read_text())
    return {int(k): ([(t, a, set(o)) for t, a, o in v['accept']], v['hard']) for k, v in raw.items()}


PROMPT = """You are an expert BJJ / MMA analyst labelling EVENTS in a stretch of combat-sports video. Answer only from the attached images. Do not run commands or read files.

{inputs}
The two athletes are tagged with small R / B markers (bodies are not painted):
- RED: {red}
- BLUE: {blue}
Other people (referee, coaches, other pairs) are untagged.

First write observations: frame by frame in your own words, what each athlete does (positions, grips, legs, who is on top, any tapping, referee action, clock/scoreboard, athletes separating or celebrating).
Then list every EVENT you can see, each with: type, actor (red | blue | referee | none: who performs it), start_tile and end_tile (tile numbers as printed), outcome, detail.
Event types: {events}.
- takedown, sweep, pass, back_take, mount_achieve, reguard, escape, leg_entry, stand_up, reset: transitions; outcome success only when the end state is visibly reached and held (a pass = guard cleared and side control / mount / knee-on-belly established; back_take = chest-to-back WITH hooks or body triangle; a seatbelt ride on a turtle is an attempt).
- submission_attempt: only when a submission mechanic is actively applied (detail = the submission: {subs}). outcome success = the defender taps / the referee stops it / the round ends on it; failed = defended or escaped; ongoing = still on at the end.
- tap: the defender taps or the match is stopped by submission. A finished submission is often followed by the athletes releasing and lying apart - that is NOT a reset.
- escape: getting out of a pin or a submission (detail says what from; success if out of the submission even if still pinned).
- referee_stoppage: referee intervenes (detail: why if visible). If the bout/round has ended (clock 0:00, referee waves it off, athletes shake hands), say so in detail of the last event or notes.
Do not invent events you cannot see; prefer outcome unknown over guessing."""


def schema():
    return obj({'observations': {'type': 'string'}, 'events': {'type': 'array', 'items': obj({
        'type': enum(EVENTS), 'actor': enum(['red', 'blue', 'referee', 'none']), 'start_tile': {'type': 'integer'},
        'end_tile': {'type': 'integer'}, 'outcome': enum(['success', 'failed', 'ongoing', 'unknown']), 'detail': {'type': 'string'}})},
        'notes': {'type': 'string'}})


def cases():
    items = {i['id']: i for i in json.loads((GOLD/'items.json').read_text())}
    out = []
    for n, (iid, (acc, hard)) in enumerate(load_gt().items()):
        it = items[iid]
        base = ROOT/'autolabel/v1'/it['src']
        fd = json.loads((base/'fight.json').read_text())
        times = {f['frame']: f['time_s'] for f in fd['frames']}
        ev = next(e for e in fd['result']['events'] if e['type'] == it['pred']['event'] and e['outcome'] == it['pred']['outcome']
                  and e['detail'][:160] == it['pred']['detail'][:160])
        out.append(dict(n=n, item=it, base=base, t0=times[ev['start_frame']], t1=max(times[ev['end_frame']], times[ev['start_frame']] + 1),
                        accept=acc, hard=hard))
    return out


def tag_tile(img_path, masks, pair, box, width, caption, hl=False):
    """masks: {actor: bool array} at this moment."""
    with Image.open(img_path) as im:
        img = im.convert('RGB').crop(box)
    sc = width/img.width
    img = img.resize((width, round(img.height*sc)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    for actor, col, letter in zip(pair, (RED, BLUE), 'RB'):
        m = masks.get(actor)
        if m is not None and m.any():
            cx, cy = anchor(m)
            x, y = (cx - box[0])*sc, (cy - box[1])*sc
            d.ellipse([x - 15, y - 15, x + 15, y + 15], fill=col, outline='white', width=3)
            d.text((x - 7, y - 12), letter, fill='white', font=font(20))
    d.rectangle([0, 0, d.textlength(caption, font=font(20)) + 10, 28], fill='black')
    d.text((5, 3), caption, fill='white', font=font(20))
    if hl:
        d.rectangle([0, 0, img.width - 1, img.height - 1], outline=(255, 215, 0), width=6)
    return img


def sheets(tiles, stem, per=12, cols=4):
    paths = []
    for s in range(0, len(tiles), per):
        group = tiles[s:s + per]
        w, h = group[0].width, max(t.height for t in group)
        out = Image.new('RGB', (w*cols, h*((len(group) + cols - 1)//cols)), 'gray')
        for i, t in enumerate(group):
            out.paste(t, ((i % cols)*w, (i//cols)*h))
        p = stem.with_name(f'{stem.name}_s{s//per}.jpg')
        out.save(p, quality=85)
        paths.append(p)
    return paths


def render_case(variant, c, d):
    L = json.loads((c['base']/'labels.json').read_text())
    pair = primary(L)
    stem = d/f"c{c['n']:02d}"
    if variant == 'interval2':  # production-like context: the whole interval at 2 fps
        F = L['frames']
        box = union_box(F, pair, pad=0.25, min_frac=0.4)
        tiles = []
        for n, f in enumerate(F):
            ms = {i['actor_id']: rle_decode(i['mask_rle']) for i in f['instances'] if i['actor_id'] in pair and i['area']}
            tiles.append(tag_tile(f['path'], ms, pair, box, 560, f"{n + 1}  t={f['time_s']:.1f}s"))
        paths = sheets(tiles, stem)
        return paths, (f'{len(paths)} sheets of numbered tiles (left-to-right, top-to-bottom), {len(F)} frames 0.5 s apart covering '
                       f'{F[-1]["time_s"] - F[0]["time_s"]:.0f} s.')
    if variant == 'window4':  # focused: event window -4 s .. +4 s at 4 fps from the native video
        F = L['frames']
        lt = np.array([f['time_s'] for f in F])
        t0, t1 = c['t0'] - 4, c['t1'] + 4
        has_dense = (c['base']/'dense/dense.json').exists()
        if has_dense:
            dd = json.loads((c['base']/'dense/dense.json').read_text())
            z = np.load(c['base']/'dense/masks.npz')
            h, w = dd['shape']
            dt = np.array([f['time_s'] for f in dd['frames']])
        tmp = Path(tempfile.mkdtemp(prefix='evbench-'))
        try:
            native = decode_span(master_of(c['base']), max(0, t0 - 0.2), t1 + 0.2, tmp)
            picks, last = [], -9
            for path, t in native:
                if t0 <= t <= t1 and t - last >= 0.249:
                    picks.append((path, t))
                    last = t
            masks = []
            for path, t in picks:  # tag positions: dense mask nearest in time, else the nearest label frame
                ms = {}
                if has_dense and np.min(np.abs(dt - t)) < 0.06:
                    j = int(np.argmin(np.abs(dt - t)))
                    ms = {a: np.unpackbits(z[f'{j}_{a}'])[:h*w].reshape(h, w).astype(bool) for a in pair if f'{j}_{a}' in z.files}
                elif np.min(np.abs(lt - t)) < 0.3:
                    f = F[int(np.argmin(np.abs(lt - t)))]
                    ms = {i['actor_id']: rle_decode(i['mask_rle']) for i in f['instances'] if i['actor_id'] in pair and i['area']}
                masks.append(ms)
            ks = [k for k in range(len(F)) if t0 <= F[k]['time_s'] <= t1] or list(range(len(F)))
            box = union_box([F[k] for k in ks], pair, pad=0.25, min_frac=0.4)
            tiles = [tag_tile(p, ms, pair, box, 560, f"{n + 1}  t={t:.2f}s", hl=c['t0'] <= t <= c['t1'])
                     for n, ((p, t), ms) in enumerate(zip(picks, masks))]
            paths = sheets(tiles, stem)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return paths, (f'{len(paths)} sheets of numbered tiles (left-to-right, top-to-bottom), {len(picks)} frames 0.25 s apart covering '
                       f'{picks[-1][1] - picks[0][1]:.0f} s. Tiles with a yellow border are where an automatic detector thought something '
                       f'happens; judge for yourself.')
    raise ValueError(variant)


def render(variants):
    for v in variants:
        d = OUT/'inputs'/v
        d.mkdir(parents=True, exist_ok=True)
        man = []
        for c in cases():
            L = json.loads((c['base']/'labels.json').read_text())
            pair = primary(L)
            desc = {a['actor_id']: a['description'] for a in L['actors']}
            paths, inputs = render_case(v, c, d)
            (d/f"c{c['n']:02d}.prompt.txt").write_text(PROMPT.format(inputs=inputs, red=desc[pair[0]], blue=desc[pair[1]],
                                                                      events=', '.join(EVENTS), subs=', '.join(SUBS)))
            man.append({'case': c['n'], 'images': [p.name for p in paths]})
            print(v, c['n'], len(paths), 'sheets', flush=True)
        (d/'manifest.json').write_text(json.dumps(man, indent=1))


def astra(variant, effort, workers=3):
    d = OUT/'inputs'/variant
    out = OUT/'answers'/f'astra-{effort}-{variant}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else {}
    man = [m for m in json.loads((d/'manifest.json').read_text()) if str(m['case']) not in done]
    used = max((u or {}).get('usedPercent', 0) for u in (read_usage()['primary'], read_usage()['secondary']))
    if used >= 98:
        raise SystemExit(f'usage {used}%')

    def one(m):
        res, rec = call_astra((d/f"c{m['case']:02d}.prompt.txt").read_text(), [d/p for p in m['images']], schema(), effort)
        print(f"astra {effort} {variant} c{m['case']:02d}: {[(e['type'], e['actor'], e['outcome']) for e in res['events']]} {rec['elapsed_s']}s", flush=True)
        return m['case'], res
    with ThreadPoolExecutor(workers) as ex:
        for n, res in ex.map(one, man):
            done[str(n)] = res
            out.write_text(json.dumps(done, indent=1))


def judge(c, ans):
    L = json.loads((c['base']/'labels.json').read_text())
    pair = primary(L)
    col = {pair[0]: 'red', pair[1]: 'blue'}
    best = (0, 0, 0)
    for e in ans.get('events', []):
        for ty, actor, outs in c['accept']:
            if e['type'] != ty:
                continue
            ok_a = actor is None or col.get(actor) == e['actor']
            best = max(best, (1, int(e['outcome'] in outs), int(ok_a)))
    return best  # (type found, outcome right, actor right)


def score():
    cs = cases()
    print(f"{'run':30} {'type':>6} {'+outcome':>9} {'full':>6} {'hard full':>10}")
    for f in sorted((OUT/'answers').glob('*.json')):
        ans = json.loads(f.read_text())
        t = o = full = hf = n = hn = 0
        for c in cs:
            a = ans.get(str(c['n']))
            if a is None:
                continue
            ty, oc, ac = judge(c, a)
            n += 1; t += ty; o += ty and oc; full += ty and oc and ac
            if c['hard']:
                hn += 1; hf += ty and oc and ac
        print(f"{f.stem:30} {t:>3}/{n:<3} {o:>4}/{n:<3} {full:>3}/{n:<3} {hf:>5}/{hn}")


def cases_table():
    runs = {f.stem: json.loads(f.read_text()) for f in sorted((OUT/'answers').glob('*.json'))}
    for c in cases():
        print(f"c{c['n']:02d} #{c['item']['id']} {'H' if c['hard'] else ' '} GT {[(t, a, sorted(o)) for t, a, o in c['accept']]}")
        for name, ans in runs.items():
            a = ans.get(str(c['n']))
            if a:
                print(f"     {judge(c, a)} {name:26} {[(e['type'], e['actor'], e['outcome'], e['detail'][:25]) for e in a['events']]}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['render', 'astra', 'score', 'cases'])
    ap.add_argument('--variant', action='append')
    ap.add_argument('--effort', default='high')
    a = ap.parse_args()
    if a.cmd == 'render':
        render(a.variant or ['interval2', 'window4'])
    elif a.cmd == 'astra':
        for v in a.variant:
            astra(v, a.effort)
    elif a.cmd == 'cases':
        cases_table()
    else:
        score()
