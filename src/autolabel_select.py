"""Scale-up cohort (autolabel-v2): choose sources, build keyframe overview sheets, let gpt-6-astra pick a 20 s
high-action interval per source, then build evidence packets with the existing trial_evidence.py (no ASR).

  sources   pick N untouched downloaded sources, balanced across sport/context cells -> manifests/autolabel-v2/cohort.json
  overview  48 keyframe thumbnails per source on a 6x8 timestamped sheet (CPU, parallel)
  pick      one Astra call per source (usage-guarded): scope check + two ranked 20 s windows
  evidence  trial_evidence.py prepare --no-asr for the first pick -> evidence/autolabel-v2/<id>/selected-v1
"""
import argparse
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

from autolabel_common import ROOT, write_json

# v2: MMA/no-gi breadth cohort (a2sNNN). v4: gi cohort (a4gNNN), added 2026-10-05 after Dhruva approved gi.
COHORTS = {'v2': {'prefix': 'a2s', 'sports': ('mma', 'nogi'), 'scopes': ('mma', 'nogi_grappling')},
           'v4': {'prefix': 'a4g', 'sports': ('gi',), 'scopes': ('gi_grappling',)}}
CFG = COHORTS['v2']
COHORT = ROOT/'manifests/autolabel-v2/cohort.json'
EVIDENCE_V2 = ROOT/'evidence/autolabel-v2'
WORK = ROOT/'autolabel/v2-select'


def use_cohort(name):
    global CFG, COHORT, EVIDENCE_V2, WORK
    CFG = COHORTS[name]
    COHORT = ROOT/f'manifests/autolabel-{name}/cohort.json'
    EVIDENCE_V2 = ROOT/f'evidence/autolabel-{name}'
    WORK = ROOT/f'autolabel/{name}-select'
PROMPT = ROOT/'docs/autolabel/SELECT_PROMPT.v1.md'
CELLS = [('mma', 'competition'), ('mma', 'training'), ('mma', 'tutorial'),
         ('nogi', 'competition'), ('nogi', 'training'), ('nogi', 'tutorial'),
         ('gi', 'competition'), ('gi', 'training'), ('gi', 'tutorial')]
WINDOW = 20.0


def used_source_keys():
    used = set()
    cohort = json.loads((ROOT/'manifests/trial100-v1/cohort.json').read_text())
    used |= {s['source_key'] for s in cohort['sources']}
    for p in (ROOT/'manifests/trial100-v1/replacements').glob('*.json'):
        r = json.loads(p.read_text())
        used |= {r['replaces_source_key'], r['new_source']['source_key']}
    return used


def choose_sources(n, seed=20261004):
    rows = [json.loads(l) for l in (ROOT/'manifests/sources.jsonl').open()]
    used = used_source_keys()
    cells = [c for c in CELLS if c[0] in CFG['sports']]
    pool = {c: sorted([r for r in rows if r['status'] == 'complete' and r['source_key'] not in used
                       and (r['sport'], r['context']) == c], key=lambda r: r['source_key']) for c in cells}
    rng = random.Random(seed)
    for c in cells:
        rng.shuffle(pool[c])
    chosen, k = [], 0
    while len(chosen) < n and any(pool.values()):  # round robin keeps cells balanced; small cells exhaust first
        c = cells[k % len(cells)]
        if pool[c]:
            chosen.append(pool[c].pop())
        k += 1
    sources = [{'trial_id': f"{CFG['prefix']}{i + 1:03d}", 'source_key': r['source_key'], 'url': r['url'], 'title': r['title'],
                'sport_hypothesis': r['sport'], 'context_hypothesis': r['context'], 'duration_s': r['duration'],
                'media_path': r['media_path'], 'source_sha256': r['sha256']} for i, r in enumerate(chosen)]
    write_json(COHORT, {'version': 1, 'created_for': 'autolabel scale-up, one 20 s interval per untouched source',
                        'seed': seed, 'sources': sources})
    print(json.dumps({c[0] + '/' + c[1]: sum((s['sport_hypothesis'], s['context_hypothesis']) == c for s in sources) for c in cells}))


def overview(source, n=48):
    out = WORK/source['trial_id']/'overview.jpg'
    if out.exists():
        return
    thumbs = WORK/source['trial_id']/'thumbs'
    thumbs.mkdir(parents=True, exist_ok=True)
    media = ROOT/source['media_path'] if not str(source['media_path']).startswith('/') else Path(source['media_path'])
    proc = subprocess.run(['ffmpeg', '-hide_banner', '-nostdin', '-threads', '2', '-skip_frame', 'nokey', '-copyts',
                           '-i', str(media), '-map', '0:v:0', '-vf', 'scale=320:-2,showinfo', '-fps_mode', 'vfr',
                           '-q:v', '4', str(thumbs/'%05d.jpg')], capture_output=True, text=True, timeout=1800)
    times = [float(t) for t in re.findall(r'pts_time:([0-9.]+)', proc.stderr)]
    files = sorted(thumbs.glob('*.jpg'))
    if not files or abs(len(files) - len(times)) > 1:
        raise RuntimeError(f"{source['trial_id']}: keyframe decode mismatch {len(files)} vs {len(times)}")
    n_ok = min(len(files), len(times))  # ffmpeg occasionally logs one extra showinfo line (or drops a final frame)
    files, times = files[:n_ok], times[:n_ok]
    duration = float(source['duration_s'])
    targets = [duration*(0.03 + 0.94*i/(n - 1)) for i in range(n)]
    picks = sorted({min(range(len(times)), key=lambda j: abs(times[j] - t)) for t in targets})
    sheet = Image.new('RGB', (8*320, 6*190), 'black')
    draw = ImageDraw.Draw(sheet)
    for slot, j in enumerate(picks[:48]):
        with Image.open(files[j]) as im:
            tile = im.convert('RGB').resize((320, 180))
        x, y = (slot % 8)*320, (slot//8)*190
        sheet.paste(tile, (x, y))
        draw.rectangle([x, y, x + 74, y + 16], fill='black')
        draw.text((x + 3, y + 2), f'{times[j]:.0f}s', fill='white')
    sheet.save(out, quality=85)
    write_json(WORK/source['trial_id']/'overview.json', {'times_s': [times[j] for j in picks[:48]], 'keyframes_total': len(times)})
    for f in files:
        f.unlink()
    thumbs.rmdir()


def pick_schema():
    from autolabel_astra import enum, obj
    return obj({
        'scope': enum(['mma', 'nogi_grappling', 'gi_grappling', 'other_sport', 'not_fight_footage']),
        'picks': {'type': 'array', 'items': obj({
            'start_s': {'type': 'number'},
            'content': enum(['standing_striking', 'standing_grappling', 'ground_grappling', 'scramble_transition',
                             'instruction_demo', 'mixed']),
            'reason': {'type': 'string'}})},
        'notes': {'type': 'string'}})


def pick(source, effort='low'):
    from autolabel_astra import call_astra, check_usage
    out = WORK/source['trial_id']/'pick.json'
    if out.exists():
        return
    ov = json.loads((WORK/source['trial_id']/'overview.json').read_text())
    prompt = PROMPT.read_text().format(title=source['title'].replace('"', "'"), sport=source['sport_hypothesis'],
                                       context=source['context_hypothesis'], duration=round(float(source['duration_s'])),
                                       window=int(WINDOW))
    usage = check_usage()
    result, receipt = call_astra(prompt, [WORK/source['trial_id']/'overview.jpg'], pick_schema(), effort)
    write_json(out, {'source': source, 'overview_times_s': ov['times_s'], 'result': result,
                     'receipt': receipt | {'usage_before': usage, 'reasoning_effort': effort}})
    print(f"{source['trial_id']}: {result['scope']} {[round(p['start_s']) for p in result['picks']]} {receipt['elapsed_s']}s", flush=True)


def evidence(source):
    pick_path = WORK/source['trial_id']/'pick.json'
    out = EVIDENCE_V2/source['trial_id']/'selected-v1'
    if out.exists() or not pick_path.exists():
        return
    p = json.loads(pick_path.read_text())['result']
    if p['scope'] not in CFG['scopes'] or not p['picks']:
        return
    duration = float(source['duration_s'])
    clamp = lambda t: max(0.0, min(float(t), duration - WINDOW - 1))
    starts = [clamp(q['start_s']) for q in p['picks'][:2]]
    if len(starts) < 2 or abs(starts[1] - starts[0]) < WINDOW + 5:
        # trial_evidence needs two disjoint windows; place the backup far from the first pick
        first = starts[0]
        starts = [first, clamp(first + duration/2 if first + duration/2 < duration - WINDOW - 1 else first - duration/2)]
    starts = sorted(starts)
    if starts[1] - starts[0] < WINDOW + 5:
        return
    ranges = WORK/source['trial_id']/'ranges.json'
    ranges.write_text(json.dumps([[round(t, 2), round(t + WINDOW, 2)] for t in starts]))
    start = starts[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT/'src/trial_evidence.py'), 'prepare', '--source-key', source['source_key'],
                    '--output', str(out), '--ranges-json', str(ranges), '--no-asr'],
                   cwd=ROOT/'src', check=True, capture_output=True, text=True, timeout=3600)
    print(f"{source['trial_id']}: evidence {[round(t) for t in starts]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['sources', 'overview', 'pick', 'evidence'])
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--cohort', choices=sorted(COHORTS), default='v2')
    a = ap.parse_args()
    use_cohort(a.cohort)
    if a.stage == 'sources':
        return choose_sources(a.n)
    sources = json.loads(COHORT.read_text())['sources']
    if a.limit:
        sources = sources[:a.limit]
    fn = {'overview': overview, 'pick': pick, 'evidence': evidence}[a.stage]
    workers = 3 if a.stage == 'pick' else a.workers
    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(fn, s) for s in sources]:
            try:
                future.result()
            except Exception as error:  # keep going; failures are reported and simply retried on rerun
                print(f'FAILED: {str(error)[:300]}', flush=True)
                if 'STOP' in str(error):
                    break


if __name__ == '__main__':
    main()
