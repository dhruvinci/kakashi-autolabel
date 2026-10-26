"""Continuous full matches/rounds (autolabel-v3): label whole live-action spans as consecutive 20 s chunks.

Lessons from the earlier Kakashi product's chunk planner:
chunks overlap for context but each chunk OWNS only its core span; identities are linked across chunks. Kakashi
linked A1/A2 by text-description keywords, which broke on similar outfits. Here identities are linked by mask IoU
on the shared overlap frames (pixels, not words).

  span     Astra marks the main continuous live-action span on the existing overview sheet (cheap, usage-guarded)
  evidence decode 2 fps frames once over the span; write one packet per 20 s chunk (+2 s overlap) under
           evidence/autolabel-v3/<id>/selected-v1/chunkNN, in the same format as trial evidence packets
  link     after labels exist: map each chunk's actor ids to the match-level ids by overlap-frame mask IoU
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from autolabel_common import ROOT, interval_dir, rle_decode, sha256, write_json

V3 = ROOT/'manifests/autolabel-v3'
COHORT = V3/'cohort.json'
EVIDENCE_V3 = ROOT/'evidence/autolabel-v3'
SELECT = ROOT/'autolabel/v2-select'
CHUNK, OVERLAP, MAX_SPAN = 20.0, 2.0, 360.0

SPAN_PROMPT = """You are choosing a continuous stretch of a combat-sports video to label end to end. Answer only from the attached overview sheet (48 thumbnails in time order, each tagged with its time in seconds). Video: "{title}" ({duration} s); metadata: sport={sport}, context={context}.
Return the start and end time (seconds) of the main CONTINUOUS live action: one full match, round or roll, without intros, interviews, replays, breaks between rounds or long talking. If the live action is longer than {max_span} s, return the most action-dense {max_span} s. If there is no continuous live action of at least 60 s (e.g. a technique tutorial cut into short clips), set usable=false."""


def span_schema():
    from autolabel_astra import enum, obj
    return obj({'usable': {'type': 'boolean'}, 'start_s': {'type': 'number'}, 'end_s': {'type': 'number'},
                'content': enum(['competition_match', 'sparring_round', 'rolling_round', 'drilling', 'other']),
                'reason': {'type': 'string'}})


def candidates(n):
    """v2 sources with short-ish full videos in competition/training, balanced by sport, from usable picks."""
    v2 = json.loads((ROOT/'manifests/autolabel-v2/cohort.json').read_text())['sources']
    out = {}
    for s in v2:
        pick = SELECT/s['trial_id']/'pick.json'
        if not pick.exists() or not 150 <= float(s['duration_s']) <= 900 or s['context_hypothesis'] == 'tutorial':
            continue
        if json.loads(pick.read_text())['result']['scope'] not in ('mma', 'nogi_grappling'):
            continue
        out.setdefault((s['sport_hypothesis'], s['context_hypothesis']), []).append(s)
    chosen, k = [], 0
    cells = sorted(out)
    while len(chosen) < n and any(out.values()):
        c = cells[k % len(cells)]
        if out[c]:
            chosen.append(out[c].pop(len(out[c])//2))
        k += 1
    return chosen


def stage_span(n):
    from autolabel_astra import call_astra, check_usage
    sources = []
    for i, s in enumerate(candidates(n), start=1):
        work = SELECT/s['trial_id']
        out = work/'span.json'
        if not out.exists():
            prompt = SPAN_PROMPT.format(title=s['title'].replace('"', "'"), duration=round(float(s['duration_s'])),
                                        sport=s['sport_hypothesis'], context=s['context_hypothesis'], max_span=int(MAX_SPAN))
            usage = check_usage()
            result, receipt = call_astra(prompt, [work/'overview.jpg'], span_schema(), 'low')
            write_json(out, {'result': result, 'receipt': receipt | {'usage_before': usage}})
        r = json.loads(out.read_text())['result']
        print(s['trial_id'], r['usable'], round(r['start_s']), round(r['end_s']), r['content'], flush=True)
        if r['usable'] and r['end_s'] - r['start_s'] >= 60:
            sources.append(s | {'trial_id': f'a3m{len(sources) + 1:03d}', 'v2_id': s['trial_id'],
                                'span_s': [max(0.0, r['start_s']), min(float(s['duration_s']) - 1, r['start_s'] + MAX_SPAN, r['end_s'])],
                                'span_content': r['content']})
    write_json(COHORT, {'version': 1, 'created_for': 'continuous chunked matches (owned 20 s chunks + 2 s overlap)',
                        'chunk_s': CHUNK, 'overlap_s': OVERLAP, 'sources': sources})
    print(json.dumps({'matches': len(sources), 'minutes': round(sum(s['span_s'][1] - s['span_s'][0] for s in sources)/60, 1)}))


def decode(master, t0, t1, dest, step=0.5):
    """Same decode recipe as trial_evidence.decode_frames: no seeking, decoded PTS, ~step cadence."""
    dest.mkdir(parents=True, exist_ok=True)
    expr = (f"select='gte(t,{t0:.6f})*lt(t,{t1:.6f})*(isnan(prev_selected_t)+gte(t-prev_selected_t,{step:.6f}))',showinfo")
    proc = subprocess.run(['ffmpeg', '-hide_banner', '-nostdin', '-threads', '2', '-copyts', '-i', str(master), '-map', '0:v:0',
                           '-vf', expr, '-fps_mode', 'vfr', '-q:v', '3', str(dest/'%06d.jpg')],
                          capture_output=True, text=True, check=True, timeout=7200)
    times = [float(t) for t in re.findall(r'pts_time:([0-9.]+)', proc.stderr)]
    files = sorted(dest.glob('*.jpg'))
    assert abs(len(files) - len(times)) <= 1, (len(files), len(times))
    n = min(len(files), len(times))
    return list(zip(files[:n], times[:n]))


def stage_evidence():
    from PIL import Image
    for s in json.loads(COHORT.read_text())['sources']:
        root = EVIDENCE_V3/s['trial_id']/'selected-v1'
        if (root/'index.json').exists():
            continue
        master = ROOT/s['media_path'] if not str(s['media_path']).startswith('/') else Path(s['media_path'])
        t0, t1 = s['span_s']
        frames = decode(master, t0, t1, root/'frames')
        with Image.open(frames[0][0]) as im:
            w, h = im.size
        packets, k, start = [], 1, t0
        while start < t1 - 5:
            core = [start, min(start + CHUNK, t1)]
            ctx = [core[0], min(core[1] + OVERLAP, t1)]
            pid = f'chunk{k:02d}'
            sel = [(p, t) for p, t in frames if ctx[0] - 1e-6 <= t < ctx[1]]
            recs = [{'frame_id': f'{pid}_f{j + 1:06d}', 'time_s': round(t, 4), 'width': w, 'height': h, 'path': str(p),
                     'sha256': None} for j, (p, t) in enumerate(sel)]
            pkt = {'packet_id': pid, 'source_key': s['source_key'], 'source_sha256': s['source_sha256'],
                   'time_range_s': [round(core[0], 3), round(core[1], 3)], 'context_range_s': [round(ctx[0], 3), round(ctx[1], 3)],
                   'owned_range_s': [round(core[0], 3), round(core[1], 3)], 'frames': recs,
                   'notes': 'Continuous match chunk: frames cover owned core + overlap context; only the core is owned.'}
            write_json(root/pid/'packet.json', pkt)
            packets.append({'packet_id': pid, 'path': str(root/pid/'packet.json'), 'time_range_s': pkt['time_range_s']})
            k, start = k + 1, core[1]
        write_json(root/'index.json', {'version': 1, 'source_key': s['source_key'], 'source_sha256': s['source_sha256'],
                                       'master_path': str(master), 'span_s': s['span_s'], 'packets': packets})
        print(f"{s['trial_id']}: {len(packets)} chunks, {len(frames)} frames", flush=True)


def stage_link(run='v1'):
    """Map chunk-local actor ids to match ids by mask IoU over overlap frames (consecutive chunks)."""
    for s in json.loads(COHORT.read_text())['sources']:
        index = json.loads((EVIDENCE_V3/s['trial_id']/'selected-v1'/'index.json').read_text())
        mapping, next_id, links = {}, 1, []
        prev = None
        for p in index['packets']:
            base = interval_dir(run, s['trial_id'], p['packet_id'])
            if not (base/'labels.json').exists():
                prev = None
                continue
            L = json.loads((base/'labels.json').read_text())
            local = {a['actor_id'] for a in L['actors'] if a['role'] in ('athlete', 'referee', 'instructor', 'coach_or_corner')}
            assign = {}
            if prev is not None:
                pf = {f['time_s']: f for f in prev['frames']}
                score = {}
                for f in L['frames']:
                    g = pf.get(f['time_s'])
                    if g is None:
                        continue
                    for i in f['instances']:
                        for j in g['instances']:
                            if i['actor_id'] in local and (prev['pid'], j['actor_id']) in mapping:
                                a, b = rle_decode(i['mask_rle']), rle_decode(j['mask_rle'])
                                iou = (a & b).sum()/max(1, (a | b).sum())
                                score[(i['actor_id'], j['actor_id'])] = score.get((i['actor_id'], j['actor_id']), []) + [iou]
                for (a, b), v in sorted(score.items(), key=lambda kv: -np.mean(kv[1])):
                    if np.mean(v) >= 0.5 and a not in assign and mapping[(prev['pid'], b)] not in assign.values():
                        assign[a] = mapping[(prev['pid'], b)]
                        links.append({'chunk': p['packet_id'], 'local': a, 'match_id': assign[a], 'overlap_iou': round(float(np.mean(v)), 3)})
            for a in sorted(local):
                if a not in assign:
                    assign[a] = f'M{next_id}'
                    next_id += 1
                mapping[(p['packet_id'], a)] = assign[a]
            prev = {'pid': p['packet_id'], 'frames': L['frames']}
        write_json(EVIDENCE_V3.parent.parent/'autolabel'/run/s['trial_id']/'match_ids.json',
                   {'mapping': {f'{k[0]}/{k[1]}': v for k, v in mapping.items()}, 'links': links, 'match_actor_count': next_id - 1})
        print(f"{s['trial_id']}: {next_id - 1} match actors, {len(links)} cross-chunk links", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['span', 'evidence', 'link'])
    ap.add_argument('--n', type=int, default=18)
    a = ap.parse_args()
    {'span': lambda: stage_span(a.n), 'evidence': stage_evidence, 'link': stage_link}[a.stage]()
