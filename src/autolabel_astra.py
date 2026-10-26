"""Stage 2: one gpt-6-astra set-of-marks call per interval through the logged-in Codex CLI.

Renders SAM detections as numbered marks on keyframes, asks Astra for identities, roles, postures,
mark verdicts, unmarked/merged people and frame/interval labels under a strict JSON schema.
Output: autolabel/<run>/<trial>/<packet>/{keyframes/*.jpg, astra.json, astra-receipt.json}
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from autolabel_common import ROOT, TAXONOMY, interval_dir, rle_decode, sha256, write_json

PROMPT = ROOT/'docs/autolabel/ASTRA_PROMPT.v4.md'
MODEL = 'gpt-6-astra'
EFFORT = 'medium'
N_KEYFRAMES = 6
COLOURS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40), (128, 0, 0)]


def keyframe_indices(n, k=N_KEYFRAMES):
    return sorted({round(i*(n-1)/(k-1)) for i in range(k)})


def font(size):
    for name in ('DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def outline(mask, width=3):
    edge = np.zeros_like(mask)
    inner = mask.copy()
    for _ in range(width):
        shrunk = inner.copy()
        shrunk[1:] &= inner[:-1]; shrunk[:-1] &= inner[1:]
        shrunk[:, 1:] &= inner[:, :-1]; shrunk[:, :-1] &= inner[:, 1:]
        edge |= inner & ~shrunk
        inner = shrunk
    return edge


def anchor(mask, step=4):
    """A point well inside the mask (approximate pole of inaccessibility on a downsampled grid)."""
    small = mask[::step, ::step].copy()
    if not small.any():
        ys, xs = np.nonzero(mask)
        return int(np.median(xs)), int(np.median(ys))
    while True:
        shrunk = small.copy()
        shrunk[1:] &= small[:-1]; shrunk[:-1] &= small[1:]
        shrunk[:, 1:] &= small[:, :-1]; shrunk[:, :-1] &= small[:, 1:]
        if not shrunk.any():
            break
        small = shrunk
    ys, xs = np.nonzero(small)
    i = np.argmin((ys - ys.mean())**2 + (xs - xs.mean())**2)
    return int(xs[i]*step), int(ys[i]*step)


MAX_MARKS = 25


def mark_detections(frame, iou=0.6, containment=0.85, max_marks=MAX_MARKS):
    """Mask-NMS over score-sorted detections, capped at max_marks; returns kept indices (mark k = kept[k-1])."""
    kept, kept_masks = [], []
    for i, det in enumerate(frame['detections']):
        if len(kept) >= max_marks:
            break
        mask = rle_decode(det['mask_rle'])
        area = mask.sum()
        if not area:
            continue
        duplicate = False
        for other in kept_masks:
            inter = (mask & other).sum()
            if inter/(area + other.sum() - inter) > iou or inter/min(area, other.sum()) > containment:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
            kept_masks.append(mask)
    return kept, kept_masks


def render_marks(frame, label, path=None, scale=1):
    """Draw numbered marks (scale>1 for tiles that get downsized); returns (detection index per mark, image)."""
    with Image.open(frame['path']) as im:
        image = im.convert('RGB')
    pixels = np.asarray(image).copy()
    kept, masks = mark_detections(frame)
    for j, mask in enumerate(masks):
        pixels[outline(mask, 3*scale)] = COLOURS[j % len(COLOURS)]
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    tag_font, banner_font = font(18*scale), font(20*scale)
    th = 22*scale
    banner = f"{label}  t={frame['time_s']:.1f}s"
    bw = draw.textlength(banner, font=banner_font) + 16
    draw.rectangle([0, 0, bw, 28*scale], fill='black')
    draw.text((8, 3), banner, fill='white', font=banner_font)
    placed = [(0, 0, bw, 28*scale)]
    for j, mask in enumerate(masks):
        cx, cy = anchor(mask)
        text = str(j + 1)
        w = draw.textlength(text, font=tag_font) + 8
        x, y = max(0, min(cx - w/2, image.width - w)), max(0, min(cy - th/2, image.height - th))
        tries = 0  # bounded: in crowded frames there may be no free spot (this loop used to spin forever)
        while tries < 200 and any(x < b[2] and b[0] < x + w and y < b[3] and b[1] < y + th for b in placed):
            y = y + th + 2 if y + 2*th + 4 < image.height else 0
            x = x if y else min(x + w + 2, image.width - w)
            tries += 1
        placed.append((x, y, x + w, y + th))
        draw.rectangle([x, y, x + w, y + th], fill=COLOURS[j % len(COLOURS)], outline='white', width=scale)
        draw.text((x + 4, y + scale), text, fill='white', font=tag_font)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, quality=92)
    return kept, image


def render_tiles(frames, ids, path):
    """2x2 sheet of half-resolution marked frames; returns detection index per mark for each frame."""
    w, h = frames[0]['width'], frames[0]['height']
    sheet = Image.new('RGB', (w, h), 'gray')
    kept = []
    for slot, (frame, fid) in enumerate(zip(frames, ids)):
        k, image = render_marks(frame, fid, scale=2)
        kept.append(k)
        sheet.paste(image.resize((w//2, h//2), Image.LANCZOS), ((slot % 2)*(w//2), (slot//2)*(h//2)))
    draw = ImageDraw.Draw(sheet)
    draw.line([(w//2, 0), (w//2, h)], fill='white', width=3)
    draw.line([(0, h//2), (w, h//2)], fill='white', width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)
    return kept


def enum(values):
    return {'type': 'string', 'enum': values}


def obj(props):
    return {'type': 'object', 'additionalProperties': False, 'properties': props, 'required': list(props)}


def schema(tax):
    roles, postures = tax['person']['role']['values'], tax['person']['posture']['values']
    marks = {'type': 'array', 'items': obj({
        'mark': {'type': 'integer'},
        'verdict': enum(['person', 'merged', 'partial_duplicate', 'not_a_person']),
        'actor_id': {'type': ['string', 'null']},
        'merged_actor_ids': {'type': 'array', 'items': {'type': 'string'}},
        'posture': enum(postures)})}
    frame_labels = {'frame': {'type': 'string'},
                    'shot_type': enum(tax['frame']['shot_type']['values']),
                    'pair_configuration': enum(tax['frame']['pair_configuration']['values']),
                    'marks': marks}
    return obj({
        'clip_context': enum(tax['interval']['clip_context']['values']),
        'actors': {'type': 'array', 'items': obj({'actor_id': {'type': 'string'}, 'role': enum(roles),
                                                  'description': {'type': 'string'}})},
        'keyframes': {'type': 'array', 'items': obj(frame_labels | {
            'unmarked_people': {'type': 'array', 'items': obj({
                'actor_id': {'type': 'string'}, 'posture': enum(postures),
                'box_xyxy': {'type': 'array', 'items': {'type': 'integer'}}, 'positive_points': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'integer'}}}})}})},
        'frames': {'type': 'array', 'items': obj(frame_labels | {
            'unmarked_people': {'type': 'array', 'items': obj({
                'actor_id': {'type': 'string'}, 'posture': enum(postures),
                'box_xyxy_sheet': {'type': 'array', 'items': {'type': 'integer'}}, 'positive_points': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'integer'}}}})}})},
        'notes': {'type': 'string'}})


def prompt_text(tax, source, n_frames, keyframe_ids, width, height):
    pair = tax['frame']['pair_configuration']
    return PROMPT.read_text().format(
        n_frames=n_frames, last=n_frames - 1, n_keyframes=len(keyframe_ids), keyframe_ids=', '.join(keyframe_ids), sport=source.get('sport_hypothesis'), context=source.get('context_hypothesis'),
        title=source.get('title', '').replace('"', "'"), roles=', '.join(tax['person']['role']['values']),
        role_definition=tax['person']['role']['definition'],
        postures=', '.join(tax['person']['posture']['values']),
        shot_types=', '.join(tax['frame']['shot_type']['values']),
        pair_configurations=', '.join(pair['values']), pair_definition=pair['definition'],
        clip_contexts=', '.join(tax['interval']['clip_context']['values']), width=width, height=height)


def cohort_source(trial_id):
    cohort = json.loads((ROOT/'manifests/trial100-v1/cohort.json').read_text())
    sources = {s['trial_id']: s for s in cohort['sources']}
    for path in (ROOT/'manifests/trial100-v1/replacements').glob('*.json'):
        new = json.loads(path.read_text())['new_source']
        sources[new['trial_id']] = new
    for cohort_path in (ROOT/'manifests/autolabel-v2/cohort.json', ROOT/'manifests/autolabel-v3/cohort.json',
                        ROOT/'manifests/autolabel-v4/cohort.json'):
        if cohort_path.exists():
            sources |= {s['trial_id']: s for s in json.loads(cohort_path.read_text())['sources']}
    return sources[trial_id]


def usage_from_events(text):
    usage = None
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'turn.completed':
            usage = event.get('usage')
    return usage


STOP_USED_PERCENT = 99  # user rule (2026-10-05): use down to 1% remaining, then hold all Astra work


def read_usage():
    """Codex plan rate limits via the local app-server (no model call). Never redeems reset credits."""
    msgs = [{'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'clientInfo': {'name': 'autolabel', 'version': '1'}}},
            {'jsonrpc': '2.0', 'method': 'initialized', 'params': {}},
            {'jsonrpc': '2.0', 'id': 2, 'method': 'account/rateLimits/read', 'params': {}}]
    proc = subprocess.Popen(['codex', 'app-server'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    try:
        proc.stdin.write(''.join(json.dumps(m) + '\n' for m in msgs))
        proc.stdin.flush()
        deadline = time.time() + 30
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            msg = json.loads(line)
            if msg.get('id') == 2:
                limits = msg['result']['rateLimits']
                return {'observed_at': datetime.now(timezone.utc).isoformat(), 'plan': limits.get('planType'),
                        'primary': limits.get('primary'), 'secondary': limits.get('secondary'),
                        'limit_reached': limits.get('rateLimitReachedType'), 'reset_credit_used': False}
    finally:
        proc.kill()
    raise RuntimeError('could not read Codex usage; refusing to call Astra blind')


def check_usage():
    usage = read_usage()
    used = max((w or {}).get('usedPercent', 0) for w in (usage['primary'], usage['secondary']))
    if used >= STOP_USED_PERCENT or usage['limit_reached']:
        raise SystemExit(f'STOP: Codex usage at {used}% (limit {STOP_USED_PERCENT}%). Ask Dhruva before more Astra calls.')
    return usage


TRANSIENT = ('at capacity', 'rate limit', 'rate_limit', 'overloaded', 'temporarily unavailable', '503', '429')


def call_astra(prompt, images, schema_value, effort=EFFORT, timeout=1800, retries=3):
    """call_astra_once with backoff on transient server-side errors (model at capacity, rate limits)."""
    for attempt in range(retries + 1):
        try:
            return call_astra_once(prompt, images, schema_value, effort, timeout)
        except RuntimeError as error:
            if attempt == retries or not any(t in str(error).lower() for t in TRANSIENT):
                raise
            wait = 120*2**attempt
            print(f'transient Astra error, retrying in {wait}s: {str(error)[:160]}', flush=True)
            time.sleep(wait)


def call_astra_once(prompt, images, schema_value, effort=EFFORT, timeout=1800):
    with tempfile.TemporaryDirectory(prefix='astra-') as tmp:
        tmp = Path(tmp)
        (tmp/'schema.json').write_text(json.dumps(schema_value))
        cmd = ['codex', 'exec', '--skip-git-repo-check', '--ephemeral', '--ignore-user-config', '--ignore-rules',
               '-s', 'read-only', '-C', str(tmp), '-m', MODEL, '-c', f'model_reasoning_effort={effort}',
               '--json', '--output-schema', str(tmp/'schema.json'), '-o', str(tmp/'out.json'),
               '-i', *map(str, images), '--', prompt]
        started = time.time()
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - started
        out = (tmp/'out.json').read_text() if (tmp/'out.json').exists() else ''
        if proc.returncode or not out.strip():
            errors = [line for line in proc.stdout.splitlines() if '"error"' in line or 'turn.failed' in line]
            raise RuntimeError(f'codex exec failed rc={proc.returncode}: events={errors[-3:]} stderr_head={proc.stderr[:800]}')
        return json.loads(out), {'elapsed_s': round(elapsed, 1), 'usage': usage_from_events(proc.stdout),
                                 'cmd': cmd[:-1] + ['<prompt>']}


def run(run_name, targets, effort=EFFORT, force=False):
    tax = json.loads(TAXONOMY.read_text())
    schema_value = schema(tax)
    for trial_id, packet_id in targets:
        base = interval_dir(run_name, trial_id, packet_id)
        out_path = base/'astra.json'
        if out_path.exists() and not force:
            print('skip', out_path)
            continue
        det = json.loads((base/'detections.json').read_text())
        frames = det['frames']
        keys = keyframe_indices(len(frames))
        others = [i for i in range(len(frames)) if i not in keys]
        fid = lambda i: f'F{i:02d}'
        images, marks = [], {}
        for i in keys:
            path = base/'astra-images'/f'key-{fid(i)}.jpg'
            marks[i], _ = render_marks(frames[i], f'{fid(i)} KEYFRAME t={frames[i]["time_s"]:.1f}s', path)
            images.append(path)
        sheets = []
        for start in range(0, len(others), 4):
            group = others[start:start + 4]
            path = base/'astra-images'/f'tiles-{fid(group[0])}-{fid(group[-1])}.jpg'
            sheets.append({'path': str(path), 'frames': [fid(i) for i in group], 'scale': 0.5,
                           'tile_origins_xy': [[(k % 2)*(frames[0]['width']//2), (k//2)*(frames[0]['height']//2)] for k in range(len(group))]})
            for i, k in zip(group, render_tiles([frames[i] for i in group], [fid(i) for i in group], path)):
                marks[i] = k
            images.append(path)
        usage_before = check_usage()
        prompt = prompt_text(tax, cohort_source(trial_id), len(frames), [fid(i) for i in keys],
                             frames[0]['width'], frames[0]['height'])
        try:
            result, receipt = call_astra(prompt, images, schema_value, effort)
        except subprocess.TimeoutExpired:
            print(f'{trial_id}/{packet_id}: timeout, retrying once', flush=True)
            check_usage()
            result, receipt = call_astra(prompt, images, schema_value, effort)
        write_json(out_path, {
            'version': 2, 'stage': 'astra_set_of_marks_all_frames', 'trial_id': trial_id, 'packet_id': packet_id,
            'keyframe_indices': keys,
            'tile_sheets': sheets,
            'frames': [{'frame': fid(i), 'frame_index': i, 'frame_id': frames[i]['frame_id'], 'time_s': frames[i]['time_s'],
                        'is_keyframe': i in keys, 'mark_to_detection_index': marks[i]} for i in range(len(frames))],
            'result': result})
        write_json(base/'astra-receipt.json', {
            'model': MODEL, 'reasoning_effort': effort, 'auth': 'codex chatgpt login (no API key)',
            'prompt_template': str(PROMPT), 'prompt_template_sha256': sha256(PROMPT),
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'taxonomy': tax['version'], 'taxonomy_sha256': sha256(TAXONOMY),
            'detections_sha256': sha256(base/'detections.json'),
            'images': [{'path': str(p), 'sha256': sha256(p)} for p in images],
            'created_at': datetime.now(timezone.utc).isoformat(), 'usage_before': usage_before, **receipt})
        print(f"{trial_id}/{packet_id}: {receipt['elapsed_s']}s usage={receipt['usage']}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v1')
    ap.add_argument('--effort', default=EFFORT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('targets', nargs='+', help='trialNNN/intervalNN')
    a = ap.parse_args()
    run(a.run, [tuple(t.split('/')) for t in a.targets], a.effort, a.force)
