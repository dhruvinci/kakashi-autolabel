"""Stage 6: fight-state labels per interval: engagement, Kakashi L1A/L1B position (+ second choice and confidence),
top/dominant actor per frame, and timed events (transitions, submissions, striking exchanges).

One gpt-6-astra call per interval via the logged-in Codex CLI. Uses the same usage guard as autolabel_astra.
The 2 fps label frames are rendered as 2x2 tile sheets with each tracked actor outlined and tagged in a fixed colour.
Output: autolabel/<run>/<trial>/<packet>/{fight.json, fight-receipt.json, fight-images/}
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw

from autolabel_astra import (EFFORT, MODEL, anchor, call_astra, check_usage, cohort_source, font, obj, enum, outline)
from autolabel_common import ROOT, interval_dir, rle_decode, sha256, write_json

PROMPT = ROOT/'docs/autolabel/FIGHT_PROMPT.v3.md'
ACTOR_COLOURS = [(230, 25, 75), (0, 130, 200), (255, 225, 25), (60, 180, 75), (245, 130, 48), (145, 30, 180)]
COLOUR_NAMES = ['red', 'blue', 'yellow', 'green', 'orange', 'purple']
L1A = ['standing', 'guard', 'side_control', 'mount', 'back', 'turtle', 'leg_entanglement', 'transition', 'not_applicable', 'unknown']
L1B = {'standing': ['open_space', 'clinch', 'body_lock', 'underhook_battle', 'collar_tie', 'russian_tie', 'front_headlock',
                    'single_leg', 'double_leg', 'throw'],
       'guard': ['closed', 'half', 'butterfly', 'de_la_riva', 'reverse_de_la_riva', 'spider', 'lasso', 'collar_sleeve',
                 'k_guard', 'single_leg_x', 'x_guard', 'deep_half', 'rubber', 'fifty_fifty', 'open_seated',
                 'open_supine', 'quarter'],
       'side_control': ['standard', 'kesa_gatame', 'reverse_kesa', 'north_south', 'twister_side'],
       'mount': ['low', 'high', 's_mount', 'technical_mount'],
       'back': ['seatbelt', 'body_triangle', 'one_hook', 'two_hooks', 'back_mount', 'truck', 'crab_ride'],
       'turtle': ['standard', 'flat', 'granby', 'sit_out', 'peek_out'],
       'leg_entanglement': ['ashi_garami', 'outside_ashi', 'inside_sankaku', '50_50', 'saddle', 'cross_ashi', '80_20']}
EVENTS = ['takedown', 'sweep', 'pass', 'back_take', 'mount_achieve', 'escape', 'reguard', 'leg_entry', 'reset', 'stand_up',
          'submission_attempt', 'tap', 'striking_exchange', 'knockdown', 'referee_stoppage']


def schema(actor_ids):
    actor = {'type': ['string', 'null']}
    details = sorted({d for v in L1B.values() for d in v}) + ['none', 'unknown']
    return obj({
        'frames': {'type': 'array', 'items': obj({
            'frame': {'type': 'string'},
            'engagement': enum(['idle', 'engaged', 'scramble', 'not_visible']),
            'position': enum(L1A), 'position_detail': enum(details), 'position_alt': enum(L1A),
            'confidence': enum(['high', 'medium', 'low']),
            'top_actor': actor, 'dominant_actor': {'type': 'string'}})},
        'events': {'type': 'array', 'items': obj({
            'type': enum(EVENTS), 'actor': actor, 'target': actor, 'start_frame': {'type': 'string'},
            'end_frame': {'type': 'string'}, 'outcome': enum(['success', 'failed', 'ongoing', 'unknown']),
            'detail': {'type': 'string'}})},
        'notes': {'type': 'string'}})


def render_frame(frame, colours, label, scale=2):
    with Image.open(frame['path']) as im:
        pixels = np.asarray(im.convert('RGB')).copy()
    tags = []
    for inst in frame['instances']:
        colour = colours.get(inst['actor_id'])
        if colour is None or not inst['area']:
            continue
        mask = rle_decode(inst['mask_rle'])
        pixels[outline(mask, 3*scale)] = colour
        tags.append((inst['actor_id'], colour, anchor(mask)))
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    tag_font = font(18*scale)
    for actor_id, colour, (cx, cy) in tags:
        w = draw.textlength(actor_id, font=tag_font) + 10
        draw.rectangle([cx - w/2, cy - 12*scale, cx + w/2, cy + 12*scale], fill=colour, outline='white', width=scale)
        draw.text((cx - w/2 + 5, cy - 11*scale), actor_id, fill='white', font=tag_font)
    banner_font = font(20*scale)
    bw = draw.textlength(label, font=banner_font) + 16
    draw.rectangle([0, 0, bw, 28*scale], fill='black')
    draw.text((8, 3), label, fill='white', font=banner_font)
    return image


def run(run_name, targets, effort=EFFORT, force=False):
    for trial_id, packet_id in targets:
        base = interval_dir(run_name, trial_id, packet_id)
        out_path = base/'fight.json'
        if out_path.exists() and not force:
            continue
        labels = json.loads((base/'labels.json').read_text())
        # all athletes/instructors (not just the primary pair: intervals can contain several pairs or a camera cut
        # to a different pair), primary first, then referee; capped at the 6 tag colours
        people = [a for a in labels['actors'] if a['role'] in ('athlete', 'instructor')]
        people.sort(key=lambda a: not a.get('primary'))
        tracked = (people + [a for a in labels['actors'] if a['role'] == 'referee'])[:6]
        colours = {a['actor_id']: ACTOR_COLOURS[i] for i, a in enumerate(tracked)}
        frames = labels['frames']
        fid = lambda i: f'F{i:02d}'
        images = []
        w, h = frames[0]['width'], frames[0]['height']
        for start in range(0, len(frames), 4):
            group = frames[start:start + 4]
            sheet = Image.new('RGB', (w, h), 'gray')
            for slot, f in enumerate(group):
                tile = render_frame(f, colours, f"{fid(f['frame_index'])} t={f['time_s']:.1f}s").resize((w//2, h//2), Image.LANCZOS)
                sheet.paste(tile, ((slot % 2)*(w//2), (slot//2)*(h//2)))
            path = base/'fight-images'/f"tiles-{fid(group[0]['frame_index'])}-{fid(group[-1]['frame_index'])}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(path, quality=90)
            images.append(path)
        src = cohort_source(trial_id)
        actor_lines = '\n'.join(f"- {a['actor_id']} ({COLOUR_NAMES[i]}): {a['role']}, {a['description']}" for i, a in enumerate(tracked))
        prompt = PROMPT.read_text().format(
            n_frames=len(frames), last=len(frames) - 1, actor_lines=actor_lines, sport=src.get('sport_hypothesis'),
            context=src.get('context_hypothesis'), title=src.get('title', '').replace('"', "'"),
            clip_context=labels['clip_context'], l1b='; '.join(f"{k}: {', '.join(v)}" for k, v in L1B.items()))
        usage_before = check_usage()
        result, receipt = call_astra(prompt, images, schema([a['actor_id'] for a in tracked]), effort)
        write_json(out_path, {'version': 1, 'stage': 'astra_fight_state', 'trial_id': trial_id, 'packet_id': packet_id,
                              'frames': [{'frame': fid(f['frame_index']), 'frame_index': f['frame_index'], 'time_s': f['time_s']} for f in frames],
                              'actors': {a['actor_id']: COLOUR_NAMES[i] for i, a in enumerate(tracked)}, 'result': result})
        write_json(base/'fight-receipt.json', {
            'model': MODEL, 'reasoning_effort': effort, 'auth': 'codex chatgpt login (no API key)',
            'prompt_template': str(PROMPT), 'prompt_template_sha256': sha256(PROMPT),
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'labels_sha256': sha256(base/'labels.json'),
            'images': [{'path': str(p), 'sha256': sha256(p)} for p in images],
            'created_at': datetime.now(timezone.utc).isoformat(), 'usage_before': usage_before, **receipt})
        print(f"{trial_id}/{packet_id}: {receipt['elapsed_s']}s usage={receipt['usage']}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v1')
    ap.add_argument('--effort', default=EFFORT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    run(a.run, [tuple(t.split('/')) for t in a.targets], a.effort, a.force)
