"""Stage 8: second-opinion verification of risky frames (Astra), with Claude adjudication of disagreements.

Frames with review or risk flags on the primary pair get rendered as zoomed crops: the primary actors are filled
red (first) and blue (second) and tagged. Up to 8 frames go into one Astra call (2x4 sheet, ~16k image tokens per
call). For each frame and actor Astra judges the mask (correct | missing_parts | includes_other_person |
wrong_person | merged_both) and which body parts are missing. For each frame it judges the fight position label.
Output: autolabel/<run>/<trial>/<packet>/verify.json. Disagreements between Astra's verdict and the pipeline flags
are queued in reviews/autolabel-v1/adjudication-queue.jsonl for Claude reviewer agents (a small share only).
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw

from autolabel_astra import EFFORT, call_astra, check_usage, enum, font, obj, outline
from autolabel_common import ROOT, interval_dir, mask_box, rle_decode, write_json

PER_CALL = 8
COLOURS = [(230, 25, 75), (0, 130, 200)]
NAMES = ['red', 'blue']
PARTS = ['head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg', 'hips', 'feet', 'hands']
PROMPT = """You are checking automatic labels on combat-sports frames (MMA / grappling). Answer only from the attached image. Do not run commands or read files.

The image is a sheet of {n} zoomed frame crops (V1..V{n}, left-to-right, top-to-bottom, id in each top-left banner). In each crop two athletes are highlighted: the RED-filled region is actor {a} ("{da}") and the BLUE-filled region is actor {b} ("{db}"). Highlights should cover exactly that person's visible body.

For every crop and both actors, judge the highlight:
- correct: covers that person's visible body (small edge errors are fine)
- missing_parts: on the right person but clearly misses visible body parts (list them)
- includes_other_person: covers this person but also a large part of the other athlete
- wrong_person: mostly covers the other athlete or background
- merged_both: one highlight covers both athletes
Also judge the position label printed in each crop banner ("pos=..."): agree, or give the correct one from: standing, guard, side_control, mount, back, turtle, leg_entanglement, not_applicable, unknown. Use the Kakashi rules: any bottom-leg entanglement = guard (half guard), not side control; chest-to-back = back, chest-to-chest straddle = mount.
Be strict about ownership: entangled limbs belong to whoever's body they attach to."""


def schema(n):
    actor = obj({'verdict': enum(['correct', 'missing_parts', 'includes_other_person', 'wrong_person', 'merged_both']),
                 'missing': {'type': 'array', 'items': enum(PARTS)}})
    return obj({'frames': {'type': 'array', 'items': obj({
        'crop': {'type': 'string'}, 'red': actor, 'blue': actor,
        'position_ok': {'type': 'boolean'},
        'position_correct': enum(['standing', 'guard', 'side_control', 'mount', 'back', 'turtle', 'leg_entanglement',
                                  'not_applicable', 'unknown'])})},
        'notes': {'type': 'string'}})


def crop_image(frame, pair, position, label):
    with Image.open(frame['path']) as im:
        px = np.asarray(im.convert('RGB')).astype(np.float32)
    h, w = px.shape[:2]
    boxes = []
    for k, actor in enumerate(pair):
        inst = next((i for i in frame['instances'] if i['actor_id'] == actor), None)
        if inst is None or not inst['area']:
            continue
        m = rle_decode(inst['mask_rle'])
        px[m] = px[m]*0.55 + np.array(COLOURS[k])*0.45
        px[outline(m, 3)] = COLOURS[k]
        boxes.append(mask_box(m))
    image = Image.fromarray(px.clip(0, 255).astype(np.uint8))
    if boxes:
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        pad = 0.15*max(x2 - x1, y2 - y1)
        image = image.crop((max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)))
    image.thumbnail((640, 480))
    tile = Image.new('RGB', (640, 480), 'gray')
    tile.paste(image, ((640 - image.width)//2, (480 - image.height)//2))
    d = ImageDraw.Draw(tile)
    text = f'{label}  pos={position}'
    d.rectangle([0, 0, d.textlength(text, font=font(18)) + 10, 24], fill='black')
    d.text((5, 2), text, fill='white', font=font(18))
    return tile


def run(run_name, targets, effort=EFFORT, force=False):
    queue_path = ROOT/'reviews/autolabel-v1/adjudication-queue.jsonl'
    for trial_id, packet_id in targets:
        base = interval_dir(run_name, trial_id, packet_id)
        if (base/'verify.json').exists() and not force:
            continue
        L = json.loads((base/'labels.json').read_text())
        primary = sorted(a['actor_id'] for a in L['actors'] if a.get('primary'))
        if len(primary) != 2:
            continue
        desc = {a['actor_id']: a['description'][:80] for a in L['actors']}
        smooth = json.loads((base/'smooth.json').read_text()) if (base/'smooth.json').exists() else {}
        positions = smooth.get('position', {}).get('smoothed') or ['unknown']*len(L['frames'])
        risky = [k for k, f in enumerate(L['frames']) if f['review_flags'] or f.get('risk_flags')]
        if not risky:
            write_json(base/'verify.json', {'version': 1, 'frames': [], 'note': 'no risky frames'})
            continue
        results = []
        for s in range(0, len(risky), PER_CALL):
            group = risky[s:s + PER_CALL]
            cols = 4
            sheet = Image.new('RGB', (640*cols, 480*((len(group) + cols - 1)//cols)), 'black')
            for j, k in enumerate(group):
                sheet.paste(crop_image(L['frames'][k], primary, positions[k], f'V{j + 1}'), ((j % cols)*640, (j//cols)*480))
            path = base/'verify-images'/f'sheet-{s//PER_CALL:02d}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(path, quality=88)
            prompt = PROMPT.format(n=len(group), a=primary[0], b=primary[1], da=desc[primary[0]], db=desc[primary[1]])
            usage = check_usage()
            res, receipt = call_astra(prompt, [path], schema(len(group)), effort)
            by_crop = {r['crop']: r for r in res['frames']}
            for j, k in enumerate(group):
                r = by_crop.get(f'V{j + 1}')
                if r is None:
                    continue
                f = L['frames'][k]
                ok = r['red']['verdict'] == 'correct' and r['blue']['verdict'] == 'correct'
                results.append({'frame_index': f['frame_index'], 'time_s': f['time_s'], 'actors': primary,
                                'red': r['red'], 'blue': r['blue'], 'masks_ok': ok,
                                'position_label': positions[k], 'position_ok': r['position_ok'],
                                'position_correct': r['position_correct'],
                                'pipeline_flags': f['review_flags'] + f.get('risk_flags', []), 'sheet': str(path)})
            print(f'{trial_id}/{packet_id}: verified {len(group)} frames, {receipt["elapsed_s"]}s', flush=True)
        write_json(base/'verify.json', {'version': 1, 'stage': 'astra_risky_frame_verification',
                                        'created_at': datetime.now(timezone.utc).isoformat(), 'frames': results})
        # Disagreement = Astra says masks fine although a REVIEW flag fired, or Astra rejects a position the smoother kept.
        disputed = [r | {'trial_id': trial_id, 'packet_id': packet_id} for r in results
                    if (r['masks_ok'] and any(':' in fl for fl in r['pipeline_flags'] if 'merged_detection' not in fl))
                    or not r['position_ok']]
        if disputed:
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            with queue_path.open('a') as q:
                for r in disputed:
                    q.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v1')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    run(a.run, [tuple(t.split('/')) for t in a.targets], force=a.force)
