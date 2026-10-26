"""Review overlays for autolabel labels.json: per-actor coloured masks, labels and review flags on a contact sheet."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from autolabel_astra import font, outline
from autolabel_common import interval_dir, rle_decode

PALETTE = [(230, 25, 75), (0, 130, 200), (255, 225, 25), (60, 180, 75), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0)]
ACTOR_COLOURS = {f'P{k + 1}': c for k, c in enumerate(PALETTE)} | {f'P{k + 13}': c for k, c in enumerate(PALETTE)}
OTHER = (170, 170, 170)


def overlay(frame, width=640, fight=None):
    with Image.open(frame['path']) as im:
        image = im.convert('RGB')
    pixels = np.asarray(image).astype(np.float32)
    for inst in frame['instances']:
        mask = rle_decode(inst['mask_rle'])
        colour = np.array(ACTOR_COLOURS.get(inst['actor_id'], OTHER), dtype=np.float32)
        alpha = 0.45 if inst['actor_id'] in ACTOR_COLOURS else 0.2
        pixels[mask] = pixels[mask]*(1 - alpha) + colour*alpha
        pixels[outline(mask, 2)] = colour
    image = Image.fromarray(pixels.clip(0, 255).astype(np.uint8))
    scale = width/image.width
    image = image.resize((width, round(image.height*scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    f = font(13)
    for inst in frame['instances']:
        if inst['actor_id'] not in ACTOR_COLOURS or not inst['box_xyxy']:
            continue
        x1, y1 = inst['box_xyxy'][:2]
        src = 'box' if inst['source'].startswith('astra_box') else 'det'
        text = f"{inst['actor_id']} {inst['posture']} ({src})"
        draw.rectangle([x1*scale, y1*scale, x1*scale + draw.textlength(text, font=f) + 6, y1*scale + 16],
                       fill=ACTOR_COLOURS[inst['actor_id']])
        draw.text((x1*scale + 3, y1*scale + 1), text, fill='black', font=f)
    head = f"F{frame['frame_index']:02d}{' KEY' if frame['is_keyframe'] else ''}  {frame['pair_configuration']}  {frame['shot_type']}"
    if fight:
        head = f"F{frame['frame_index']:02d}  pos={fight.get('smoothed') or fight['position']}/{fight['position_detail']}  top={fight['top_actor']}  dom={fight['dominant_actor']}"
    draw.rectangle([0, 0, draw.textlength(head, font=f) + 8, 18], fill='black')
    draw.text((4, 2), head, fill='white', font=f)
    if frame['review_flags']:
        flag = '; '.join(frame['review_flags'])[:95]
        draw.rectangle([0, image.height - 18, image.width, image.height], fill=(120, 0, 0))
        draw.text((4, image.height - 16), flag, fill='white', font=font(11))
    return image


def contact_sheet(labels, indices, cols=3, width=640, fight=None):
    def fight_of(i):
        if not fight or i >= len(fight['frames']):
            return None
        return fight['frames'][i]
    tiles = [overlay(labels['frames'][i], width, fight_of(i)) for i in indices]
    th = tiles[0].height
    rows = (len(tiles) + cols - 1)//cols
    sheet = Image.new('RGB', (cols*width, rows*th + 30), 'black')
    draw = ImageDraw.Draw(sheet)
    actors = ', '.join(f"{a['actor_id']}={a['role']}: {a['description'][:50]}" for a in labels['actors'])
    draw.text((6, 6), f"{labels['trial_id']}/{labels['packet_id']} {labels['clip_context']} | {actors}"[:230], fill='white', font=font(13))
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % cols)*width, 30 + (k//cols)*th))
    return sheet


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v1')
    ap.add_argument('--frames', default='', help='comma-separated frame indices (default: every 4th)')
    ap.add_argument('--out')
    ap.add_argument('target', help='trialNNN/intervalNN')
    a = ap.parse_args()
    trial_id, packet_id = a.target.split('/')
    base = interval_dir(a.run, trial_id, packet_id)
    labels = json.loads((base/'labels.json').read_text())
    n = len(labels['frames'])
    indices = [min(int(i), n - 1) for i in a.frames.split(',')] if a.frames else list(range(0, n, 4))
    out = Path(a.out) if a.out else base/'review'/f"sheet-{'-'.join(map(str, indices[:1] + indices[-1:]))}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fight = None
    if (base/'fight.json').exists():
        fight = json.loads((base/'fight.json').read_text())['result']
        if (base/'smooth.json').exists():
            sp = json.loads((base/'smooth.json').read_text()).get('position', {})
            for k, (f, v) in enumerate(zip(fight['frames'], sp.get('smoothed', []))):
                f['smoothed'] = v
                if 'detail' in sp:
                    f['position_detail'], f['top_actor'] = sp['detail'][k], sp['top'][k]
    contact_sheet(labels, indices, fight=fight).save(out, quality=88)
    print(out)
