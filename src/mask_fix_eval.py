"""Before/after sheets for mask_fix on gold-judged actor-frames (no writes to labels). Output: reviews/autolabel-v1/maskfix/eval/"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

from autolabel_astra import font, outline
from autolabel_common import ROOT, rle_decode
from mask_fix import RUN, OUT, fix_interval, original_labels

ROWS = ROOT/'reviews/gold-v1/mask_postmortem/rows.json'
COL = {'red': (230, 25, 75), 'blue': (0, 130, 200)}


def crop(f, insts):
    H, W = f['height'], f['width']
    bs = [i['box_xyxy'] for i in insts if i and i.get('box_xyxy')]
    if not bs:
        return 0, 0, W, H
    x1, y1 = min(b[0] for b in bs), min(b[1] for b in bs)
    x2, y2 = max(b[2] for b in bs), max(b[3] for b in bs)
    p = 0.15*max(x2 - x1, y2 - y1)
    return int(max(0, x1 - p)), int(max(0, y1 - p)), int(min(W, x2 + p)), int(min(H, y2 + p))


def draw(f, insts, box, title):
    with Image.open(f['path']) as im:
        px = np.asarray(im.convert('RGB')).astype(np.float32)
    for inst, col in insts:
        if inst and inst.get('mask_rle') and inst['area']:
            m = rle_decode(inst['mask_rle'])
            px[m] = px[m]*0.55 + np.array(col)*0.45
            px[outline(m, 2)] = col
    img = Image.fromarray(px.clip(0, 255).astype(np.uint8)).crop(box)
    img.thumbnail((640, 480))
    tile = Image.new('RGB', (640, 480), 'gray')
    tile.paste(img, ((640 - img.width)//2, (480 - img.height)//2))
    d = ImageDraw.Draw(tile)
    d.rectangle([0, 0, 640, 26], fill='black')
    d.text((6, 3), title, fill='white', font=font(17))
    return tile


def main():
    rows = json.loads(ROWS.read_text())
    sel = [r for r in rows if r.get('source') == 'astra_box_to_sam31_mask' or r['label'] in ('partial', 'wrong') or r.get('praised')]
    by_iv = {}
    for r in sel:
        by_iv.setdefault(r['iv'], []).append(r)
    out = OUT/'eval'
    out.mkdir(parents=True, exist_ok=True)
    report, tiles = [], []
    for iv, rs in sorted(by_iv.items()):
        old = original_labels(RUN/iv)
        if (RUN/iv/'labels.pre-maskfix.json.gz').exists():  # already applied: compare backup vs current
            new = json.loads((RUN/iv/'labels.json').read_text())
        else:
            _, _, new = fix_interval(RUN/iv, write=False)
        pair = sorted(a['actor_id'] for a in old['actors'] if a.get('primary'))
        for fi in sorted({r['fi'] for r in rs}):
            fo, fn = old['frames'][fi], new['frames'][fi]
            get = lambda F, a: next((i for i in F['instances'] if i['actor_id'] == a), None)
            io = [(get(fo, a), c) for a, c in zip(pair, COL.values())]
            inn = [(get(fn, a), c) for a, c in zip(pair, COL.values())]
            changed = [a for a in pair if get(fn, a) and get(fn, a).get('maskfix')]
            if not changed and not any(r.get('praised') for r in rs if r['fi'] == fi):
                continue
            box = crop(fo, [i for i, _ in io] + [i for i, _ in inn])
            judged = '; '.join(f"{r['slot']}={r['label']}{'*' if r.get('praised') else ''}" for r in rs if r['fi'] == fi)
            acts = ','.join(f"{a}:{get(fn, a)['maskfix']['action']}" for a in changed) or 'untouched'
            name = f"{iv.replace('/', '_')}_f{fi:02d}.jpg"
            sheet = Image.new('RGB', (1280, 480))
            sheet.paste(draw(fo, io, box, f'BEFORE {iv}#{fi} gold: {judged}'), (0, 0))
            sheet.paste(draw(fn, inn, box, f'AFTER {acts}'), (640, 0))
            sheet.save(out/name, quality=85)
            report.append({'iv': iv, 'fi': fi, 'judged': judged, 'actions': acts, 'image': name})
    (out/'index.json').write_text(json.dumps(report, indent=1))
    print(len(report), 'sheets')


if __name__ == '__main__':
    main()
