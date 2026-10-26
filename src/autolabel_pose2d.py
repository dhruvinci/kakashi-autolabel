"""Stage 5a: per-actor 2D skeletons (COCO-17) on dense frames with ViTPose+ base (runs in .venv-pose).

Mask-guided: each actor is estimated from its own box, with the OTHER tracked actors' pixels greyed out so limbs
cannot be taken from the partner. Each joint is then tagged:
  own        on the actor's own (slightly dilated) mask
  partner    on another tracked actor's mask (ownership conflict; treat as unreliable)
  off_mask   on neither (occluded/inferred, or a pose error)
Output: autolabel/<run>/<trial>/<packet>/dense/pose2d.npz (keypoints, scores, tags) + pose2d.json
"""
import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from autolabel_common import interval_dir, mask_box, write_json

MODEL = 'usyd-community/vitpose-plus-base'
COCO17 = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
          'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
          'left_ankle', 'right_ankle']
TAGS = {'own': 0, 'partner': 1, 'off_mask': 2}
BATCH = 16


def dilate(mask, steps=6):
    import cv2
    return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=steps).astype(bool)


def load_model():
    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    processor = AutoProcessor.from_pretrained(MODEL)
    model = VitPoseForPoseEstimation.from_pretrained(MODEL, torch_dtype=torch.float16).to('cuda').eval()
    return processor, model


def estimate(processor, model, images, boxes_xywh):
    """images: list of PIL; boxes_xywh: list of [N_i, 4]. Returns list of (keypoints[N_i,17,2], scores[N_i,17])."""
    import torch
    inputs = processor(images, boxes=boxes_xywh, return_tensors='pt').to('cuda')
    inputs['pixel_values'] = inputs['pixel_values'].half()
    n = inputs['pixel_values'].shape[0]
    with torch.inference_mode():
        out = model(**inputs, dataset_index=torch.zeros(n, dtype=torch.long, device='cuda'))
    out.heatmaps = out.heatmaps.float()  # scipy post-processing rejects fp16
    results = processor.post_process_pose_estimation(out, boxes=boxes_xywh)
    return [(np.stack([p['keypoints'].float().cpu().numpy() for p in r]) if r else np.zeros((0, 17, 2)),
             np.stack([p['scores'].float().cpu().numpy() for p in r]) if r else np.zeros((0, 17)))
            for r in results]


def estimate_jobs(processor, model, jobs, h, w):
    """jobs: (frame_index, actor, image_path, box_xywh, own_mask, partner_mask). Mask-guided ViTPose + joint tags."""
    keypoints = np.zeros((len(jobs), 17, 2), np.float32)
    scores = np.zeros((len(jobs), 17), np.float32)
    tags = np.zeros((len(jobs), 17), np.uint8)
    for s in range(0, len(jobs), BATCH):
        batch = jobs[s:s + BATCH]
        images = []
        for k, a, path, box, m, others in batch:
            with Image.open(path) as im:
                arr = np.asarray(im.convert('RGB')).copy()
            arr[others & ~m] = 128  # grey out the other tracked actors
            images.append(Image.fromarray(arr))
        results = estimate(processor, model, images, [[b[3]] for b in batch])
        for j, (kp, sc) in enumerate(results):
            i = s + j
            keypoints[i], scores[i] = kp[0], sc[0]
            own, other = dilate(batch[j][4]), batch[j][5]
            for q in range(17):
                x, y = int(round(kp[0, q, 0])), int(round(kp[0, q, 1]))
                if not (0 <= x < w and 0 <= y < h):
                    tags[i, q] = TAGS['off_mask']
                elif own[y, x]:
                    tags[i, q] = TAGS['own']
                elif other[y, x]:
                    tags[i, q] = TAGS['partner']
                else:
                    tags[i, q] = TAGS['off_mask']
    return keypoints, scores, tags


def run(run_name, targets, force=False, cleanup=True):
    processor = model = None
    for trial_id, packet_id in targets:
        base = interval_dir(run_name, trial_id, packet_id)
        dense = base/'dense'
        if not (dense/'dense.json').exists() or ((dense/'pose2d.json').exists() and not force):
            continue
        if processor is None:
            processor, model = load_model()
        d = json.loads((dense/'dense.json').read_text())
        h, w = d['shape']
        z = np.load(dense/'masks.npz')
        started = time.time()
        jobs = []  # (frame_index, actor, image_path, box_xywh, own_mask, partner_mask)
        for f in d['frames']:
            k = f['dense_index']
            masks = {a: np.unpackbits(z[f'{k}_{a}'])[:h*w].reshape(h, w).astype(bool) for a in f['actor_areas']}
            for a, m in masks.items():
                box = mask_box(m)
                if box is None or m.sum() < 400:
                    continue
                others = np.zeros((h, w), bool)
                for b, o in masks.items():
                    if b != a:
                        others |= o
                x1, y1, x2, y2 = box
                pad = 0.1*max(x2 - x1, y2 - y1)
                bx = [max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)]
                jobs.append((k, a, dense/'frames'/f'{k:05d}.jpg', [bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]], m, others))
        keypoints, scores, tags = estimate_jobs(processor, model, jobs, h, w)
        np.savez_compressed(dense/'pose2d.npz', keypoints=keypoints, scores=scores, tags=tags,
                            frame_index=np.array([j[0] for j in jobs], np.int32),
                            actor=np.array([j[1] for j in jobs]))
        good = (scores > 0.3) & (tags == TAGS['own'])
        summary = {'instances': len(jobs), 'joints_confident_on_own_mask_pct': round(100*float(good.mean()), 1) if len(jobs) else None,
                   'joints_on_partner_pct': round(100*float((tags == TAGS['partner']).mean()), 1) if len(jobs) else None}
        write_json(dense/'pose2d.json', {'version': 1, 'stage': 'vitpose_plus_base_mask_guided', 'model': MODEL,
                                         'keypoint_names': COCO17, 'tags': TAGS, 'elapsed_s': round(time.time() - started, 1),
                                         'created_at': datetime.now(timezone.utc).isoformat(), 'summary': summary})
        if cleanup:
            # dense JPEGs are regenerable from the master (dense.json keeps exact times); keep disk above the 30 GB reserve
            import shutil
            shutil.rmtree(dense/'frames', ignore_errors=True)
        print(f'{trial_id}/{packet_id}: {summary} {time.time() - started:.0f}s', flush=True)


def run_labels(run_name, targets, force=False):
    """2 fps intervals (not densified): skeletons on the label frames, from labels.json masks.
    Output: <interval>/pose2d_labels.npz + pose2d_labels.json (same fields as the dense pose2d)."""
    from autolabel_common import rle_decode
    processor = model = None
    roles = ('athlete', 'instructor', 'referee', 'coach_or_corner')
    for trial_id, packet_id in targets:
        base = interval_dir(run_name, trial_id, packet_id)
        if not (base/'labels.json').exists() or ((base/'pose2d_labels.json').exists() and not force):
            continue
        if processor is None:
            processor, model = load_model()
        L = json.loads((base/'labels.json').read_text())
        started, jobs = time.time(), []
        for f in L['frames']:
            h, w = f['height'], f['width']
            masks = {i['actor_id']: rle_decode(i['mask_rle']) for i in f['instances'] if i['role'] in roles and i['area'] >= 400}
            for a, m in masks.items():
                others = np.zeros((h, w), bool)
                for b, o in masks.items():
                    if b != a:
                        others |= o
                x1, y1, x2, y2 = mask_box(m)
                pad = 0.1*max(x2 - x1, y2 - y1)
                bx = [max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)]
                jobs.append((f['frame_index'], a, f['path'], [bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]], m, others))
        if jobs:
            keypoints, scores, tags = estimate_jobs(processor, model, jobs, h, w)
        else:
            keypoints, scores, tags = np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32), np.zeros((0, 17), np.uint8)
        np.savez_compressed(base/'pose2d_labels.npz', keypoints=keypoints, scores=scores, tags=tags,
                            frame_index=np.array([j[0] for j in jobs], np.int32), actor=np.array([j[1] for j in jobs]))
        good = (scores > 0.3) & (tags == TAGS['own'])
        summary = {'instances': len(jobs), 'joints_confident_on_own_mask_pct': round(100*float(good.mean()), 1) if len(jobs) else None}
        write_json(base/'pose2d_labels.json', {'version': 1, 'stage': 'vitpose_plus_base_mask_guided_label_frames', 'model': MODEL,
                                               'keypoint_names': COCO17, 'tags': TAGS, 'elapsed_s': round(time.time() - started, 1),
                                               'created_at': datetime.now(timezone.utc).isoformat(), 'summary': summary})
        print(f'{trial_id}/{packet_id}: label-frame pose {summary} {time.time() - started:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v1')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--keep-frames', action='store_true')
    ap.add_argument('--label-frames', action='store_true', help='2 fps intervals: skeletons on labels.json frames')
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    targets = [tuple(t.split('/')) for t in a.targets]
    if a.label_frames:
        run_labels(a.run, targets, a.force)
    else:
        run(a.run, targets, a.force, cleanup=not a.keep_frames)
