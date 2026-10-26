"""Lossless claim views and selected-frame COCO boxes; no fabricated Q/A or pose."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from catalog import Catalog,now
from validate_annotations import validate
from evidence_integrity import load_verified_evidence

ROOT=Path(__file__).resolve().parents[1]
KINDS=('visual_observation','temporal_inference','spoken_statement','uncertain_hypothesis')


def build_exports(doc,evidence):
    errors=validate(doc,evidence)
    if errors:raise ValueError('; '.join(errors))
    frames={r['frame_id']:r for r in evidence['frames']}
    speech={r['id']:r for r in evidence.get('transcript_segments',[])}
    claims=[];by_evidence={kind:[] for kind in KINDS}
    for claim in doc['claims']:
        row=copy.deepcopy(claim);row['source_key']=doc['source_key']
        row['frame_refs']=[{k:v for k,v in frames[fid].items() if k in ('frame_id','time_s','image','width','height','sha256')} for fid in claim.get('frame_ids',[])]
        row['speech_refs']=[{k:v for k,v in speech[sid].items() if k in ('id','start','end','text')} for sid in claim.get('transcript_segment_ids',[])]
        claims.append(row);by_evidence[row['evidence_kind']].append(row)
    images=[];boxes=[];geometry=[];image_ids={}
    withheld_frames={g['frame_id'] for g in doc.get('geometry',[]) if g.get('training_eligible') is False}
    for geo in doc.get('geometry',[]):
        frame=frames[geo['frame_id']]
        geometry.append(copy.deepcopy(geo)|{'source_key':doc['source_key'],'frame':copy.deepcopy(frame)})
        if geo.get('bbox_xyxy_normalized') is None or geo['frame_id'] in withheld_frames:continue
        fid=geo['frame_id']
        if fid not in image_ids:
            image_ids[fid]=len(images)+1
            images.append({'id':image_ids[fid],'file_name':frame['image'],'width':frame['width'],
                           'height':frame['height'],'frame_id':fid,'time_s':frame['time_s'],
                           'label_completeness':'selected_actors_only','negative_regions_verified':False})
        x1,y1,x2,y2=geo['bbox_xyxy_normalized'];w=frame['width'];h=frame['height']
        box=[round(x1*w,6),round(y1*h,6),round((x2-x1)*w,6),round((y2-y1)*h,6)]
        boxes.append({'id':len(boxes)+1,'image_id':image_ids[fid],'category_id':1,
                      'bbox':box,'area':round(box[2]*box[3],6),'iscrowd':0,
                      'actor_id':geo['actor_id'],'quality':geo.get('quality','unspecified'),
                      'bbox_semantics':geo.get('bbox_semantics','see original geometry record')})
    coco={'info':{'description':'Partial AI-proposed selected-actor boxes; approximate; see manifest for review state. Unlabelled people/regions are not verified negatives. Use partial-label-aware training or positive crops, not unqualified detector training.',
                  'withheld_frame_ids':sorted(withheld_frames)},
          'images':images,'annotations':boxes,'categories':[{'id':1,'name':'person'}]}
    return {'claims':claims,'by_evidence':by_evidence,'geometry':geometry,'coco':coco}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames-manifest',type=Path,default=ROOT/'evidence/pilot-v1/frames.json')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Export directory exists; use an explicit new version')
    doc=json.loads(args.annotations.read_text())
    db=Catalog(ROOT/'manifests/catalog.sqlite');source=db.get(doc['source_key']);db.close()
    if not source or source['status']!='complete':raise ValueError('Verified source required')
    frames,transcript=load_verified_evidence(source,args.frames_manifest)
    for frame in frames:
        path=Path(frame.pop('path'))
        if not path.is_file():raise FileNotFoundError(path)
        frame['image']=str(path.relative_to(ROOT))
    evidence={'source_key':source['source_key'],'duration_s':source['duration'],'frames':frames,
              'transcript_segments':transcript['segments']}
    out=build_exports(doc,evidence);args.output.mkdir(parents=True)
    views={'claims':out['claims'],'geometry':out['geometry']}|out['by_evidence']
    for name,rows in views.items():
        (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    (args.output/'coco.json').write_text(json.dumps(out['coco'],indent=2))
    manifest={'created_at':now(),'source_key':source['source_key'],'source_url':source['url'],
              'source_sha256':source['sha256'],'master_path':source['media_path'],
              'split':source['metadata'].get('split'),
              'label_status':doc.get('review',{}).get('status','ai_proposal; independent review pending'),
              'review':copy.deepcopy(doc.get('review',{})),
              'input_path':str(args.annotations.resolve().relative_to(ROOT)),
              'input_sha256':hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
              'image_root_relative_to_export':os.path.relpath(ROOT,args.output.resolve()),
              'counts':{name:len(rows) for name,rows in views.items()},
              'coco_images':len(out['coco']['images']),'coco_boxes':len(out['coco']['annotations']),
              'coverage':doc.get('coverage'),'rights_status':source['metadata'].get('rights_status','not_cleared_for_redistribution'),
              'notes':['Counts are overlapping views, not independent examples.','Arbitrary landmarks remain in geometry.jsonl; they are not a fabricated standard pose skeleton.','All unfamiliar fields remain in the canonical input; claims retain their free attributes.','Speech is derived from unverified local ASR.','COCO contains partial selected-actor proposals, not exhaustive person labels. Unlabelled people and regions are not verified negatives.','Any frame with an explicitly training-ineligible geometry proposal is withheld entirely from COCO; its raw geometry remains available.']}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'output':str(args.output),'counts':manifest['counts'],'coco_images':manifest['coco_images']}))


if __name__=='__main__':main()
