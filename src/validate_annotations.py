"""Validate grounding/structure without deleting unfamiliar rich label fields."""
import argparse
import json
import math
from pathlib import Path


def validate(doc,evidence):
    errors=[]
    def bad(path,message): errors.append(f'{path}: {message}')
    def time_range(value,path,lo=0,hi=None):
        hi=evidence['duration_s'] if hi is None else hi
        if (not isinstance(value,list) or len(value)!=2 or
            any(not isinstance(x,(float,int)) or not math.isfinite(x) for x in value) or
            not lo<=value[0]<value[1]<=hi):
            bad(path,'invalid time_range'); return False
        return True
    if doc.get('source_key')!=evidence['source_key']:
        bad('source_key','source mismatch')
    for field in ('packet_id','provenance','inspection','actors','claims'):
        if field not in doc: bad(field,'missing required field')
    pr=doc.get('time_range_s')
    if not time_range(pr,'packet.time_range'):
        pr=[0,evidence['duration_s']]
    if 'packet_id' in evidence and doc.get('packet_id')!=evidence['packet_id']:
        bad('packet_id','does not match frozen evidence packet')
    if 'time_range_s' in evidence and doc.get('time_range_s')!=evidence['time_range_s']:
        bad('packet.time_range','does not match frozen evidence interval')
    actor_ids=[a.get('actor_id') for a in doc.get('actors',[])]
    if None in actor_ids or len(set(actor_ids))!=len(actor_ids):
        bad('actors','missing or duplicate actor IDs')
    actors=set(actor_ids)
    frames={f['frame_id']:f for f in evidence.get('frames',[])}
    speech={s['id']:s for s in evidence.get('transcript_segments',[])}
    inspected=set(doc.get('inspection',{}).get('frame_ids',[]))
    read_segments=set(doc.get('inspection',{}).get('transcript_segment_ids',[]))
    if inspected-set(frames): bad('inspection','unknown frame')
    if read_segments-set(speech): bad('inspection','unknown transcript segment')
    seen=set()
    kinds={'visual_observation','temporal_inference','spoken_statement','uncertain_hypothesis'}
    for i,claim in enumerate(doc.get('claims',[])):
        path=f'claims[{i}]'
        cid=claim.get('claim_id')
        if not cid or cid in seen: bad(path,'missing or duplicate claim_id')
        seen.add(cid)
        time_range(claim.get('time_range_s'),path+'.time_range',*pr)
        if not claim.get('kind') or not claim.get('text'): bad(path,'kind and text required')
        if set(claim.get('actor_ids',[]))-actors: bad(path,'unknown actor reference')
        kind=claim.get('evidence_kind')
        if kind not in kinds: bad(path,'unknown evidence_kind')
        fids=set(claim.get('frame_ids',[]))
        sids=set(claim.get('transcript_segment_ids',[]))
        if fids-set(frames): bad(path,'unknown frame reference')
        if sids-set(speech): bad(path,'unknown transcript reference')
        if fids-inspected: bad(path,'frame not inspected')
        if sids-read_segments: bad(path,'transcript not inspected')
        if kind in ('visual_observation','temporal_inference') and not fids:
            bad(path,'visual evidence required')
        if kind=='spoken_statement' and not sids:
            bad(path,'speech evidence required')
        if kind=='uncertain_hypothesis' and not (fids or sids):
            bad(path,'at least one evidence reference required')
    for i,geo in enumerate(doc.get('geometry',[])):
        path=f'geometry[{i}]'
        if geo.get('actor_id') not in actors: bad(path,'unknown actor')
        fid=geo.get('frame_id')
        if fid not in frames: bad(path,'unknown frame')
        elif fid not in inspected: bad(path,'frame not inspected')
        box=geo.get('bbox_xyxy_normalized')
        if box is not None:
            if (not isinstance(box,list) or len(box)!=4 or
                any(not isinstance(x,(float,int)) or not math.isfinite(x) or not 0<=x<=1 for x in box) or
                not (box[0]<box[2] and box[1]<box[3])):
                bad(path,'invalid bbox')
        for point in geo.get('keypoints',[]):
            xy=point.get('xy_normalized')
            if xy is not None and (not isinstance(xy,list) or len(xy)!=2 or
                                   any(not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1 for x in xy)):
                bad(path,'invalid keypoint')
    try:
        json.dumps(doc,allow_nan=False)
    except (TypeError,ValueError):
        bad('document','must be finite JSON data')
    return errors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('annotation',type=Path)
    parser.add_argument('evidence',type=Path)
    args=parser.parse_args()
    errors=validate(json.loads(args.annotation.read_text()),json.loads(args.evidence.read_text()))
    print(json.dumps({'valid':not errors,'errors':errors},indent=2))
    raise SystemExit(bool(errors))


if __name__=='__main__':
    main()
