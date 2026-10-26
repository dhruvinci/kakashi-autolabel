"""Bind existing decoded evidence to the verified master; never decode media."""
import hashlib
import json
import os
from pathlib import Path
import re

from catalog import Catalog, now

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BINDINGS=ROOT/'evidence/pilot-v1/evidence-bindings-v1.json'


def absolute(path):
    path=Path(path)
    return (path if path.is_absolute() else ROOT/path).resolve()


def sha256(path):
    with absolute(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def checked_json(record):
    raw=absolute(record['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['sha256']:
        raise ValueError('Evidence hash mismatch: '+str(record['path']))
    return json.loads(raw)


def verify_frames(frames):
    ids=set()
    for frame in frames:
        if frame['frame_id'] in ids:raise ValueError('Duplicate frame ID')
        ids.add(frame['frame_id'])
        if sha256(frame['path'])!=frame['sha256']:
            raise ValueError('Image hash mismatch: '+frame['frame_id'])


def load_verified_evidence(source, frames_manifest, bindings_path=DEFAULT_BINDINGS):
    """Return (original frame records, full transcript) after identity/hash checks."""
    binding=json.loads(Path(bindings_path).read_text())
    if (source.get('status')!='complete' or source['source_key']!=binding['source_key'] or
        source['sha256']!=binding['source_sha256'] or
        absolute(source['media_path'])!=absolute(binding['master_path'])):
        raise ValueError('Source binding mismatch')
    if sha256(source['media_path'])!=source['sha256']:
        raise ValueError('Master hash mismatch')
    matches=[r for r in binding['manifests'] if absolute(r['path'])==absolute(frames_manifest)]
    if len(matches)!=1:raise ValueError('Expected a registered manifest')
    frames=checked_json(matches[0]);verify_frames(frames)
    transcript=checked_json(binding['transcript'])
    if transcript['source_sha256']!=source['sha256']:
        raise ValueError('Source/transcript mismatch')
    return frames,transcript


def freeze_pilot_binding():
    """Audit existing provenance once and exclusively create the pilot receipt."""
    if DEFAULT_BINDINGS.exists():raise FileExistsError('Evidence binding already frozen')
    db=Catalog(ROOT/'manifests/catalog.sqlite')
    key=os.environ.get('KAKASHI_PILOT_SOURCE_KEY')  # the pilot source key is not published
    if not key:raise ValueError('Set KAKASHI_PILOT_SOURCE_KEY to the pilot source key')
    try:source=db.get(key)
    finally:db.close()
    if not source or source['status']!='complete' or sha256(source['media_path'])!=source['sha256']:
        raise ValueError('Verified master required')
    base=ROOT/'evidence/pilot-v1'
    def ref(path):return {'path':str(absolute(path)),'sha256':sha256(path)}
    baseline=base/'frames.json';combined=base/'frames.full-v1.json'
    derivation=ROOT/'receipts/calibration-v1/full-evidence-preparation.json'
    enrichment=base/'enrichment-v1/index.json'
    frames=json.loads(baseline.read_text());full=json.loads(combined.read_text())
    report=json.loads(derivation.read_text());extra=json.loads(enrichment.read_text())
    if extra['source_key']!=source['source_key']:raise ValueError('Enrichment source mismatch')
    logs=[base/'frames-decode.log']
    extras=[]
    for request in extra['requests']:
        logs.append(base/'enrichment-v1'/request['request_id']/'decode.log')
        extras.extend(request['frames'])
    log_records=[]
    for log in logs:
        inputs=re.findall(r"^Input #\d+,.*? from '(.+)':$",log.read_text(),re.M)
        if len(inputs)!=1 or absolute(inputs[0])!=absolute(source['media_path']):
            raise ValueError('Decode log master mismatch: '+str(log))
        log_records.append(ref(log)|{'recorded_input_path':inputs[0]})
    verify_frames(frames);verify_frames(extras);verify_frames(full)
    expected={f['time_s']:f for f in frames};aliases=[]
    for frame in extras:
        if frame['time_s'] in expected:
            canonical=expected[frame['time_s']]
            aliases.append({'extra_frame_id':frame['frame_id'],'canonical_frame_id':canonical['frame_id'],
                            'time_s':frame['time_s'],'encoded_image_hash_equal':frame['sha256']==canonical['sha256']})
        else:expected[frame['time_s']]=frame
    if (full!=sorted(expected.values(),key=lambda f:f['time_s']) or
        report['combined_manifest_sha256']!=sha256(combined) or report['aliases']!=aliases or
        report['base_frames']!=len(frames) or report['enrichment_frames']!=len(extras) or
        report['combined_frames']!=len(full) or report['added_unique_timestamps']!=len(full)-len(frames)):
        raise ValueError('Combined manifest derivation mismatch')
    transcript_path=base/'transcript-small-en.json'
    transcript=json.loads(transcript_path.read_text())
    if transcript['source_sha256']!=source['sha256']:raise ValueError('Source/transcript mismatch')
    receipt={'version':1,'created_at':now(),'source_key':source['source_key'],
             'source_sha256':source['sha256'],'master_path':str(absolute(source['media_path'])),
             'manifests':[ref(baseline),ref(combined)],'transcript':ref(transcript_path),
             'provenance':{'baseline_decode_log':log_records[0],'enrichment_decode_logs':log_records[1:],
                           'enrichment_index':ref(enrichment),'combined_derivation_receipt':ref(derivation),
                           'verified_image_records':len(frames)+len(extras)+len(full),
                           'baseline_frames':len(frames),'combined_frames':len(full),
                           'method':'Retrospective verification of existing master, images, manifests, decode-log input paths and deterministic union. No new decoding occurred.'}}
    with DEFAULT_BINDINGS.open('x') as output:json.dump(receipt,output,indent=2)
    return receipt


if __name__=='__main__':
    print(json.dumps(freeze_pilot_binding(),indent=2))
