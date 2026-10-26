"""Resumable public metadata discovery. Query context is a hypothesis, not a label."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from catalog import now

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=Path, default=ROOT/'manifests'/'discovery-queries.json')
    parser.add_argument('--per-query', type=int, default=100)
    parser.add_argument('--limit-queries', type=int, default=100)
    args = parser.parse_args()
    directory = ROOT/'manifests'/'discovery'
    directory.mkdir(exist_ok=True)
    for query in json.loads(args.queries.read_text())[:args.limit_queries]:
        key = hashlib.sha256((query['query']+str(args.per_query)).encode()).hexdigest()[:16]
        out = directory/f'{key}.json'
        if out.exists():
            continue
        command = [sys.executable,'-m','yt_dlp','--ignore-config','--js-runtimes','node',
                   '--flat-playlist','--skip-download','--socket-timeout','20','--retries','1',
                   '--extractor-retries','1','--dump-single-json', f"ytsearch{args.per_query}:{query['query']}"]
        with (ROOT/'receipts'/'discovery.log').open('a') as log:
            try:
                result = subprocess.run(command, stdout=subprocess.PIPE,stderr=log,text=True,timeout=240,check=True)
                raw = json.loads(result.stdout)
                rows = [{k:r.get(k) for k in ('id','url','title','duration','channel','channel_id','channel_url',
                         'uploader','view_count','availability','live_status')} for r in raw.get('entries',[]) if r]
                temp = out.with_suffix('.tmp')
                temp.write_text(json.dumps({'query':query,'retrieved_at':now(),'entries':rows},indent=2,ensure_ascii=False))
                temp.replace(out)
                print(json.dumps({'query':query['query'],'candidates':len(rows)}),flush=True)
            except (subprocess.SubprocessError,ValueError) as exc:
                print(json.dumps({'query':query['query'],'error':type(exc).__name__}),flush=True)
        time.sleep(2)
    unique = {}
    for path in sorted(directory.glob('*.json')):
        doc = json.loads(path.read_text())
        for row in doc['entries']:
            if not row['id']:
                continue
            if row['id'] not in unique:
                unique[row['id']] = row | {'discovery_evidence':[], 'review_state':'candidate_only'}
            unique[row['id']]['discovery_evidence'].append(doc['query'])
    dest = ROOT/'manifests'/'discovered-candidates.jsonl'
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in unique.values()))
    tmp.replace(dest)
    print(json.dumps({'unique_candidates':len(unique)}),flush=True)


if __name__ == '__main__':
    main()
