"""One resumable download worker; no inference APIs and no subclip media."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from catalog import Catalog, StorageGuard, StorageLimit, now

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'bv[height<=720]+ba/b[height<=720]'
INFO_FIELDS = ('id','title','description','duration','webpage_url','original_url','channel',
               'channel_id','channel_url','uploader','upload_date','license','availability',
               'tags','categories','chapters','language','live_status','extractor_key')
FORMAT_FIELDS = ('format_id','ext','width','height','fps','vcodec','acodec','filesize',
                 'filesize_approx','tbr','vbr','abr','audio_channels')


def summarize_info(info):
    out = {k: info[k] for k in INFO_FIELDS if k in info}
    formats = info.get('requested_formats') or [info]
    out['selected_formats'] = [{k:f[k] for k in FORMAT_FIELDS if k in f} for f in formats]
    out['caption_languages'] = list((info.get('subtitles') or {}).keys())
    out['auto_caption_languages'] = list((info.get('automatic_captions') or {}).keys())
    out['rights_status'] = 'recorded_license_only; redistribution_not_cleared'
    return out


def validate_probe(probe):
    duration = float(probe.get('format', {}).get('duration', 0))
    videos = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
    audios = [s for s in probe.get('streams', []) if s.get('codec_type') == 'audio']
    if not math.isfinite(duration) or duration <= 0 or not videos or not audios:
        raise ValueError('Master must have a positive duration, video and audio')
    if any(not 0 < s.get('height', 0) <= 720 for s in videos):
        raise ValueError('Video is missing its dimensions or exceeds 720p')
    return {'duration':duration, 'streams':probe['streams']}


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_guarded(command, *, cwd, log_path, guard, timeout_s):
    guard.check()
    start = time.monotonic()
    with open(log_path, 'a') as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                guard.check()
                if time.monotonic() - start > timeout_s:
                    raise subprocess.TimeoutExpired(command, timeout_s)
                time.sleep(0.1)
            guard.check()
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            stop_process(process)


def ytdlp_base():
    return [sys.executable, '-m', 'yt_dlp', '--ignore-config', '--js-runtimes', 'node',
            '--no-playlist', '--socket-timeout','20', '--retries','2', '--fragment-retries','2',
            '--extractor-retries','1', '--no-progress', '--concurrent-fragments','1']


def fetch_info(url, log_path):
    with open(log_path, 'a') as log:
        result = subprocess.run(ytdlp_base() + ['-f',FORMAT,'--skip-download','--dump-single-json',url],
                                stdout=subprocess.PIPE, stderr=log, text=True, timeout=180, check=True)
    info = json.loads(result.stdout)
    if info.get('live_status') in ('is_live', 'is_upcoming') or not info.get('duration'):
        raise ValueError('Live/upcoming/unknown-length source is not eligible')
    return info


def validate_source_identity(row, info):
    expected=row.get('metadata',{}).get('expected_channel_id')
    if expected and info.get('channel_id') != expected:
        raise ValueError('Resolved channel does not match the selected publisher')


def peak_reservation(info):
    total = 0
    for f in info.get('requested_formats') or [info]:
        exact = f.get('filesize')
        estimate = f.get('filesize_approx')
        if exact:
            total += exact
        elif estimate:
            total += estimate * 1.5
        else:
            total += info['duration'] * max(f.get('tbr') or 0, 4000) * 125 * 1.5
    # Download components coexist with remuxed output, plus captions/metadata.
    return int(2 * total + 16_000_000)


def source_directory(key):
    return ROOT / 'media' / key.replace(':', '_')


def finish_master(db, row, info, directory):
    masters = [p for p in directory.glob('master.*') if p.suffix in ('.mkv','.mp4','.webm','.mov')]
    if len(masters) != 1:
        raise ValueError(f'Expected one final master, found {len(masters)}')
    master = masters[0]
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams',
                                               '-of','json',str(master)], timeout=60))
    verified = validate_probe(probe)
    if abs(verified['duration'] - info['duration']) > max(2, info['duration'] * 0.005):
        raise ValueError('Downloaded duration does not match source')
    with master.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata = row['metadata'] | {'publisher_metadata':summarize_info(info), 'probe':verified,
                                  'retrieved_at':now(), 'scope_review':'metadata_screened; visual_review_pending'}
    db.update(row['source_key'], status='complete', duration=verified['duration'],
              media_path=str(master.relative_to(ROOT)), bytes=master.stat().st_size,
              sha256=digest, metadata=metadata, error=None)
    # Chapters were ingested from publisher metadata before downloading. The
    # rounded publisher duration can exceed the master by a fraction of a second.
    db.reconcile_publisher_chapters(row['source_key'])
    receipt = {'source_key':row['source_key'],'completed_at':now(),'sha256':digest,
               'bytes':master.stat().st_size,'media_path':str(master.relative_to(ROOT)),
               'probe':verified, 'selected_formats':summarize_info(info)['selected_formats']}
    (ROOT/'receipts'/f"{row['source_key'].replace(':','_')}.json").write_text(json.dumps(receipt, indent=2))


def acquire_one(db, row, guard):
    key = row['source_key']
    log = ROOT/'receipts'/f'{key.replace(":","_")}.log'
    db.update(key, status='downloading', attempts=row['attempts'] + 1, error=None)
    db.event(key, 'started', {})
    try:
        info = fetch_info(row['url'], log)
        validate_source_identity(row, info)
        db.update(key, duration=info['duration'], metadata=row['metadata'] | {'publisher_metadata':summarize_info(info)})
        for chapter in info.get('chapters') or []:
            try:
                db.add_clip(key, chapter['start_time'], chapter['end_time'], chapter.get('title') or 'Chapter',
                            origin='publisher_chapter', metadata={'review_state':'unreviewed'})
            except (ValueError, KeyError, TypeError) as exc:
                db.event(key, 'invalid_chapter', {'error':str(exc)})
        guard.check(additional_bytes=peak_reservation(info))
        directory = source_directory(key)
        directory.mkdir(parents=True, exist_ok=True)
        # Reuse the exact resolved format choices; signed URLs are ephemeral and local.
        resolved = directory/'resolved.raw.json'
        resolved.write_text(json.dumps(info))
        command = ytdlp_base() + ['--load-info-json', str(resolved), '-f', FORMAT,
                  '--paths',str(directory), '-o','master.%(ext)s', '--merge-output-format','mkv',
                  '--continue', '--no-overwrites', '--limit-rate','4M', '--write-subs',
                  '--sub-langs','en.*,en,-live_chat', '--sub-format','vtt/best']
        run_guarded(command, cwd=ROOT, log_path=log, guard=guard, timeout_s=2400)
        finish_master(db, db.get(key), info, directory)
        db.event(key, 'complete', {})
        print(json.dumps({'source_key':key,'status':'complete','bytes':db.get(key)['bytes']}), flush=True)
        return True
    except StorageLimit as exc:
        db.update(key, status='paused_storage', error=str(exc), attempts=row['attempts'])
        db.event(key, 'paused_storage', {'reason':str(exc)})
        print(str(exc), flush=True)
        return False
    except (Exception, KeyboardInterrupt) as exc:
        db.update(key, status='retry' if row['attempts'] < 2 else 'failed', error=type(exc).__name__ + ': ' + str(exc)[:500])
        db.event(key, 'failed_attempt', {'error':type(exc).__name__})
        print(json.dumps({'source_key':key,'status':'failed_attempt','error':type(exc).__name__}), flush=True)
        if isinstance(exc, KeyboardInterrupt):
            raise
        return True
    finally:
        db.export(ROOT/'manifests'/'sources.jsonl', ROOT/'manifests'/'subclips.jsonl')


def main():
    def interrupted(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--import-jsonl', type=Path)
    parser.add_argument('--run', type=int, default=0, help='Maximum number of sources attempted in this invocation')
    parser.add_argument('--wait-for-lock', action='store_true', help='Wait for an existing worker to finish before acquiring sources')
    parser.add_argument('--cap-gb', type=float, default=70, help='Media cache cap (user approved 85 GB on 2026-10-04 for gi)')
    args = parser.parse_args()
    db = Catalog(ROOT/'manifests'/'catalog.sqlite')
    if args.import_jsonl:
        for line in args.import_jsonl.read_text().splitlines():
            row = json.loads(line)
            key = db.add(row['url'],row['title'],row['sport'],row['context'],priority=row.get('priority',0),metadata=row.get('metadata',{}))
            if row.get('ready') and db.get(key)['status'] == 'candidate':
                db.update(key, status='ready')
    guard = StorageGuard(ROOT/'media', cap_bytes=int(args.cap_gb*1e9), headroom_bytes=512_000_000)
    # Exclusive OS lock prevents multiple workers from spending the same reservation.
    with open(ROOT/'manifests'/'acquisition.lock','w') as lock:
        if args.wait_for_lock:
            print(json.dumps({'status':'waiting_for_worker_lock'}),flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_lock else fcntl.LOCK_NB))
        print(json.dumps({'status':'worker_lock_acquired'}),flush=True)
        for completed in db.sources(('complete',)):
            db.reconcile_publisher_chapters(completed['source_key'])
        rows = db.sources(('ready','retry','downloading','paused_storage'))[:args.run]
        for row in rows:
            if row['attempts'] >= 3:
                continue
            if not acquire_one(db,row,guard):
                break
            time.sleep(2)
    db.export(ROOT/'manifests'/'sources.jsonl',ROOT/'manifests'/'subclips.jsonl')
    counts = {}
    for row in db.sources():
        counts[row['status']] = counts.get(row['status'],0) + 1
    print(json.dumps({'statuses':counts, 'cache_bytes':guard.used_bytes()}))
    db.close()


if __name__ == '__main__':
    main()
