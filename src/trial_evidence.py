"""Frozen local evidence for selected-interval trials (no downloads or model APIs).

Ranges JSON: [[start_s,end_s],[start_s,end_s]], half-open source-relative seconds.
Use two ~20s intervals, each at most 30s. Context is evidence, not coverage.
Only a completed index.json is consumable; interrupted directories are never reused.
"""
import argparse
from dataclasses import asdict
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess

from catalog import now
from evidence_integrity import absolute, sha256, verify_frames, checked_json
from prepare import make_sheets

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 30_000_000_000


def ranges_and_complement(ranges, duration):
    if not math.isfinite(duration) or duration <= 0 or not isinstance(ranges, list) or len(ranges) != 2:
        raise ValueError('Two selected intervals and positive finite duration required')
    clean = []
    for pair in ranges:
        if (not isinstance(pair, list) or len(pair) != 2 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in pair)):
            raise ValueError('Ranges must be finite [start,end] pairs')
        start, end = pair
        if not 0 <= start < end <= duration or end-start > 30:
            raise ValueError('Intervals must be within master and at most 30 seconds')
        clean.append([start, end])
    clean.sort()
    cursor = 0
    gaps = []
    for start, end in clean:
        if start < cursor:
            raise ValueError('Selected intervals overlap')
        if start > cursor:
            gaps.append([cursor, start])
        cursor = end
    if cursor < duration:
        gaps.append([cursor, duration])
    return clean, gaps


def source_record(key, catalog=ROOT/'manifests/catalog.sqlite'):
    # Do not initialize/mutate the production catalogue just to read it.
    with sqlite3.connect(Path(catalog).resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        row = db.execute('SELECT * FROM sources WHERE source_key=?', (key,)).fetchone()
    if row is None or row['status'] != 'complete':
        raise ValueError('Already downloaded complete catalog source required')
    source = dict(row)
    source['metadata'] = json.loads(source['metadata'])
    source['media_path'] = str(absolute(source['media_path']))
    if sha256(source['media_path']) != source['sha256']:
        raise ValueError('Master hash mismatch')
    return source


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format',
        '-show_streams', '-of', 'json', str(path)], timeout=60))


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path)}


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    return ref(path)


def decode_frames(video, dest, start, end, step, origin=0):
    """Decode from the beginning; no seek offset or inferred frame numbering times."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    expression = (f"setpts=PTS-({origin:.9f})/TB,select='gte(t,{start:.9f})*lt(t,{end:.9f})*"
                  f"(isnan(prev_selected_t)+gte(t-prev_selected_t,{step:.9f}))',showinfo")
    command = ['ffmpeg', '-hide_banner', '-nostdin', '-threads', '2', '-copyts', '-i', str(video),
        '-map', '0:v:0', '-vf', expression, '-fps_mode', 'vfr', '-q:v', '3', '-threads', '2',
        str(dest/'%06d.jpg')]
    log = dest.parent/(dest.name+'-decode.log')
    with log.open('x') as stream:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=stream, check=True, timeout=1800)
    text = log.read_text()
    timebase = re.search(r'config in time_base: (\d+/\d+)', text)
    pts = [int(p) for p in re.findall(r'\bn:\s*\d+\s+pts:\s*(-?\d+)', text)]
    images = sorted(dest.glob('*.jpg'))
    if not timebase or not pts or len(pts) != len(images):
        raise ValueError('Decoded PTS/image count mismatch or empty evidence')
    from PIL import Image
    frames = []
    for i, (path, value) in enumerate(zip(images, pts), 1):
        with Image.open(path) as im:
            width, height = im.size
        frames.append({'frame_id': f'{dest.parent.name}_f{i:06d}',
            'time_s': float(value*Fraction(timebase[1])), 'decoded_pts': value,
            'time_base': timebase[1], 'width': width, 'height': height, **ref(path)})
    if any(not start-1e-6 <= f['time_s'] < end+1e-6 for f in frames):
        raise ValueError('PTS outside requested evidence bounds')
    return frames, ref(log), command


def shift_segments(segments, offset, prefix):
    result = []
    for i, segment in enumerate(segments):
        row = asdict(segment) if not isinstance(segment, dict) else json.loads(json.dumps(segment))
        row['local_segment_id'] = row.get('id')
        row['id'] = f'{prefix}_s{i:04d}'
        for field in ('start', 'end'):
            row[field] += offset
        for word in row.get('words') or []:
            for field in ('start', 'end'):
                word[field] += offset
        row['evidence_kind'] = 'spoken_asr_unverified'
        result.append(row)
    return result


def transcribe_window(model, video, start, end, origin, prefix):
    import numpy as np
    # Keep format-relative A/V offsets, including leading silence. No audio file.
    filters = (f'asetpts=PTS-({origin:.9f})/TB,aresample=16000:async=1:first_pts=0,'
               f'atrim=start={start:.9f}:end={end:.9f},asetpts=PTS-STARTPTS')
    pcm = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2',
        '-copyts', '-i', str(video), '-map', '0:a:0', '-af', filters, '-ac', '1',
        '-ar', '16000', '-f', 'f32le', 'pipe:1'], timeout=1800)
    audio = np.frombuffer(pcm, dtype=np.float32)
    if not len(audio):
        raise ValueError('Audio decode yielded no samples; use explicit --no-asr for visual-only evidence')
    segments, info = model.transcribe(audio, language='en', beam_size=5, word_timestamps=True,
        vad_filter=True, condition_on_previous_text=False)
    return shift_segments(segments, start, prefix), {'decoded_samples': len(audio),
        'sample_rate': 16000, 'language': info.language, 'audio_window_s': [start, end]}


def verify_index(path, expected_sha256=None):
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError('Index hash mismatch')
    index = json.loads(Path(path).read_text())
    if sha256(index['master_path']) != index['source_sha256']:
        raise ValueError('Master hash mismatch')
    for artifact in index['artifacts']:
        if sha256(artifact['path']) != artifact['sha256']:
            raise ValueError('Artifact hash mismatch: '+artifact['path'])
    for item in index.get('packets', []):
        packet = checked_json(item)
        if packet['source_sha256'] != index['source_sha256'] or packet['source_key'] != index['source_key']:
            raise ValueError('Packet source binding mismatch')
        verify_frames(packet['frames'])
    return index


def build(source, output, ranges=None, no_asr=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new version; frozen or interrupted output must not be overwritten')
    if sha256(source['media_path']) != source['sha256']:
        raise ValueError('Master hash mismatch')
    metadata = probe(source['media_path'])
    duration = float(source['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Known positive source duration required')
    actual_duration = float(metadata['format']['duration'])
    if abs(duration-actual_duration) > 0.1:
        raise ValueError('Catalog/probe duration mismatch')
    origin = float(metadata['format'].get('start_time', 0))
    video = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
    selected, gaps = ranges_and_complement(ranges, duration) if ranges is not None else ([], [[0, duration]])
    windows = [(max(0, a-0.55), min(duration, b+0.55)) for a, b in selected] or [(0, duration)]
    steps = [0.5]*len(windows) if selected else [duration/12]
    frame_bound = sum(math.ceil((b-a)/step)+1 for (a,b), step in zip(windows, steps))
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    reserve = frame_bound*(video['width']*video['height']*8+2_000_000)+64_000_000
    if shutil.disk_usage(parent).free-reserve < FLOOR:
        raise RuntimeError('Evidence would encroach on 30GB free-space floor')
    model = None
    if selected and not no_asr:
        if not any(s['codec_type'] == 'audio' for s in metadata['streams']):
            raise ValueError('Source has no audio; explicitly choose --no-asr')
        from faster_whisper import WhisperModel
        model = WhisperModel('small.en', device='cpu', compute_type='int8', cpu_threads=4,
            num_workers=1, download_root=str(ROOT/'models/whisper'), local_files_only=True)
    output.mkdir(parents=True)
    index = {'version': 1, 'created_at': now(), 'source_key': source['source_key'],
        'source_sha256': source['sha256'], 'master_path': source['media_path'],
        'duration_s': duration, 'selected_intervals_s': selected, 'unlabelled_complement_s': gaps,
        'annotation_state': 'evidence_only_no_labels', 'annotated_intervals_s': [],
        'coverage_note': 'Selected intervals are annotation targets, not completed annotations. Context and overview do not count as coverage.',
        'timebase': 'Decoded integer PTS times stream time_base after subtracting format.start_time; no seeking.',
        'format_start_time_s': origin, 'packets': []}
    write_json(output/'source.json', {'catalog_source': source, 'ffprobe': metadata})
    all_frames, all_segments = [], []
    for i, ((start, end), step) in enumerate(zip(windows, steps), 1):
        pid = f'interval{i:02d}' if selected else 'overview'
        folder = output/pid
        frames, log, command = decode_frames(source['media_path'], folder/'frames', start, end, step, origin)
        panels = make_sheets(frames, folder/'sheets')
        core = selected[i-1] if selected else [0, duration]
        if not any(core[0] <= f['time_s'] < core[1] for f in frames):
            raise ValueError('No core evidence')
        segments, audio = transcribe_window(model, source['media_path'], start, end, origin, pid) if model else ([], None)
        all_frames.extend(frames)
        all_segments.extend(segments)
        packet = {'packet_id': pid, 'source_key': source['source_key'], 'source_sha256': source['sha256'],
            'duration_s': duration, 'time_range_s': core, 'evidence_range_s': [start, end],
            'frames': frames, 'panels': panels, 'decode_log': log, 'decode_command': command,
            'transcript_segments': segments, 'audio_decode': audio,
            'transcript_status': 'unverified_machine_transcript' if model else 'not_run_visual_only',
            'transcript_model': 'small.en' if model else None,
            'transcript_warning': 'English-only model. Non-English speech, names, technical terms, speaker identity and timing may be wrong. Empty successful ASR does not prove absence of speech.',
            'actor_ledger_seed': [], 'timebase': index['timebase'], 'format_start_time_s': origin,
            'notes': 'Full-resolution JPEG samples, nominal 0.5s cadence in selected intervals; stills are not continuous video inspection. Context is not annotation coverage.'}
        index['packets'].append(write_json(folder/'packet.json', packet) | {'packet_id': pid, 'time_range_s': core})
    if sha256(source['media_path']) != source['sha256']:
        raise ValueError('Master changed while decoding')
    if shutil.disk_usage(output).free < FLOOR:
        raise RuntimeError('30GB free-space floor breached; no completed index produced')
    index['frames_manifest'] = write_json(output/'frames.json', all_frames)
    index['transcript'] = write_json(output/'transcript.json', {
        'source_key': source['source_key'], 'source_sha256': source['sha256'],
        'duration_s': duration, 'model': 'small.en' if model else None,
        'status': 'unverified_machine_transcript' if model else 'not_run_visual_only',
        'warning': 'English-only model; names, non-English speech and alignment may be wrong.',
        'segments': all_segments})
    index['artifacts'] = [ref(p) for p in sorted(output.rglob('*')) if p.is_file()]
    write_json(output/'index.json', index)
    verify_index(output/'index.json')
    return ref(output/'index.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('overview', 'prepare'):
        child = sub.add_parser(name)
        child.add_argument('--source-key', required=True)
        child.add_argument('--output', type=Path, required=True)
        if name == 'prepare':
            child.add_argument('--ranges-json', type=Path, required=True)
            child.add_argument('--no-asr', action='store_true')
    args = parser.parse_args()
    ranges = json.loads(args.ranges_json.read_text()) if args.command == 'prepare' else None
    print(json.dumps(build(source_record(args.source_key), args.output, ranges, getattr(args, 'no_asr', False))))


if __name__ == '__main__':
    main()
