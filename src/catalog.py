"""Persistent source/interval catalogue and conservative byte-space guards."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from urllib.parse import parse_qs, urlparse, urlunparse


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_source_key(url: str) -> str:
    p = urlparse(url)
    if p.scheme not in ('http', 'https') or not p.hostname:
        raise ValueError('A public HTTP(S) source URL is required')
    host = p.hostname.lower().removeprefix('www.')
    video_id = None
    if host == 'youtu.be':
        video_id = p.path.strip('/').split('/')[0]
    elif host in ('youtube.com', 'm.youtube.com', 'youtube-nocookie.com'):
        if p.path == '/watch':
            video_id = parse_qs(p.query).get('v', [None])[0]
        elif re.match(r'^/(shorts|embed|live)/', p.path):
            video_id = p.path.split('/')[2]
    if video_id:
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', video_id):
            raise ValueError('Malformed YouTube ID')
        return f'youtube:{video_id}'
    clean = urlunparse((p.scheme, p.netloc.lower(), p.path.rstrip('/'), '', p.query, ''))
    return 'url:' + hashlib.sha256(clean.encode()).hexdigest()[:24]


class Catalog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys=ON')
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.executescript('''
          CREATE TABLE IF NOT EXISTS sources (
            source_key TEXT PRIMARY KEY, url TEXT NOT NULL, title TEXT NOT NULL,
            sport TEXT NOT NULL, context TEXT NOT NULL, status TEXT NOT NULL,
            priority INTEGER NOT NULL, duration REAL, media_path TEXT, bytes INTEGER,
            sha256 TEXT, metadata TEXT NOT NULL, error TEXT, attempts INTEGER DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
          );
          CREATE TABLE IF NOT EXISTS clips (
            clip_id TEXT PRIMARY KEY, source_key TEXT NOT NULL REFERENCES sources(source_key),
            start_s REAL NOT NULL, end_s REAL NOT NULL, title TEXT NOT NULL,
            origin TEXT NOT NULL, metadata TEXT NOT NULL
          );
          CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY, source_key TEXT, time TEXT NOT NULL,
            kind TEXT NOT NULL, detail TEXT NOT NULL
          );
        ''')

    def close(self):
        self.conn.close()

    def add(self, url, title, sport, context, *, priority=0, metadata=None):
        if sport not in ('mma', 'nogi', 'gi') or context not in ('competition', 'training', 'tutorial'):
            raise ValueError('Explicit MMA/no-gi and a valid context are required')
        key = canonical_source_key(url)
        if key.startswith('youtube:'):
            url = 'https://www.youtube.com/watch?v=' + key.split(':')[1]
        existing = self.get(key)
        if existing:
            merged = existing['metadata'] | (metadata or {})
            if existing['metadata'].get('split_lock'):
                for field in ('split','split_lock','source_group'):
                    if field in existing['metadata']:
                        merged[field]=existing['metadata'][field]
            self.update(key, title=title, metadata=merged)
            return key
        with self.conn:
            self.conn.execute('''INSERT INTO sources
              (source_key,url,title,sport,context,status,priority,metadata,created_at,updated_at)
              VALUES (?,?,?,?,?,'candidate',?,?,?,?)''',
              (key, url, title, sport, context, priority, json.dumps(metadata or {}), now(), now()))
        return key

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        result = dict(row)
        result['metadata'] = json.loads(result['metadata'])
        return result

    def get(self, key):
        return self._decode(self.conn.execute('SELECT * FROM sources WHERE source_key=?', (key,)).fetchone())

    def sources(self, statuses=None):
        sql = 'SELECT * FROM sources'
        args = []
        if statuses:
            sql += ' WHERE status IN (' + ','.join('?' for _ in statuses) + ')'
            args = list(statuses)
        sql += ' ORDER BY priority DESC, created_at, source_key'
        return [self._decode(r) for r in self.conn.execute(sql, args)]

    def update(self, key, **values):
        allowed = {'title','status','priority','duration','media_path','bytes','sha256','metadata','error','attempts','sport','context'}
        if not values or set(values) - allowed:
            raise ValueError('Unsupported update')
        if 'sport' in values and values['sport'] not in ('mma','nogi','gi'):
            raise ValueError('Off-scope sport')
        if 'context' in values and values['context'] not in ('competition','training','tutorial'):
            raise ValueError('Unknown context')
        if 'metadata' in values:
            values['metadata'] = json.dumps(values['metadata'], ensure_ascii=False)
        values['updated_at'] = now()
        with self.conn:
            cursor = self.conn.execute('UPDATE sources SET ' + ','.join(f'{k}=?' for k in values) + ' WHERE source_key=?',
                                      (*values.values(), key))
            if cursor.rowcount != 1:
                raise ValueError('Unknown source')

    def event(self, key, kind, detail):
        with self.conn:
            self.conn.execute('INSERT INTO events(source_key,time,kind,detail) VALUES(?,?,?,?)',
                              (key, now(), kind, json.dumps(detail, ensure_ascii=False)))

    def add_clip(self, key, start, end, title, *, origin, metadata=None):
        src = self.get(key)
        if (not src or src['duration'] is None or not all(math.isfinite(v) for v in (start, end))
                or not 0 <= start < end <= src['duration']):
            raise ValueError('Subclip must be within a known source duration')
        clip_id = hashlib.sha256(f'{key}|{start:.6f}|{end:.6f}|{origin}'.encode()).hexdigest()[:24]
        with self.conn:
            self.conn.execute('INSERT OR IGNORE INTO clips VALUES(?,?,?,?,?,?,?)',
                              (clip_id, key, start, end, title, origin, json.dumps(metadata or {})))
        return clip_id

    def clips(self, key=None):
        sql = 'SELECT * FROM clips' + (' WHERE source_key=?' if key else '') + ' ORDER BY source_key,start_s'
        return [self._decode(r) for r in self.conn.execute(sql, (key,) if key else ())]

    def reconcile_publisher_chapters(self,key):
        source=self.get(key)
        if not source or source['duration'] is None:
            raise ValueError('Known master duration required')
        with self.conn:
            for clip in self.clips(key):
                if clip['origin']!='publisher_chapter' or clip['end_s']<=source['duration']:
                    continue
                if clip['start_s']>=source['duration']:
                    self.conn.execute('INSERT INTO events(source_key,time,kind,detail) VALUES(?,?,?,?)',
                                      (key,now(),'publisher_chapter_outside_master',json.dumps({'original_clip':clip})))
                    self.conn.execute('DELETE FROM clips WHERE clip_id=?',(clip['clip_id'],))
                else:
                    metadata=clip['metadata']
                    metadata.setdefault('publisher_interval_s',[clip['start_s'],clip['end_s']])
                    metadata['endpoint_adjustment']='clamped_to_verified_master_duration'
                    self.conn.execute('UPDATE clips SET end_s=?,metadata=? WHERE clip_id=?',
                                      (source['duration'],json.dumps(metadata),clip['clip_id']))

    def export(self, sources_path, clips_path):
        for path, rows in ((sources_path, self.sources()), (clips_path, self.clips())):
            path = Path(path)
            # Independent workers may export while the parent adds evidence clips.
            # Each writer owns its temporary file; readers see one complete snapshot.
            with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
                stream.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
                temp=Path(stream.name)
            temp.replace(path)


class StorageLimit(RuntimeError):
    pass


class StorageGuard:
    def __init__(self, root, cap_bytes=70_000_000_000, reserve_bytes=30_000_000_000,
                 headroom_bytes=64_000_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cap_bytes, self.reserve_bytes, self.headroom_bytes = cap_bytes, reserve_bytes, headroom_bytes

    def used_bytes(self):
        total = 0
        for p in self.root.rglob('*'):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except FileNotFoundError:
                pass  # Downloader may rename a part file during this snapshot.
        return total

    def check(self, additional_bytes=0, *, free_bytes=None):
        if additional_bytes < 0:
            raise ValueError('Negative reservation')
        used = self.used_bytes()
        free = shutil.disk_usage(self.root).free if free_bytes is None else free_bytes
        if used + additional_bytes + self.headroom_bytes > self.cap_bytes:
            raise StorageLimit(f'Cache capacity pause: {used} used, {additional_bytes} reserved')
        if free - additional_bytes - self.headroom_bytes < self.reserve_bytes:
            raise StorageLimit(f'Free-space pause: {free} free, {additional_bytes} reserved')
        return {'used_bytes': used, 'free_bytes': free, 'reserved_bytes': additional_bytes}
