import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from catalog import Catalog, StorageGuard, StorageLimit, canonical_source_key


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = Catalog(self.root / 'catalog.sqlite')

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_youtube_aliases_and_time_offsets_do_not_create_duplicate_masters(self):
        a = self.db.add('https://youtu.be/abcdefghijk?t=30', 'A', 'mma', 'competition')
        b = self.db.add('https://www.youtube.com/watch?v=abcdefghijk&list=foo', 'B', 'mma', 'competition')
        self.assertEqual(a, b)
        self.assertEqual(len(self.db.sources()), 1)
        self.assertEqual(canonical_source_key('https://youtube.com/shorts/abcdefghijk'), 'youtube:abcdefghijk')

    def test_unknown_domain_cannot_be_silently_accepted_into_scope(self):
        for sport in ['judo', '', 'kickboxing']:
            with self.assertRaises(ValueError):
                self.db.add('https://example.com/video', 'bad', sport, 'training')

    def test_visual_review_can_correct_context_without_permitting_off_scope_sports(self):
        key=self.db.add('https://youtu.be/abcdefghijk','A','mma','training')
        self.db.update(key,context='tutorial')
        self.assertEqual(self.db.get(key)['context'],'tutorial')
        with self.assertRaises(ValueError): self.db.update(key,sport='judo')
        self.assertEqual(self.db.get(key)['sport'],'mma')

    def test_metadata_updates_preserve_completion_after_reopen(self):
        key = self.db.add('https://youtu.be/abcdefghijk', 'A', 'nogi', 'training')
        self.db.update(key, status='complete', media_path='media/a.webm', sha256='123', bytes=100)
        self.db.close()
        self.db = Catalog(self.root / 'catalog.sqlite')
        self.db.add('https://youtu.be/abcdefghijk', 'Updated title', 'nogi', 'training')
        self.assertEqual(self.db.get(key)['status'], 'complete')
        self.assertEqual(self.db.get(key)['sha256'], '123')

    def test_subclips_require_valid_source_time_and_create_no_media_files(self):
        key = self.db.add('https://youtu.be/abcdefghijk', 'A', 'nogi', 'tutorial')
        self.db.update(key, duration=100)
        self.db.add_clip(key, 10, 20, 'Guard entry', origin='publisher_chapter')
        self.db.add_clip(key, 10, 20, 'Guard entry', origin='publisher_chapter')
        self.assertEqual(len(self.db.clips(key)), 1)
        self.assertEqual(self.db.clips(key)[0]['end_s'], 20)
        for start, end in [(-1, 5), (20, 10), (0, 101), (float('nan'), 20)]:
            with self.assertRaises(ValueError):
                self.db.add_clip(key, start, end, 'bad', origin='publisher_chapter')
        with self.assertRaises(ValueError):
            self.db.add_clip('missing', 1, 2, 'bad', origin='publisher_chapter')
        self.assertEqual(list(self.root.glob('**/*.mp4')), [])

    def test_export_preserves_rich_metadata_without_treating_candidates_as_complete(self):
        key = self.db.add('https://youtu.be/abcdefghijk', 'A', 'nogi', 'tutorial', metadata={'unfamiliar': {'x': 1}})
        self.db.export(self.root / 'sources.jsonl', self.root / 'clips.jsonl')
        row = json.loads((self.root / 'sources.jsonl').read_text())
        self.assertEqual(row['status'], 'candidate')
        self.assertEqual(row['metadata']['unfamiliar'], {'x': 1})

    def test_bulk_reimport_cannot_remove_a_locked_development_split(self):
        url='https://youtu.be/abcdefghijk'
        key=self.db.add(url,'pilot','mma','competition',metadata={'split':'dev','split_lock':'pilot'})
        self.db.add(url,'rediscovered','mma','competition',metadata={'split':'unassigned','split_lock':None})
        self.assertEqual(self.db.get(key)['metadata']['split'],'dev')
        self.assertEqual(self.db.get(key)['metadata']['split_lock'],'pilot')

    def test_publisher_rounding_is_reconciled_to_verified_duration_without_losing_original(self):
        key=self.db.add('https://youtu.be/abcdefghijk','A','nogi','tutorial')
        self.db.update(key,duration=10)
        self.db.add_clip(key,5,10,'Final chapter',origin='publisher_chapter',metadata={'extra':'preserve'})
        self.db.update(key,duration=9.7)
        self.db.reconcile_publisher_chapters(key)
        clips=self.db.clips(key)
        self.assertEqual(clips[0]['end_s'],9.7)
        self.assertEqual(clips[0]['metadata']['publisher_interval_s'],[5,10])
        self.assertEqual(clips[0]['metadata']['extra'],'preserve')
        self.db.reconcile_publisher_chapters(key)
        self.assertEqual(self.db.clips(key)[0]['metadata']['publisher_interval_s'],[5,10])

    def test_chapter_wholly_after_master_is_archived_as_invalid_not_made_into_zero_length_clip(self):
        key=self.db.add('https://youtu.be/abcdefghijk','A','nogi','tutorial')
        self.db.update(key,duration=10)
        self.db.add_clip(key,9.8,10,'Impossible chapter',origin='publisher_chapter')
        self.db.update(key,duration=9.7)
        self.db.reconcile_publisher_chapters(key)
        self.assertEqual(self.db.clips(key),[])
        event=self.db.conn.execute("SELECT detail FROM events WHERE kind='publisher_chapter_outside_master'").fetchone()
        self.assertEqual(json.loads(event[0])['original_clip']['start_s'],9.8)


class GuardTests(unittest.TestCase):
    def test_partial_and_merge_files_consume_capacity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'x.webm.part').write_bytes(b'0' * 60)
            (root / 'x.temp.webm').write_bytes(b'0' * 30)
            guard = StorageGuard(root, cap_bytes=100, reserve_bytes=10, headroom_bytes=0)
            self.assertEqual(guard.used_bytes(), 90)
            with self.assertRaises(StorageLimit):
                guard.check(additional_bytes=11, free_bytes=1000)
            guard.check(additional_bytes=10, free_bytes=1000)

    def test_reservation_accounts_for_download_and_merge_peak_and_free_floor(self):
        with tempfile.TemporaryDirectory() as td:
            guard = StorageGuard(Path(td), cap_bytes=1000, reserve_bytes=300, headroom_bytes=20)
            with self.assertRaises(StorageLimit):
                guard.check(additional_bytes=100, free_bytes=419)
            guard.check(additional_bytes=100, free_bytes=420)
            with self.assertRaises(StorageLimit):
                guard.check(additional_bytes=981, free_bytes=10000)


if __name__ == '__main__':
    unittest.main()
