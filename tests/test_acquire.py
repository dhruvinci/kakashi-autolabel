from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from acquire import run_guarded, summarize_info, validate_probe, validate_source_identity
from catalog import StorageGuard, StorageLimit


class AcquireTests(unittest.TestCase):
    def test_resolved_media_from_wrong_publisher_is_not_downloaded(self):
        row={'metadata':{'expected_channel_id':'official'}}
        with self.assertRaises(ValueError):
            validate_source_identity(row,{'channel_id':'other'})
        validate_source_identity(row,{'channel_id':'official'})

    def test_media_without_audio_or_over_resolution_cannot_be_marked_complete(self):
        good = {'format': {'duration': '12.5'}, 'streams': [
            {'codec_type':'video', 'width':1280, 'height':720, 'codec_name':'vp9', 'avg_frame_rate':'30/1'},
            {'codec_type':'audio', 'codec_name':'opus'}]}
        self.assertEqual(validate_probe(good)['duration'], 12.5)
        with self.assertRaises(ValueError):
            validate_probe({'format':good['format'], 'streams':good['streams'][:1]})
        with self.assertRaises(ValueError):
            validate_probe({'format':good['format'], 'streams':[
                {'codec_type':'video','width':1920,'height':1080},good['streams'][1]]})

    def test_sanitization_retains_provenance_and_chapters_but_not_signed_media_urls(self):
        info = {'id':'abc','title':'A','duration':12,'webpage_url':'https://example.com/watch',
                'url':'https://cdn.example.com/private-signature', 'chapters':[{'start_time':0,'end_time':12,'title':'A'}],
                'requested_formats':[{'format_id':'247','height':720,'filesize':100,'url':'secret'}]}
        row = summarize_info(info)
        self.assertEqual(row['chapters'][0]['end_time'], 12)
        self.assertEqual(row['selected_formats'][0]['filesize'], 100)
        self.assertNotIn('url', row['selected_formats'][0])
        self.assertNotIn('url', row)

    def test_watchdog_terminates_process_on_space_limit_without_deleting_partial(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            guard = StorageGuard(root, cap_bytes=5000, reserve_bytes=0, headroom_bytes=0)
            cmd = [sys.executable, '-c',
                   'import pathlib,time; pathlib.Path("partial.part").write_bytes(b"x"*6000); time.sleep(5)']
            with self.assertRaises(StorageLimit):
                run_guarded(cmd, cwd=root, log_path=root/'worker.log', guard=guard, timeout_s=10)
            self.assertTrue((root/'partial.part').exists())

    def test_timeout_stops_child_and_returns_an_explicit_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(subprocess.TimeoutExpired):
                run_guarded([sys.executable, '-c', 'import time; time.sleep(5)'], cwd=root,
                            log_path=root/'worker.log', guard=StorageGuard(root, reserve_bytes=0), timeout_s=0.2)


if __name__ == '__main__':
    unittest.main()
