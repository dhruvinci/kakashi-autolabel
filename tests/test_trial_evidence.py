from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from trial_evidence import (build, decode_frames, ranges_and_complement, shift_segments,
                            sha256, transcribe_window, verify_index)


class TrialEvidenceTests(unittest.TestCase):
    def test_exact_partial_complement_and_bounds(self):
        self.assertEqual(ranges_and_complement([[60,80],[10,30]], 100),
                         ([[10,30],[60,80]], [[0,10],[30,60],[80,100]]))
        self.assertEqual(ranges_and_complement([[0,20],[20,40]], 40)[1], [])
        for ranges in ([[0,31],[40,60]], [[0,20],[19,39]], [[-1,19],[30,50]],
                       [[0,20],[90,110]], [[0,20]], [[0,float('nan')],[30,50]],
                       [[False,20],[30,50]]):
            with self.assertRaises(ValueError):
                ranges_and_complement(ranges,100)

    def test_segment_and_word_offsets_and_unique_ids(self):
        raw = [{'id':0, 'start':0.1,'end':1.2,'words':[{'start':0.2,'end':0.8,'word':'hi'}]}]
        shifted = shift_segments(raw, 19.45, 'interval02')
        self.assertEqual(shifted[0]['id'], 'interval02_s0000')
        self.assertAlmostEqual(shifted[0]['start'],19.55)
        self.assertAlmostEqual(shifted[0]['words'][0]['end'],20.25)
        self.assertEqual(raw[0]['start'],0.1)
        self.assertNotEqual(shift_segments(raw,0,'interval01')[0]['id'], shifted[0]['id'])

    @staticmethod
    def fixture(path, offset=0):
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=10',
            '-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-t','4',
            '-vf',f'setpts=PTS+{offset}/TB','-af',f'asetpts=PTS+{offset}/TB',
            '-c:v','ffv1','-c:a','pcm_s16le',str(path)], check=True)

    def test_real_pts_nonzero_range_and_format_origin(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root/'source.mkv'
            # Output timestamp offset preserves all frames and a nonzero origin.
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=10',
                '-t','4','-c:v','ffv1','-output_ts_offset','5',str(path)], check=True)
            rows, _, _ = decode_frames(path, root/'interval01/frames', 1.15, 2.76, 0.5, origin=5)
            self.assertEqual([f['time_s'] for f in rows], [1.2,1.7,2.2,2.7])
            self.assertTrue(all((f['width'],f['height']) == (160,90) for f in rows))
            self.assertTrue(all(f['decoded_pts'] == round(f['time_s']*1000) for f in rows))
            with self.assertRaises(FileExistsError):
                decode_frames(path,root/'interval01/frames',1,2,0.5,5)

    def test_real_audio_window_stays_in_memory_and_shifts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root/'source.mkv'
            self.fixture(path)
            class Model:
                def transcribe(self, audio, **kwargs):
                    self.samples = len(audio)
                    return iter([{'id':0,'start':0.1,'end':0.8,'words':[]}]), SimpleNamespace(language='en')
            model = Model()
            rows, audio = transcribe_window(model,path,1.5,2.5,0,'interval02')
            self.assertEqual(model.samples,16000)
            self.assertEqual(rows[0]['start'],1.6)
            self.assertEqual(audio['audio_window_s'],[1.5,2.5])
            self.assertEqual(list(root.iterdir()),[path])

    def test_frozen_build_master_and_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root/'source.mkv'
            self.fixture(path)
            source = {'source_key':'fixture:test','status':'complete','media_path':str(path),
                      'sha256':sha256(path),'duration':4}
            out = root/'evidence'
            with patch('trial_evidence.shutil.disk_usage',return_value=SimpleNamespace(free=100_000_000_000)):
                receipt = build(source,out,[[0.5,1.5],[2.5,3.5]],no_asr=True)
                with self.assertRaises(FileExistsError):
                    build(source,out,[[0.5,1.5],[2.5,3.5]],no_asr=True)
            index = verify_index(receipt['path'],receipt['sha256'])
            with self.assertRaisesRegex(ValueError,'Index hash mismatch'):
                verify_index(receipt['path'],'bad-hash')
            self.assertEqual(index['unlabelled_complement_s'],[[0,0.5],[1.5,2.5],[3.5,4]])
            self.assertEqual(index['annotated_intervals_s'],[])
            packets = [json.loads(Path(p['path']).read_text()) for p in index['packets']]
            self.assertEqual(packets[0]['transcript_status'],'not_run_visual_only')
            self.assertEqual(len({f['frame_id'] for p in packets for f in p['frames']}),
                             sum(len(p['frames']) for p in packets))
            packet = Path(index['packets'][0]['path'])
            original = packet.read_bytes()
            packet.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'Artifact hash mismatch'):
                verify_index(receipt['path'])
            packet.write_bytes(original)
            frame = Path(packets[0]['frames'][0]['path'])
            frame.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'Artifact hash mismatch'):
                verify_index(receipt['path'])
            path.write_bytes(b'tampered master')
            with self.assertRaisesRegex(ValueError,'Master hash mismatch'):
                verify_index(receipt['path'])

    def test_asr_failure_never_publishes_empty_success(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); path=root/'source.mkv'; self.fixture(path)
            source={'source_key':'fixture:test','media_path':str(path),'sha256':sha256(path),'duration':4}
            fake_whisper=SimpleNamespace(WhisperModel=lambda *a, **kw: object())
            with patch.dict(sys.modules, {'faster_whisper':fake_whisper}), \
                 patch('trial_evidence.shutil.disk_usage',return_value=SimpleNamespace(free=100_000_000_000)), \
                 patch('trial_evidence.transcribe_window',side_effect=RuntimeError('ASR failed')):
                with self.assertRaisesRegex(RuntimeError,'ASR failed'):
                    build(source,root/'out',[[0,1],[2,3]])
            self.assertFalse((root/'out/index.json').exists())

    def test_free_floor_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); path=root/'source.mkv'; self.fixture(path)
            source={'source_key':'fixture:test','media_path':str(path),'sha256':sha256(path),'duration':4}
            with patch('trial_evidence.shutil.disk_usage',return_value=SimpleNamespace(free=30_000_000_000)):
                with self.assertRaisesRegex(RuntimeError,'floor'):
                    build(source,root/'out',[[0,1],[2,3]],no_asr=True)
            self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()
