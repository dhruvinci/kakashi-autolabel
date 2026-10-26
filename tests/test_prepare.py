from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare import extract_frames, make_sheets


class PreparationTests(unittest.TestCase):
    def test_evidence_times_match_decoded_source_not_arbitrary_image_number(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root/'fixture.mkv'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=10',
                            '-t','2','-c:v','ffv1',str(video)],check=True)
            frames = extract_frames(video,root/'frames',interval_s=0.5)
            self.assertEqual([r['time_s'] for r in frames], [0.0,0.5,1.0,1.5])
            self.assertEqual(len({r['sha256'] for r in frames}),4)
            self.assertTrue(all(Path(r['path']).exists() for r in frames))
            sheets = make_sheets(frames,root/'panels',per_sheet=3)
            self.assertEqual([len(s['frame_ids']) for s in sheets],[3,1])
            self.assertEqual(sheets[1]['frame_ids'],['f000004'])
            with self.assertRaises(FileExistsError):
                extract_frames(video,root/'frames',interval_s=0.5)


if __name__ == '__main__':
    unittest.main()
