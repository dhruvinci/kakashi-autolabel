from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from transcribe import decode_audio


class SpeechPreparationTests(unittest.TestCase):
    def test_ffmpeg_audio_decoder_produces_16khz_mono_without_a_second_audio_file(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'stereo.wav'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:sample_rate=48000',
                            '-t','1','-ac','2',str(path)],check=True)
            audio=decode_audio(path)
            self.assertEqual(audio.shape,(16000,))
            self.assertGreater(float(abs(audio).max()),0.05)
            self.assertEqual(len(list(Path(td).iterdir())),1)


if __name__=='__main__':
    unittest.main()
