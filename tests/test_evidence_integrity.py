import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evidence_integrity import load_verified_evidence


class EvidenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.master=self.root/'master.mkv';self.master.write_bytes(b'master')
        self.image=self.root/'frame.jpg';self.image.write_bytes(b'frame')
        self.source={'source_key':'test:source','sha256':self.sha(self.master),'media_path':str(self.master),'status':'complete'}
        self.frames=self.root/'frames.json'
        self.frames.write_text(json.dumps([{'frame_id':'f1','time_s':0,'path':str(self.image),'sha256':self.sha(self.image)}]))
        self.transcript=self.root/'transcript.json'
        self.transcript.write_text(json.dumps({'source_sha256':self.source['sha256'],'duration_s':9.98,'segments':[]}))
        self.receipt=self.root/'bindings.json'
        self.binding={'source_key':self.source['source_key'],'source_sha256':self.source['sha256'],
                      'master_path':str(self.master),'manifests':[self.ref(self.frames)],'transcript':self.ref(self.transcript)}
        self.save()

    def sha(self,path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def ref(self,path):return {'path':str(path),'sha256':self.sha(path)}
    def save(self):self.receipt.write_text(json.dumps(self.binding))
    def load(self):return load_verified_evidence(self.source,self.frames,self.receipt)

    def test_valid_evidence_preserves_records_and_independent_asr_duration(self):
        frames,transcript=self.load()
        self.assertEqual(frames[0]['path'],str(self.image))
        self.assertEqual(transcript['duration_s'],9.98)

    def test_arbitrary_manifest_with_same_frame_ids_is_rejected(self):
        other=self.root/'other.json';other.write_bytes(self.frames.read_bytes())
        with self.assertRaisesRegex(ValueError,'registered manifest'):
            load_verified_evidence(self.source,other,self.receipt)

    def test_mutated_manifest_image_master_and_transcript_are_rejected(self):
        for path in [self.frames,self.image,self.master,self.transcript]:
            with self.subTest(path=path.name):
                original=path.read_bytes();path.write_bytes(original+b' ')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):self.load()
                path.write_bytes(original)

    def test_wrong_source_and_transcript_source_are_rejected(self):
        self.source['source_key']='another:source'
        with self.assertRaisesRegex(ValueError,'Source binding'):self.load()
        self.source['source_key']=self.binding['source_key']
        self.transcript.write_text(json.dumps({'source_sha256':'different','segments':[]}))
        self.binding['transcript']=self.ref(self.transcript);self.save()
        with self.assertRaisesRegex(ValueError,'transcript'):self.load()
