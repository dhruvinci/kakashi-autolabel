import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from validate_annotations import validate


def record():
    return {'source_key':'youtube:abcdefghijk','packet_id':'p1','time_range_s':[0,12],
            'actors':[{'actor_id':'A','description':'red wrist tape'}],
            'claims':[{'claim_id':'c1','time_range_s':[2,4],'actor_ids':['A'],
                       'kind':'configuration','text':'Standing with raised hands.',
                       'evidence_kind':'visual_observation','frame_ids':['f1']}],
            'geometry':[{'frame_id':'f1','actor_id':'A','bbox_xyxy_normalized':[0.1,0.1,0.4,0.9]}],
            'provenance':{'job_id':'j1','model':'gpt-6-astra','effort':'medium'},
            'inspection':{'frame_ids':['f1'],'transcript_segment_ids':[]},
            'novel_label_family':{'free_text':'An unfamiliar but useful detail'}}


EVIDENCE={'source_key':'youtube:abcdefghijk','duration_s':12,
          'frames':[{'frame_id':'f1','time_s':2}],
          'transcript_segments':[{'id':1,'start':2,'end':4}]}


class AnnotationTests(unittest.TestCase):
    def test_annotation_cannot_be_validated_against_a_different_packet(self):
        evidence=EVIDENCE | {'packet_id':'p1','time_range_s':[0,12],
                             'duration_s':120}
        doc=record()
        doc['packet_id']='different-packet'
        self.assertTrue(any('packet_id' in e for e in validate(doc,evidence)))
        doc=record()
        doc['time_range_s']=[0,60]
        self.assertTrue(any('packet.time_range' in e for e in validate(doc,evidence)))

    def test_rich_unknown_fields_survive_validation(self):
        doc=record()
        before=copy.deepcopy(doc)
        self.assertEqual(validate(doc,EVIDENCE),[])
        self.assertEqual(doc,before)

    def test_bad_time_actor_and_geometry_are_reported_without_silent_repair(self):
        doc=record()
        doc['claims'][0]['time_range_s']=[2,13]
        doc['claims'][0]['actor_ids']=['B']
        doc['geometry'][0]['bbox_xyxy_normalized']=[0.5,0.1,0.1,1.1]
        errors=validate(doc,EVIDENCE)
        self.assertTrue(any('time_range' in e for e in errors))
        self.assertTrue(any('actor' in e for e in errors))
        self.assertTrue(any('bbox' in e for e in errors))

    def test_visual_claim_cannot_cite_only_speech_and_spoken_claim_needs_transcript(self):
        doc=record()
        doc['claims'][0]['frame_ids']=[]
        doc['claims'][0]['transcript_segment_ids']=[1]
        self.assertTrue(any('visual evidence' in e for e in validate(doc,EVIDENCE)))
        doc=record()
        doc['claims'][0]['evidence_kind']='spoken_statement'
        self.assertTrue(any('speech evidence' in e for e in validate(doc,EVIDENCE)))

    def test_uninspected_or_nonexistent_evidence_is_rejected(self):
        doc=record()
        doc['inspection']['frame_ids']=[]
        self.assertTrue(any('inspected' in e for e in validate(doc,EVIDENCE)))
        doc=record()
        doc['claims'][0]['frame_ids']=['f999']
        self.assertTrue(any('unknown frame' in e for e in validate(doc,EVIDENCE)))


if __name__=='__main__':
    unittest.main()
