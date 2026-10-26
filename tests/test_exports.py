import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from export_annotations import build_exports
from test_annotations import record,EVIDENCE


class ExportTests(unittest.TestCase):
    def test_ambiguous_actor_withholds_entire_frame_from_coco(self):
        doc=record();doc['geometry'][0]['training_eligible']=False
        other=copy.deepcopy(doc['geometry'][0]);other['training_eligible']=True
        doc['geometry'].append(other)
        evidence=copy.deepcopy(EVIDENCE);evidence['frames'][0].update(width=100,height=200,image='f1.jpg')
        out=build_exports(doc,evidence)
        self.assertEqual(len(out['geometry']),2)
        self.assertEqual(out['coco']['images'],[])
        self.assertEqual(out['coco']['annotations'],[])

    def test_null_box_is_preserved_but_not_exported_as_coco(self):
        doc=record();doc['geometry'][0]['bbox_xyxy_normalized']=None
        evidence=copy.deepcopy(EVIDENCE)
        evidence['frames'][0].update(width=100,height=200,image='f1.jpg')
        out=build_exports(doc,evidence)
        self.assertIsNone(out['geometry'][0]['bbox_xyxy_normalized'])
        self.assertEqual(out['coco']['images'],[])
        self.assertEqual(out['coco']['annotations'],[])

    def test_only_box_labelled_frames_enter_coco_not_unlabelled_negatives(self):
        evidence=copy.deepcopy(EVIDENCE)
        evidence['frames']=[{'frame_id':'f1','time_s':2,'width':100,'height':200,'image':'evidence/f1.jpg'},
                            {'frame_id':'unlabelled','time_s':8,'width':100,'height':200,'image':'evidence/f2.jpg'}]
        out=build_exports(record(),evidence)
        self.assertEqual(len(out['coco']['images']),1)
        self.assertEqual(out['coco']['annotations'][0]['bbox'],[10,20,30,160])
        self.assertNotIn('keypoints',out['coco']['annotations'][0])

    def test_evidence_kinds_stay_separate_and_rich_claim_fields_survive(self):
        doc=record();doc['claims'][0]['unexpected_detail']={'contact':'occluded'}
        speech=copy.deepcopy(doc['claims'][0]);speech.update(claim_id='spoken',evidence_kind='spoken_statement',frame_ids=[],transcript_segment_ids=[1])
        doc['claims'].append(speech);doc['inspection']['transcript_segment_ids']=[1]
        doc['geometry']=[]
        evidence=copy.deepcopy(EVIDENCE);evidence['frames'][0].update(width=100,height=200,image='f1.jpg')
        out=build_exports(doc,evidence)
        self.assertEqual(len(out['by_evidence']['visual_observation']),1)
        self.assertEqual(len(out['by_evidence']['spoken_statement']),1)
        self.assertEqual(out['claims'][0]['unexpected_detail'],{'contact':'occluded'})
        self.assertEqual(out['by_evidence']['spoken_statement'][0]['frame_refs'],[])


if __name__=='__main__':unittest.main()
