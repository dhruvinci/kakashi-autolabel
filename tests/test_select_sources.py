from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from select_sources import screen

# A made-up config: the real channel lists are not published (see src/config/selection-config.example.json).
CFG={'mma_channels':{'MMA_CHANNEL':'an MMA promotion'},'nogi_channels':{'NOGI_GYM':'a no-gi gym'},
     'training_channels':['NOGI_GYM'],'grappling_channels':['MIXED_PROMOTION'],
     'nogi_title_terms':[],'other_scope_terms':[],'excluded_title_patterns':[r'striker a vs grappler b']}


def candidate(title,channel_id='other',context='tutorial',duration=300):
    return {'id':'abcdefghijk','url':'https://www.youtube.com/watch?v=abcdefghijk',
            'title':title,'duration':duration,'channel_id':channel_id,'channel':'Test',
            'discovery_evidence':[{'sport':'nogi','context':context,'query':'search'}]}


class SelectionTests(unittest.TestCase):
    def test_search_query_alone_does_not_establish_sport(self):
        self.assertIsNone(screen(candidate('Guard Passing Concepts'),CFG)[0])

    def test_mixed_gi_video_and_standalone_kickboxing_are_excluded(self):
        for title in ['20 Rolling Tips for BOTH Gi and Nogi BJJ','Gi vs No Gi Rolling',
                      'Muay Thai Full Fight','MMA fighter boxing sparring','No Gi interview podcast']:
            self.assertIsNone(screen(candidate(title),CFG)[0],title)

    def test_source_title_can_correct_search_context(self):
        row,reason=screen(candidate('No Gi Rolling Rounds',context='tutorial'),CFG)
        self.assertEqual(row['sport'],'nogi')
        self.assertEqual(row['context'],'training')
        row,reason=screen(candidate('MMA Wall Wrestling Tutorial',context='training'),CFG)
        self.assertEqual(row['sport'],'mma')
        self.assertEqual(row['context'],'tutorial')

    def test_official_mma_match_and_known_mixed_rules_are_distinguished(self):
        row,_=screen(candidate('Fighter A vs Fighter B | Full Fight','MMA_CHANNEL','competition'),CFG)
        self.assertEqual(row['sport'],'mma')
        row,_=screen(candidate('Striker A vs Grappler B | Full Fight','MIXED_PROMOTION','competition'),CFG)
        self.assertIsNone(row)

    def test_gym_training_is_provisional_not_visual_verification(self):
        row,_=screen(candidate('Athlete A vs Athlete B | Full Training Round','NOGI_GYM','training'),CFG)
        self.assertEqual(row['sport'],'nogi')
        self.assertEqual(row['metadata']['scope_review'],'metadata_screened; visual_review_pending')

    def test_unknown_duration_and_excessively_long_video_do_not_enter_download_queue(self):
        for duration in [None,0,10000]:
            self.assertIsNone(screen(candidate('MMA Full Fight',duration=duration),CFG)[0])


if __name__=='__main__':
    unittest.main()
