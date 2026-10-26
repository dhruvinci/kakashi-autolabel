import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from autolabel_astra import keyframe_indices, mark_detections, schema
from autolabel_common import TAXONOMY, box_iou, mask_box, rle_decode, rle_encode
from autolabel_fuse import clip_box, exclusive, frame_points, sheet_to_frame


def rect(shape, x1, y1, x2, y2):
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


class RleTests(unittest.TestCase):
    def test_round_trip_and_coco_column_major_order(self):
        mask = rect((5, 7), 1, 2, 4, 5)
        mask[0, 6] = True
        rle = rle_encode(mask)
        self.assertEqual(rle['size'], [5, 7])
        self.assertEqual(sum(rle['counts']), 35)
        self.assertTrue((rle_decode(rle) == mask).all())
        self.assertEqual(rle_encode(np.ones((2, 2), bool))['counts'], [0, 4])

    def test_mask_box_is_exclusive_xyxy(self):
        self.assertEqual(mask_box(rect((10, 10), 2, 3, 5, 9)), [2, 3, 5, 9])
        self.assertIsNone(mask_box(np.zeros((4, 4), bool)))
        self.assertAlmostEqual(box_iou([0, 0, 2, 2], [1, 1, 3, 3]), 1/7)


class MarkTests(unittest.TestCase):
    def test_mask_nms_drops_duplicates_and_contained_fragments(self):
        big = rect((40, 40), 0, 0, 20, 40)
        frame = {'detections': [{'mask_rle': rle_encode(m)} for m in
                                (big, rect((40, 40), 0, 0, 20, 38), rect((40, 40), 2, 2, 6, 6), rect((40, 40), 25, 0, 40, 40))]}
        kept, masks = mark_detections(frame)
        self.assertEqual(kept, [0, 3])
        self.assertEqual(len(masks), 2)

    def test_keyframes_span_first_and_last_frame(self):
        self.assertEqual(keyframe_indices(41), [0, 8, 16, 24, 32, 40])
        self.assertEqual(keyframe_indices(43)[-1], 42)


class FuseGeometryTests(unittest.TestCase):
    sheet = {'frames': ['F01', 'F02', 'F03', 'F04'], 'scale': 0.5,
             'tile_origins_xy': [[0, 0], [640, 0], [0, 360], [640, 360]]}

    def test_sheet_coordinates_map_back_to_full_frame(self):
        self.assertEqual(sheet_to_frame([650, 370, 700, 400], self.sheet, 'F04'), [20, 20, 120, 80])
        self.assertEqual(frame_points([[650, 370], [10, 10]], 1280, 720, self.sheet, 'F04'), [[20, 20]])

    def test_clip_box_orders_corners_and_rejects_slivers(self):
        self.assertEqual(clip_box([50, 60, -5, 10], 100, 100), [0, 10, 50, 60])
        self.assertIsNone(clip_box([5, 5, 7, 50], 100, 100))

    def test_exclusive_prefers_clean_detection_over_box_mask(self):
        shape = (10, 10)
        det = {'source': 'sam31_text_detection', 'score': 0.9, 'mask': rect(shape, 0, 0, 6, 10)}
        box = {'source': 'astra_box_to_sam31_mask', 'score': 0.0, 'mask': rect(shape, 4, 0, 10, 10)}
        exclusive([box, det], shape)
        self.assertFalse((box['mask'] & det['mask']).any())
        self.assertEqual(int(det['mask'].sum()), 60)
        self.assertEqual(box['pixels_removed_by_overlap'], 20)

    def test_contaminated_detection_loses_to_box_mask(self):
        shape = (10, 10)
        det = {'source': 'sam31_text_detection', 'score': 0.9, 'mask': rect(shape, 0, 0, 6, 10),
               'contains_other_actor_point': True}
        box = {'source': 'astra_box_to_sam31_mask', 'score': 0.0, 'mask': rect(shape, 4, 0, 10, 10)}
        exclusive([det, box], shape)
        self.assertEqual(int(box['mask'].sum()), 60)


class SchemaTests(unittest.TestCase):
    def test_strict_schema_requires_every_property(self):
        def walk(node):
            if isinstance(node, dict):
                if node.get('type') == 'object':
                    self.assertFalse(node['additionalProperties'])
                    self.assertEqual(sorted(node['required']), sorted(node['properties']))
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
        walk(schema(json.loads(TAXONOMY.read_text())))



class DensifyAndSmoothTests(unittest.TestCase):
    def test_dense_selection_keeps_every_label_frame_and_target_rate(self):
        from autolabel_densify import select_dense
        decoded = [(None, i/25) for i in range(126)]  # 5 s at 25 fps
        labels = [0.0, 0.52, 1.04, 4.98]
        keep, pos = select_dense(decoded, labels, fps=12.5)
        times = [decoded[i][1] for i in keep]
        for lt in labels:
            self.assertAlmostEqual(times[pos[lt]], lt, delta=0.021)
        self.assertTrue(55 <= len(keep) <= 70)

    def test_viterbi_removes_single_frame_blip_but_keeps_sustained_change(self):
        from autolabel_smooth import POSITIONS, emission, position_transitions, viterbi
        seq = ['guard']*5 + ['mount'] + ['guard']*5 + ['side_control']*6
        E = np.stack([emission(POSITIONS, s, 'medium') for s in seq])
        path = [POSITIONS[i] for i in viterbi(position_transitions(), E)]
        self.assertEqual(path[5], 'guard')
        self.assertEqual(path[-1], 'side_control')

    def test_transition_rows_are_distributions(self):
        from autolabel_smooth import position_transitions
        T = position_transitions()
        self.assertTrue(np.allclose(T.sum(1), 1))
        self.assertTrue((T > 0).all())


if __name__ == '__main__':
    unittest.main()
