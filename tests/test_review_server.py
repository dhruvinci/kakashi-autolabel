from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from serve_review import byte_range


class RangeTests(unittest.TestCase):
    def test_standard_open_ended_and_suffix_ranges_are_inclusive_and_clamped(self):
        self.assertEqual(byte_range('bytes=4-8',10),(4,8))
        self.assertEqual(byte_range('bytes=4-',10),(4,9))
        self.assertEqual(byte_range('bytes=-3',10),(7,9))
        self.assertEqual(byte_range('bytes=4-999',10),(4,9))

    def test_invalid_or_empty_ranges_are_rejected(self):
        for value,size in [('bytes=10-',10),('bytes=8-2',10),('bytes=-0',10),
                           ('bytes=0-1,3-4',10),('wat',10),('bytes=0-',0)]:
            with self.assertRaises(ValueError): byte_range(value,size)


if __name__=='__main__':
    unittest.main()
