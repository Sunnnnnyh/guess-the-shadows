"""Check the experiment's invariants: mask fidelity, matched sizes and no leakage."""
import random
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api

from prepare_dataset import BANDS, decode_mask, matched_triplet, overlap_at, source_splits


class DatasetGeometryTests(unittest.TestCase):
    def test_intersection_with_negative_offsets(self):
        a = np.ones((10, 10), bool)
        b = np.ones((8, 8), bool)
        self.assertEqual(overlap_at(a, b, -4, 2), 32)
        self.assertEqual(overlap_at(a, b, 30, 30), 0)

    def test_original_rle_hole_survives_decoding(self):
        image = {'height': 30, 'width': 30}
        original = np.zeros((30, 30), np.uint8)
        original[3:27, 3:27] = 1
        original[10:20, 10:20] = 0
        rle = mask_api.encode(np.asfortranarray(original))
        rle['counts'] = rle['counts'].decode('ascii')
        decoded = decode_mask({'segmentation': rle}, image)
        np.testing.assert_array_equal(decoded, original.astype(bool))

    def test_matched_triplet_is_reproducible_and_not_clipped(self):
        a = np.zeros((65, 50), np.uint8)
        a[2:63, 5:45] = 255
        b = np.zeros((50, 65), np.uint8)
        b[5:45, 2:63] = 255
        masks = [Image.fromarray(a), Image.fromarray(b)]
        results = [matched_triplet(*masks, random.Random(360)) for _ in range(2)]
        self.assertIsNotNone(results[0])
        self.assertEqual(set(results[0]), set(BANDS))
        for name, (lo, hi) in BANDS.items():
            row = results[0][name]
            self.assertTrue(lo <= row['ios'] < hi)
            self.assertEqual(row['sizes'], results[0]['low']['sizes'])
            self.assertEqual(row['mask_areas'], results[0]['low']['mask_areas'])
            self.assertEqual(row['positions'], results[1][name]['positions'])
            self.assertLessEqual(max(row['mask_areas'])/min(row['mask_areas']), 2)
            self.assertEqual(row['image'].size, (128, 128))
            self.assertTrue(set(np.unique(row['image'])) <= {0, 255})
            for (w, h), (x, y) in zip(row['sizes'], row['positions']):
                self.assertTrue(0 <= x and 0 <= y and x+w <= 128 and y+h <= 128)
            expected_union = sum(row['mask_areas'])-row['intersection_pixels']
            self.assertEqual(int(np.count_nonzero(np.asarray(row['image'])==0)), expected_union)

    def test_source_split_does_not_depend_on_input_order(self):
        normal = source_splits(range(100), 360)
        reverse = source_splits(reversed(range(100)), 360)
        self.assertEqual(normal, reverse)
        self.assertEqual(sum(v=='train' for v in normal.values()), 70)
        self.assertEqual(sum(v=='validation' for v in normal.values()), 15)
        self.assertEqual(sum(v=='test' for v in normal.values()), 15)


if __name__ == '__main__':
    unittest.main()
