import numpy as np
import unittest

from refuge_seg.data import mask_values_to_classes, classes_to_mask_values


class LabelMappingTests(unittest.TestCase):
    def test_refuge_mask_values_are_mapped_to_training_classes(self):
        mask = np.array([[255, 128], [0, 255]], dtype=np.uint8)

        mapped = mask_values_to_classes(mask)

        self.assertEqual(mapped.dtype, np.int64)
        self.assertEqual(mapped.tolist(), [[0, 1], [2, 0]])

    def test_training_classes_are_mapped_back_to_refuge_mask_values(self):
        classes = np.array([[0, 1], [2, 0]], dtype=np.int64)

        restored = classes_to_mask_values(classes)

        self.assertEqual(restored.dtype, np.uint8)
        self.assertEqual(restored.tolist(), [[255, 128], [0, 255]])

    def test_unknown_mask_value_raises_clear_error(self):
        mask = np.array([[255, 42]], dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "Unexpected REFUGE label values"):
            mask_values_to_classes(mask)
