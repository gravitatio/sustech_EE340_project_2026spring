import numpy as np
import torch
import unittest

from refuge_seg.losses import DiceLoss, TopologyLoss
from refuge_seg.metrics import dice_score, iou_score
from refuge_seg.postprocess import enforce_cup_inside_disc, fill_holes, keep_largest_component


class LossMetricPostprocessTests(unittest.TestCase):
    def test_dice_and_iou_are_one_for_perfect_prediction(self):
        pred = torch.tensor([[[0, 1], [2, 2]]])
        target = torch.tensor([[[0, 1], [2, 2]]])

        dice = dice_score(pred, target, num_classes=3)
        iou = iou_score(pred, target, num_classes=3)

        self.assertTrue(torch.allclose(dice, torch.ones(3)))
        self.assertTrue(torch.allclose(iou, torch.ones(3)))

    def test_dice_loss_is_near_zero_for_confident_correct_logits(self):
        logits = torch.tensor(
            [[[[8.0, -8.0], [-8.0, -8.0]], [[-8.0, 8.0], [-8.0, -8.0]], [[-8.0, -8.0], [8.0, 8.0]]]]
        )
        target = torch.tensor([[[0, 1], [2, 2]]])

        loss = DiceLoss(num_classes=3)(logits, target)

        self.assertLess(loss.item(), 1e-3)

    def test_topology_loss_penalizes_cup_probability_outside_disc(self):
        logits = torch.zeros(1, 3, 2, 2)
        logits[:, 1] = torch.tensor([[[-4.0, -4.0], [-4.0, -4.0]]])
        logits[:, 2] = torch.tensor([[[4.0, -4.0], [-4.0, -4.0]]])

        loss = TopologyLoss(disc_class=1, cup_class=2)(logits)

        self.assertGreater(loss.item(), 0.1)

    def test_keep_largest_component_removes_small_island(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0] = True
        mask[2:5, 2:5] = True

        cleaned = keep_largest_component(mask)

        self.assertEqual(int(cleaned.sum()), 9)
        self.assertFalse(bool(cleaned[0, 0]))

    def test_fill_holes_closes_internal_background_region(self):
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False

        filled = fill_holes(mask)

        self.assertTrue(bool(filled.all()))

    def test_enforce_cup_inside_disc_removes_detached_cup_component(self):
        pred = np.array([[2, 0, 0], [0, 1, 2], [0, 1, 1]], dtype=np.uint8)

        fixed = enforce_cup_inside_disc(pred, disc_class=1, cup_class=2)

        self.assertEqual(int(fixed[0, 0]), 0)
        self.assertEqual(int(fixed[1, 2]), 2)
