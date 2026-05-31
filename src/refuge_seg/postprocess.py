from __future__ import annotations

import cv2
import numpy as np


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = flood == 0
    return (mask_u8.astype(bool) | holes).astype(bool)


def enforce_cup_inside_disc(pred: np.ndarray, disc_class: int = 1, cup_class: int = 2) -> np.ndarray:
    fixed = pred.copy()
    disc_mask = fixed == disc_class
    cup_mask = fixed == cup_class
    if not disc_mask.any() or not cup_mask.any():
        fixed[cup_mask & ~disc_mask] = 0
        return fixed

    optic_region = disc_mask | cup_mask
    count, labels = cv2.connectedComponents(optic_region.astype(np.uint8), connectivity=4)
    valid_labels = set(np.unique(labels[disc_mask]).tolist())
    valid_labels.discard(0)
    valid_cup = cup_mask & np.isin(labels, list(valid_labels))
    fixed[cup_mask & ~valid_cup] = 0
    return fixed


def postprocess_prediction(pred: np.ndarray, disc_class: int = 1, cup_class: int = 2) -> np.ndarray:
    fixed = pred.copy()
    for class_id in (disc_class, cup_class):
        component = fixed == class_id
        if component.any():
            component = fill_holes(keep_largest_component(component))
            fixed[fixed == class_id] = 0
            fixed[component] = class_id
    return enforce_cup_inside_disc(fixed, disc_class=disc_class, cup_class=cup_class)
