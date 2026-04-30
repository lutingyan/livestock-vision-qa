import pytest
from evaluator.detection_metrics import compute_iou, compute_precision_recall, compute_f1, compute_map


def test_iou_perfect_overlap():
    """两个完全一样的框，IoU应该是1.0"""
    box = [100, 100, 300, 300]
    assert compute_iou(box, box) == 1.0


def test_iou_no_overlap():
    """两个完全不重叠的框，IoU应该是0.0"""
    box1 = [0, 0, 100, 100]
    box2 = [200, 200, 300, 300]
    assert compute_iou(box1, box2) == 0.0


def test_iou_partial_overlap():
    """部分重叠，IoU应该在0到1之间"""
    box1 = [0, 0, 200, 200]
    box2 = [100, 100, 300, 300]
    iou = compute_iou(box1, box2)
    assert 0 < iou < 1


def test_precision_recall_perfect():
    """完美检测，Precision和Recall都应该是1.0"""
    predictions = [
        [100, 100, 300, 300, 0.9],
        [400, 200, 600, 450, 0.8],
    ]
    ground_truths = [
        [100, 100, 300, 300],
        [400, 200, 600, 450],
    ]
    p, r = compute_precision_recall(predictions, ground_truths)
    assert p == 1.0
    assert r == 1.0


def test_precision_recall_with_false_positive():
    """有误检的情况，Precision应该小于1.0"""
    predictions = [
        [100, 100, 300, 300, 0.9],  # 正确
        [700, 700, 900, 900, 0.8],  # 误检
    ]
    ground_truths = [
        [100, 100, 300, 300],
    ]
    p, r = compute_precision_recall(predictions, ground_truths)
    assert p < 1.0
    assert r == 1.0


def test_map_threshold():
    """mAP应该大于0.6才算pass"""
    all_predictions = [
        [[100, 100, 300, 300, 0.9]],
        [[400, 200, 600, 450, 0.8]],
    ]
    all_ground_truths = [
        [[100, 100, 300, 300]],
        [[400, 200, 600, 450]],
    ]
    map_score = compute_map(all_predictions, all_ground_truths)
    assert map_score >= 0.6, f"mAP {map_score} is below threshold 0.6"