import json
import pytest
import numpy as np
from evaluator.detection_metrics import compute_precision_recall, compute_f1, compute_map

@pytest.fixture
def gt_and_pred():
    with open("data/gt.json") as f:
        gt_data = json.load(f)
    with open("data/predictions.json") as f:
        pred_data = json.load(f)
    return gt_data, pred_data

def test_precision_vs_gt(gt_and_pred, record_property):
    """YOLOv8 Precision必须高于0.75"""
    gt_data, pred_data = gt_and_pred
    gt_by_frame = {}
    for ann in gt_data["annotations"]:
        fid = ann["image_id"]
        if fid not in gt_by_frame:
            gt_by_frame[fid] = []
        x, y, w, h = ann["bbox"]
        gt_by_frame[fid].append([x, y, x+w, y+h])

    precisions = []
    for frame_id, pred in pred_data.items():
        fid = int(frame_id)
        preds = [[d["bbox"][0], d["bbox"][1],
                  d["bbox"][0]+d["bbox"][2],
                  d["bbox"][1]+d["bbox"][3],
                  d["confidence"]] for d in pred["detections"]]
        gts = gt_by_frame.get(fid, [])
        p, _ = compute_precision_recall(preds, gts)
        precisions.append(p)

    avg_precision = np.mean(precisions)
    record_property("Precision", f"{avg_precision:.4f}")
    record_property("Threshold", "≥ 0.75")
    assert avg_precision >= 0.75, f"Precision {avg_precision:.4f} too low"

def test_recall_vs_gt(gt_and_pred, record_property):
    """YOLOv8 Recall必须高于0.90"""
    gt_data, pred_data = gt_and_pred
    gt_by_frame = {}
    for ann in gt_data["annotations"]:
        fid = ann["image_id"]
        if fid not in gt_by_frame:
            gt_by_frame[fid] = []
        x, y, w, h = ann["bbox"]
        gt_by_frame[fid].append([x, y, x+w, y+h])

    recalls = []
    for frame_id, pred in pred_data.items():
        fid = int(frame_id)
        preds = [[d["bbox"][0], d["bbox"][1],
                  d["bbox"][0]+d["bbox"][2],
                  d["bbox"][1]+d["bbox"][3],
                  d["confidence"]] for d in pred["detections"]]
        gts = gt_by_frame.get(fid, [])
        _, r = compute_precision_recall(preds, gts)
        recalls.append(r)

    avg_recall = np.mean(recalls)
    record_property("Recall", f"{avg_recall:.4f}")
    record_property("Threshold", "≥ 0.90")
    assert avg_recall >= 0.90, f"Recall {avg_recall:.4f} too low"

def test_f1_vs_gt(gt_and_pred, record_property):
    """F1 Score必须高于0.80"""
    gt_data, pred_data = gt_and_pred
    gt_by_frame = {}
    for ann in gt_data["annotations"]:
        fid = ann["image_id"]
        if fid not in gt_by_frame:
            gt_by_frame[fid] = []
        x, y, w, h = ann["bbox"]
        gt_by_frame[fid].append([x, y, x+w, y+h])

    precisions, recalls = [], []
    for frame_id, pred in pred_data.items():
        fid = int(frame_id)
        preds = [[d["bbox"][0], d["bbox"][1],
                  d["bbox"][0]+d["bbox"][2],
                  d["bbox"][1]+d["bbox"][3],
                  d["confidence"]] for d in pred["detections"]]
        gts = gt_by_frame.get(fid, [])
        p, r = compute_precision_recall(preds, gts)
        precisions.append(p)
        recalls.append(r)

    f1 = compute_f1(np.mean(precisions), np.mean(recalls))
    record_property("F1 Score", f"{f1:.4f}")
    record_property("Threshold", "≥ 0.80")
    assert f1 >= 0.80, f"F1 {f1:.4f} too low"

def test_map_vs_gt(gt_and_pred, record_property):
    """mAP必须高于0.7"""
    gt_data, pred_data = gt_and_pred
    gt_by_frame = {}
    for ann in gt_data["annotations"]:
        fid = ann["image_id"]
        if fid not in gt_by_frame:
            gt_by_frame[fid] = []
        x, y, w, h = ann["bbox"]
        gt_by_frame[fid].append([x, y, x+w, y+h])

    all_predictions, all_ground_truths = [], []
    for frame_id, pred in pred_data.items():
        fid = int(frame_id)
        preds = [[d["bbox"][0], d["bbox"][1],
                  d["bbox"][0]+d["bbox"][2],
                  d["bbox"][1]+d["bbox"][3],
                  d["confidence"]] for d in pred["detections"]]
        gts = gt_by_frame.get(fid, [])
        all_predictions.append(preds)
        all_ground_truths.append(gts)

    map_score = compute_map(all_predictions, all_ground_truths)
    record_property("mAP", f"{map_score:.4f}")
    record_property("Threshold", "≥ 0.70")
    assert map_score >= 0.7, f"mAP {map_score:.4f} too low"