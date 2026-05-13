import json
import pytest
import numpy as np
from evaluator.tracking_metrics import TrackingMetrics

@pytest.fixture
def gt_and_pred():
    with open("data/gt_tracking.json") as f:
        gt_data = json.load(f)
    with open("data/track_predictions.json") as f:
        pred_data = json.load(f)
    return gt_data, pred_data

def get_metrics(gt_data, pred_data):
    """共用函数，避免重复代码"""
    gt_by_frame = {}
    for ann in gt_data["annotations"]:
        fid = ann["image_id"]
        if fid not in gt_by_frame:
            gt_by_frame[fid] = []
        x, y, w, h = ann["bbox"]
        gt_by_frame[fid].append([x, y, x+w, y+h, ann["track_id"]])

    metrics = TrackingMetrics()
    for frame_id, pred in pred_data.items():
        fid = int(frame_id)
        preds = [[d["bbox"][0], d["bbox"][1],
                  d["bbox"][0]+d["bbox"][2],
                  d["bbox"][1]+d["bbox"][3],
                  d["track_id"]] for d in pred["tracks"]]
        gts = gt_by_frame.get(fid, [])
        metrics.update(preds, gts)
    return metrics.compute()

def test_mota(gt_and_pred, record_property):
    """MOTA必须高于0.5"""
    gt_data, pred_data = gt_and_pred
    result = get_metrics(gt_data, pred_data)
    record_property("MOTA", f"{result['MOTA']}")
    record_property("MOTP", f"{result['MOTP']}")
    record_property("ID_Switches", f"{result['ID_Switches']}")
    record_property("Misses", f"{result['Misses']}")
    record_property("False_Positives", f"{result['False_Positives']}")
    record_property("Total_GT", f"{result['Total_GT']}")
    assert result["MOTA"] >= 0.5, f"MOTA {result['MOTA']} too low"

def test_motp(gt_and_pred, record_property):
    """MOTP必须高于0.7"""
    gt_data, pred_data = gt_and_pred
    result = get_metrics(gt_data, pred_data)
    record_property("MOTP", f"{result['MOTP']}")
    record_property("Threshold", "≥ 0.70")
    assert result["MOTP"] >= 0.7, f"MOTP {result['MOTP']} too low"

def test_id_switches(gt_and_pred, record_property):
    """ID Switch次数必须少于50"""
    gt_data, pred_data = gt_and_pred
    result = get_metrics(gt_data, pred_data)
    record_property("ID_Switches", f"{result['ID_Switches']}")
    record_property("Threshold", "≤ 50")
    assert result["ID_Switches"] <= 50, f"Too many ID switches: {result['ID_Switches']}"