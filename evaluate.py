import json
import numpy as np
from evaluator.detection_metrics import compute_iou, compute_precision_recall, compute_f1

# 加载GT
with open("data/gt.json") as f:
    gt_data = json.load(f)

# 加载predictions
with open("data/predictions.json") as f:
    pred_data = json.load(f)

# 把GT按frame_id整理
gt_by_frame = {}
for ann in gt_data["annotations"]:
    fid = ann["image_id"]
    if fid not in gt_by_frame:
        gt_by_frame[fid] = []
    x, y, w, h = ann["bbox"]
    gt_by_frame[fid].append([x, y, x+w, y+h])

# 逐帧对比
all_precisions = []
all_recalls = []
all_fps_list = []

for frame_id, pred in pred_data.items():
    fid = int(frame_id)

    preds = [[d["bbox"][0], d["bbox"][1],
              d["bbox"][0]+d["bbox"][2],
              d["bbox"][1]+d["bbox"][3],
              d["confidence"]] for d in pred["detections"]]

    gts = gt_by_frame.get(fid, [])

    p, r = compute_precision_recall(preds, gts)
    all_precisions.append(p)
    all_recalls.append(r)
    all_fps_list.append(pred["fps"])

avg_precision = np.mean(all_precisions)
avg_recall = np.mean(all_recalls)
avg_f1 = compute_f1(avg_precision, avg_recall)
avg_fps = np.mean(all_fps_list)

print("=" * 40)
print("Detection Evaluation Results")
print("=" * 40)
print(f"Precision:  {avg_precision:.4f}")
print(f"Recall:     {avg_recall:.4f}")
print(f"F1 Score:   {avg_f1:.4f}")
print(f"Avg FPS:    {avg_fps:.2f}")
print("=" * 40)

# 保存结果
results = {
    "precision": round(avg_precision, 4),
    "recall": round(avg_recall, 4),
    "f1": round(avg_f1, 4),
    "avg_fps": round(avg_fps, 2)
}
with open("data/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("结果保存到 data/eval_results.json")