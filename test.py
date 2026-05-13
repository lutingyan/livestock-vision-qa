# import json

# with open("data/gt.json") as f:
#     gt = json.load(f)

# with open("data/predictions.json") as f:
#     pred = json.load(f)

# # 检查平均每帧GT有几个标注
# total_ann = len(gt["annotations"])
# total_frames = len(gt["images"])
# print(f"GT: 总标注={total_ann}, 总帧数={total_frames}, 平均每帧={total_ann/total_frames:.2f}个")

# # 检查平均每帧prediction有几个
# total_pred = sum(len(v["detections"]) for v in pred.values())
# total_pred_frames = len(pred)
# print(f"Pred: 总检测={total_pred}, 总帧数={total_pred_frames}, 平均每帧={total_pred/total_pred_frames:.2f}个")



import json
from collections import Counter

with open("data/gt_tracking.json") as f:
    gt = json.load(f)

gt_ids = Counter(ann["track_id"] for ann in gt["annotations"])
print(f"GT总共有 {len(gt_ids)} 个不同track_id")
print(f"各ID出现帧数: {dict(sorted(gt_ids.items()))}")

print()

with open("data/track_predictions.json") as f:
    pred = json.load(f)

pred_ids = Counter()
for frame in pred.values():
    for track in frame["tracks"]:
        pred_ids[track["track_id"]] += 1

print(f"Tracker总共有 {len(pred_ids)} 个不同track_id")
print(f"各ID出现帧数: {dict(sorted(pred_ids.items()))}")