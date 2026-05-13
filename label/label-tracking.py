import cv2
import json
import numpy as np
from ultralytics import YOLO

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

gt_model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture("../data/cows.mp4")

images = []
annotations = []
ann_id = 1
frame_id = 0
every_n = 1
COW_CLASS_ID = 19

prev_boxes = []   # 上一帧的 [x1,y1,x2,y2, track_id]
next_track_id = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % every_n == 0:
        h, w = frame.shape[:2]
        images.append({
            "id": frame_id,
            "file_name": f"frame_{frame_id:04d}.jpg",
            "width": w,
            "height": h
        })

        # 检测当前帧
        results = gt_model(frame, verbose=False)[0]
        curr_boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != COW_CLASS_ID:
                continue
            if float(box.conf[0]) < 0.1:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            curr_boxes.append([x1, y1, x2, y2])

        # 用IoU匹配上一帧，分配track_id
        assigned_ids = []
        used_prev = set()

        for curr in curr_boxes:
            best_iou = 0.1  # IoU阈值
            best_prev_idx = -1
            for i, prev in enumerate(prev_boxes):
                if i in used_prev:
                    continue
                iou = compute_iou(curr, prev[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_prev_idx = i

            if best_prev_idx >= 0:
                # 匹配到上一帧的牛，用同一个track_id
                track_id = prev_boxes[best_prev_idx][4]
                used_prev.add(best_prev_idx)
            else:
                # 新出现的牛
                track_id = next_track_id
                next_track_id += 1

            assigned_ids.append(track_id)

        # 保存标注
        for (x1, y1, x2, y2), track_id in zip(curr_boxes, assigned_ids):
            annotations.append({
                "id": ann_id,
                "image_id": frame_id,
                "category_id": 1,
                "bbox": [x1, y1, x2-x1, y2-y1],
                "track_id": track_id
            })
            ann_id += 1

        # 更新prev_boxes
        prev_boxes = [[*box, tid] for box, tid in zip(curr_boxes, assigned_ids)]

    frame_id += 1

cap.release()

gt = {
    "images": images,
    "annotations": annotations,
    "categories": [{"id": 1, "name": "cow"}]
}

with open("../data/gt_tracking.json", "w") as f:
    json.dump(gt, f, indent=2)

print(f"Tracking GT生成完成！{len(images)}帧，{len(annotations)}个标注")