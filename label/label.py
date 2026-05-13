import cv2
import json
import os
from ultralytics import YOLO

gt_model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture("../data/cows.mp4")

images = []
annotations = []
ann_id = 1
frame_id = 0
every_n = 1

COW_CLASS_ID = 19

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

        results = gt_model(frame, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != COW_CLASS_ID:
                continue
            conf = float(box.conf[0])
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            annotations.append({
                "id": ann_id,
                "image_id": frame_id,
                "category_id": 1,
                "bbox": [x1, y1, x2-x1, y2-y1],
                "confidence": conf
            })
            ann_id += 1

    frame_id += 1

cap.release()

gt = {
    "images": images,
    "annotations": annotations,
    "categories": [{"id": 1, "name": "cow"}]
}

with open("../data/gt.json", "w") as f:
    json.dump(gt, f, indent=2)

print(f"GT生成完成！{len(images)}帧，{len(annotations)}个标注")