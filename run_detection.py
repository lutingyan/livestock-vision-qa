# import cv2
# from detector.yolo_detector import LivestockDetector
# # 初始化检测器
# detector = LivestockDetector()
# # 打开视频
# cap = cv2.VideoCapture("data/cows.mp4")
# print("开始检测,按Q退出...")
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # 检测 + 计算FPS
#     detections, fps = detector.detect_with_fps(frame)
#     print(detections)
#     # 画检测框
#     for det in detections:
#         x1, y1, x2, y2, conf, class_name = det
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"{class_name} {conf:.2f}"
#         cv2.putText(frame, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     # 显示FPS
#     cv2.putText(frame, f"FPS: {fps}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     cv2.imshow("Livestock Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
# print("检测完成！")

import cv2
import json
from detector.yolo_detector import LivestockDetector

detector = LivestockDetector()
cap = cv2.VideoCapture("data/cows.mp4")

predictions = {}
frame_id = 0

print("开始检测...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections, fps = detector.detect_with_fps(frame)

    predictions[frame_id] = {
        "detections": [
            {
                "bbox": [x1, y1, x2-x1, y2-y1],
                "confidence": conf,
                "class": class_name
            }
            for x1, y1, x2, y2, conf, class_name in detections
        ],
        "fps": fps
    }

    frame_id += 1

cap.release()

with open("data/predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

print(f"预测完成！{frame_id}帧，结果保存到 data/predictions.json")