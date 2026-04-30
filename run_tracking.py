import cv2
from detector.yolo_detector import LivestockDetector
from detector.tracker import SimpleTracker

# 初始化
detector = LivestockDetector()
tracker = SimpleTracker()

# 打开视频
cap = cv2.VideoCapture("data/cows.mp4")

print("开始追踪，按Q退出...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 检测
    detections, fps = detector.detect_with_fps(frame)

    # 追踪
    tracks = tracker.update(detections)

    # 画追踪框
    for track in tracks:
        x1, y1, x2, y2, track_id, class_name = track
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} ID:{track_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示FPS
    cv2.putText(frame, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Livestock Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("追踪完成！")