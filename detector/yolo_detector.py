from ultralytics import YOLO
import time
class LivestockDetector:
    """
    封装YOLOv8，专门用于牲畜检测
    """
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # COCO数据集里的牲畜类别
        self.livestock_classes = {
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
        }
    def detect(self, frame):
        """
        对单帧图像做检测
        返回: list of [x1, y1, x2, y2, confidence, class_name]
        """
        results = self.model(frame, verbose=False)[0]
        #print(results)
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.livestock_classes:
                continue
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append([x1, y1, x2, y2, conf, self.livestock_classes[cls_id]])
        return detections
    def detect_with_fps(self, frame):
        """
        检测同时计算FPS
        """
        start = time.time()
        detections = self.detect(frame)
        fps = 1.0 / (time.time() - start)
        return detections, round(fps, 2)