import numpy as np
from evaluator.detection_metrics import compute_iou


class SimpleTracker:
    """
    简单的多目标追踪器
    基于IoU匹配，帧与帧之间追踪同一头牛
    """
    def __init__(self, iou_threshold=0.2, max_lost=10):
        self.iou_threshold = iou_threshold  # IoU低于这个就不算同一头牛
        self.max_lost = max_lost            # 消失多少帧后删除这个track
        self.tracks = []                    # 当前所有track
        self.next_id = 1                    # 下一个新track的ID

    def update(self, detections):
        """
        输入：当前帧的检测结果 list of [x1, y1, x2, y2, conf, class_name]
        输出：list of [x1, y1, x2, y2, track_id, class_name]
        """
        if len(self.tracks) == 0:
            # 第一帧，所有检测都是新track
            for det in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'box': det[:4],
                    'class': det[5],
                    'lost': 0
                })
                self.next_id += 1
            return [[t['box'][0], t['box'][1], t['box'][2], t['box'][3],
                     t['id'], t['class']] for t in self.tracks]

        # 用IoU匹配当前检测和已有track
        matched, unmatched_dets, unmatched_tracks = self._match(detections)

        # 更新匹配到的track
        for det_idx, track_idx in matched:
            self.tracks[track_idx]['box'] = detections[det_idx][:4]
            self.tracks[track_idx]['lost'] = 0

        # 新检测没有匹配到track，创建新track
        for det_idx in unmatched_dets:
            self.tracks.append({
                'id': self.next_id,
                'box': detections[det_idx][:4],
                'class': detections[det_idx][5],
                'lost': 0
            })
            self.next_id += 1

        # 没匹配到检测的track，lost计数加1
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]['lost'] += 1

        # 删除lost太久的track
        self.tracks = [t for t in self.tracks if t['lost'] < self.max_lost]

        return [[t['box'][0], t['box'][1], t['box'][2], t['box'][3],
                 t['id'], t['class']] for t in self.tracks]

    def _match(self, detections):
        """用IoU贪心匹配detections和tracks"""
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(detections) == 0 or len(self.tracks) == 0:
            return matched, unmatched_dets, unmatched_tracks

        # 计算IoU矩阵
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                iou_matrix[d, t] = compute_iou(det[:4], track['box'])

        # 贪心匹配：找IoU最高的配对
        while True:
            if iou_matrix.max() < self.iou_threshold:
                break
            d, t = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched.append((d, t))
            unmatched_dets.remove(d)
            unmatched_tracks.remove(t)
            iou_matrix[d, :] = 0
            iou_matrix[:, t] = 0

        return matched, unmatched_dets, unmatched_tracks