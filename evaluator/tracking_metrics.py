import numpy as np
from .detection_metrics import compute_iou
class TrackingMetrics:
    """
    计算多目标追踪KPI:
    - MOTA (Multi-Object Tracking Accuracy)
    - MOTP (Multi-Object Tracking Precision)
    - ID Switch次数
    - Track Fragmentation
    """
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    def reset(self):
        self.total_gt = 0          # 总ground truth数
        self.total_misses = 0      # 漏检数
        self.total_fp = 0          # 误检数
        self.total_id_switches = 0 # ID切换次数
        self.total_motp = 0        # MOTP累计
        self.total_matched = 0     # 匹配成功总数
        self.prev_matches = {}     # 上一帧的匹配关系 {gt_id: track_id}
    def update(self, predictions, ground_truths):
        """
        每帧调用一次
        predictions: list of [x1, y1, x2, y2, track_id]
        ground_truths: list of [x1, y1, x2, y2, gt_id]
        """
        self.total_gt += len(ground_truths)
        if len(ground_truths) == 0:
            self.total_fp += len(predictions)
            return
        if len(predictions) == 0:
            self.total_misses += len(ground_truths)
            return
        # 计算IoU矩阵
        iou_matrix = np.zeros((len(ground_truths), len(predictions)))
        for i, gt in enumerate(ground_truths):
            for j, pred in enumerate(predictions):
                iou_matrix[i, j] = compute_iou(gt[:4], pred[:4])
        # 贪心匹配
        current_matches = {}  # {gt_id: track_id}
        matched_preds = set()
        for gt_idx, gt in enumerate(ground_truths):
            best_iou = self.iou_threshold
            best_pred_idx = -1
            for pred_idx in range(len(predictions)):
                if pred_idx in matched_preds:
                    continue
                if iou_matrix[gt_idx, pred_idx] > best_iou:
                    best_iou = iou_matrix[gt_idx, pred_idx]
                    best_pred_idx = pred_idx
            if best_pred_idx >= 0:
                gt_id = gt[4]
                track_id = predictions[best_pred_idx][4]
                current_matches[gt_id] = track_id
                matched_preds.add(best_pred_idx)
                # 检查ID Switch
                if gt_id in self.prev_matches:
                    if self.prev_matches[gt_id] != track_id:
                        self.total_id_switches += 1
                # 累计MOTP
                self.total_motp += best_iou
                self.total_matched += 1
            else:
                self.total_misses += 1
        # 未匹配的predictions算FP
        self.total_fp += len(predictions) - len(matched_preds)
        self.prev_matches = current_matches
    def compute(self):
        """返回所有KPI"""
        mota = 1 - (
            (self.total_misses + self.total_fp + self.total_id_switches)
            / max(self.total_gt, 1)
        )
        motp = self.total_motp / max(self.total_matched, 1)

        return {
            "MOTA": round(mota, 4),
            "MOTP": round(motp, 4),
            "ID_Switches": self.total_id_switches,
            "Misses": self.total_misses,
            "False_Positives": self.total_fp,
            "Total_GT": self.total_gt,
        }