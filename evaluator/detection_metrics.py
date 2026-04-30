import numpy as np


def compute_iou(box1, box2):
    """
    计算两个bounding box的IoU
    box格式: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def compute_precision_recall(predictions, ground_truths, iou_threshold=0.5):
    """
    计算Precision和Recall
    predictions: list of [x1, y1, x2, y2, confidence]
    ground_truths: list of [x1, y1, x2, y2]
    """
    if len(ground_truths) == 0:
        return 1.0 if len(predictions) == 0 else 0.0, 1.0

    if len(predictions) == 0:
        return 0.0, 0.0

    # 按confidence排序
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)

    matched_gt = set()
    tp = 0
    fp = 0

    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred[:4], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def compute_map(all_predictions, all_ground_truths, iou_thresholds=None):
    """
    计算mAP
    all_predictions: list of frames, 每帧是predictions列表
    all_ground_truths: list of frames, 每帧是ground_truths列表
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]  # 简化版，可扩展到0.5:0.95

    ap_list = []
    for threshold in iou_thresholds:
        precisions = []
        recalls = []
        for preds, gts in zip(all_predictions, all_ground_truths):
            p, r = compute_precision_recall(preds, gts, threshold)
            precisions.append(p)
            recalls.append(r)
        ap_list.append(np.mean(precisions))

    return np.mean(ap_list)