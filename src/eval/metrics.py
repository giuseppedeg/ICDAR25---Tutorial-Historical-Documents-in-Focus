import numpy as np

def compute_iou(box1, box2):
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - inter_area
    
    return inter_area / union if union > 0 else 0

def average_precision(recalls, precisions):
    """11-point interpolation (VOC 2007 style)."""
    ap = 0
    for t in np.linspace(0, 1, 11):
        p = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0
        ap += p
    return ap / 11

def compute_ap_class(preds, gts, cls, iou_thr=0.5):
    """Compute AP for one class and IoU threshold."""
    preds = [p for p in preds if p[1] == cls]
    gts = [g for g in gts if g[1] == cls]
    npos = len(gts)
    
    if npos == 0:
        return None
    
    preds = sorted(preds, key=lambda x: -x[2])  # sort by confidence
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    matched = {}
    
    for i, p in enumerate(preds):
        gt_for_img = [g for g in gts if g[0] == p[0]]
        ious = [compute_iou(p[3:], g[2:]) for g in gt_for_img]
        
        if len(ious) > 0 and max(ious) >= iou_thr:
            j = np.argmax(ious)
            if (p[0], j) not in matched:
                tp[i] = 1
                matched[(p[0], j)] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recalls = tp / npos if npos > 0 else np.zeros(len(tp))
    precisions = tp / (tp + fp + 1e-6)
    
    return average_precision(recalls, precisions)

def compute_map(predictions, ground_truths, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    predictions: list of [image_id, class_id, score, x1,y1,x2,y2]
    ground_truths: list of [image_id, class_id, x1,y1,x2,y2]
    """
    classes = set([g[1] for g in ground_truths])
    
    ap_per_class = {cls: [] for cls in classes}
    
    for cls in classes:
        for thr in iou_thresholds:
            ap = compute_ap_class(predictions, ground_truths, cls, iou_thr=thr)
            if ap is not None:
                ap_per_class[cls].append(ap)
    
    # media sulle IoU per ogni classe
    mean_ap_per_class = {cls: np.mean(aps) if len(aps) > 0 else 0
                         for cls, aps in ap_per_class.items()}
    
    # mAP globale (media tra classi)
    mAP = np.mean(list(mean_ap_per_class.values()))
    
    return mAP, mean_ap_per_class
