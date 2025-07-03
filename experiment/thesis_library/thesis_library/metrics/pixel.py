from typing import Any

import numpy as np
import torch
from anomalib.metrics import AnomalibMetric, AUROC, AUPRO, F1Max
from torchmetrics.classification import BinaryAveragePrecision, BinaryF1Score, BinaryAccuracy

from thesis_library.metrics.IoU import mIoU, mIoUMax
from thesis_library.metrics.LimitedMetrics import _LimitDuringUpdate


class AP(AnomalibMetric, BinaryAveragePrecision):
    pass

class F1(AnomalibMetric, _LimitDuringUpdate, BinaryF1Score):
    pass

class Acc(AnomalibMetric, _LimitDuringUpdate, BinaryAccuracy):
    pass


# Metrics used on the pixel level per image:
#   AUROC
#   AP
#   AU-PRO
#   F1-max
#   mIoU

# Threshold based metrcis
#   mF1-2-8
#   mAcc-2-8
#   mIoU-2-8
#   IoU-max

def get_metrics() -> list[AnomalibMetric]:
    num_thresholds = 100

    # Attributes for pixel level metrics
    prefix = "PX_"
    auroc = AUROC(["anomaly_map", "gt_mask"], prefix, thresholds=num_thresholds)
    ap = AP(["anomaly_map", "gt_mask"], prefix, thresholds=num_thresholds)
    pro = AUPRO(["anomaly_map", "gt_mask"], prefix, num_thresholds=num_thresholds)
    f1max = F1Max(["anomaly_map", "gt_mask"], prefix)
    miou = mIoU(["anomaly_map", "gt_mask"], prefix, thresholds=num_thresholds)

    # Threshold metric values
    prefix = "PX_0.2_0.8_"
    thresholds = torch.arange(start=0.2, end=0.8 + np.finfo(float).eps, step=0.1)

    f1_2_8 = F1(["anomaly_map", "gt_mask"], prefix, score_l=0.2, score_h=0.8)
    acc_2_8 = Acc(["anomaly_map", "gt_mask"], prefix, score_l=0.2, score_h=0.8)
    miou_2_8 = mIoU(["anomaly_map", "gt_mask"], prefix, thresholds=thresholds)
    miou_max = mIoUMax(["anomaly_map", "gt_mask"], prefix="PX_", thresholds=10)

    return [
        auroc, ap, pro, f1max, miou,
        f1_2_8, acc_2_8, miou_2_8, miou_max
    ]

def get_val_metrics() -> list[AnomalibMetric]:
    num_thresholds = 100

    # Attributes for pixel level metrics
    prefix = "PX_"
    auroc = AUROC(["anomaly_map", "gt_mask"], prefix, thresholds=num_thresholds)
    return [auroc]