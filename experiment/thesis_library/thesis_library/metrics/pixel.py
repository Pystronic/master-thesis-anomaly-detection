import numpy as np
import torch
from anomalib.metrics import AnomalibMetric, AUROC
from anomalib.metrics.f1_score import _F1Max
from torchmetrics.classification import BinaryAveragePrecision, BinaryAccuracy, BinaryF1Score

from thesis_library.metrics.EfficientPRO import AUPRO
from thesis_library.metrics.IoU import mIoU, mIoUMax
from thesis_library.metrics.LimitedMetrics import _LimitDuringUpdate
from thesis_library.metrics.ThresholdedF1Max import F1Max

##### Deprecated #######
class F1(AnomalibMetric, _LimitDuringUpdate, BinaryF1Score):
    """
    Deprecated. Required to load old models which wrongly used this metric.
    """
    pass
class F1Limited(AnomalibMetric, _LimitDuringUpdate, _F1Max):
    """
    Deprecated. Required to load old models which used this metric.
    """
    pass
##########################



class AP(AnomalibMetric, BinaryAveragePrecision):
    pass

class Acc(AnomalibMetric, _LimitDuringUpdate, BinaryAccuracy):
    pass


# Metrics used on the pixel level per image:
#   AUROC
#   AP
#   AU-PRO
#   F1-max
#   mIoU

# Threshold based metrics
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
    pro = AUPRO(["anomaly_map", "gt_mask"], prefix, thresholds=num_thresholds)
    f1max = F1Max(["anomaly_map", "gt_mask"], prefix, thresholds=1000)
    miou = mIoU(["anomaly_map", "gt_mask"], prefix, thresholds=num_thresholds)

    # Threshold metric values
    prefix = "PX_0.2_0.8_"
    thresholds = torch.arange(start=0.2, end=0.8 + np.finfo(float).eps, step=0.1)

    f1_2_8 = F1Max(["anomaly_map", "gt_mask"], prefix, thresholds=thresholds)
    acc_2_8 = Acc(["anomaly_map", "gt_mask"], prefix, score_l=0.2, score_h=0.8)
    miou_2_8 = mIoU(["anomaly_map", "gt_mask"], prefix, thresholds=thresholds)
    miou_max = mIoUMax(["anomaly_map", "gt_mask"], prefix="PX_", thresholds=10)

    return [
        auroc, ap, pro, f1max, miou,
        f1_2_8, acc_2_8, miou_2_8, miou_max
    ]


def get_val_metrics() -> list[AnomalibMetric]:
    px_f1max = F1Max(["anomaly_map", "gt_mask"], prefix="PX_", thresholds=1000)
    img_auroc = AUROC(["pred_score", "gt_label"], prefix="IMG_", thresholds=100)
    return [img_auroc, px_f1max]
