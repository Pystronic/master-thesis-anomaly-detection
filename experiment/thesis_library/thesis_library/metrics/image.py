from anomalib.metrics import AnomalibMetric, AUROC
from torch import Tensor
from torchmetrics.classification import BinaryAveragePrecision

from thesis_library.metrics.ThresholdedF1Max import F1Max


class _AP(BinaryAveragePrecision):
    """
    Inputs need to be flattened to use with prediction scores and labels.
    """
    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds.flatten(), target.flatten())

class AP(AnomalibMetric, _AP):
    pass


# Metrics used on the image level:
#   TP, FP, TN, FN
#   F1-Score
#   AUROC
def get_metrics() -> list[AnomalibMetric]:
    num_thresholds = 100
    prefix = "IMG_"

    ap = AP(["pred_score", "gt_label"], prefix, thresholds=num_thresholds)
    f1_score = F1Max(["pred_score", "gt_label"], prefix, thresholds=1000)
    auroc = AUROC(["pred_score", "gt_label"], prefix, thresholds=num_thresholds)

    return [ap, f1_score, auroc]