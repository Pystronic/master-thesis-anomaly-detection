from anomalib.metrics import AnomalibMetric, AUROC, F1Score
from torch import Tensor
from torchmetrics.classification import BinaryAveragePrecision


class _AP(BinaryAveragePrecision):
    """
    Fix format in which the update method is called.
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
    f1_score = F1Score(["pred_label", "gt_label"], prefix)
    auroc = AUROC(["pred_score", "gt_label"], prefix, thresholds=num_thresholds)

    return [ap, f1_score, auroc]