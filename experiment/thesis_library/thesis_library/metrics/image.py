from anomalib.metrics import AnomalibMetric, AUROC, F1Score
from torchmetrics.classification import BinaryStatScores, BinaryPrecision, BinaryRecall, BinaryAUROC

from thesis_library.metrics.AurocLimitedFPR import AUROCLimitFPR


# TP, FP, TN, FN
class StatScores(AnomalibMetric, BinaryStatScores):
    pass

# Metrics used on the image level:
#   TP, FP, TN, FN
#   F1-Score
#   AUROC
#   AUROC>30%

def get_metrics() -> list[AnomalibMetric]:
    prefix = "IMG_"
    stat_score = StatScores(["pred_label", "gt_label"], prefix)
    f1_score = F1Score(["pred_label", "gt_label"], prefix)
    auroc = AUROC(["pred_score", "gt_label"], prefix)
    auroc_30 = AUROCLimitFPR(["pred_score", "gt_label"], prefix + '30%_', False, min_fpr=0.3)
    return [stat_score, f1_score, auroc, auroc_30]

# Calculate TPR, FPR, Precision, Recall from StatScores
def calculate_rates() -> None:
    # ToDo:
    pass