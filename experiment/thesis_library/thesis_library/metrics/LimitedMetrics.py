from anomalib.metrics import AnomalibMetric
from torch import Tensor
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy

class _LimitDuringUpdate:
    """
    Implementation to limit metrics during the update call.
    Workaround for Metrics which do not accept fixed thresholds.
    """
    def __init__(
            self,
            score_l: float = 0.0,
            score_h: float = 1.0,
            **kwargs
    ) -> None:
        self.score_l = score_l
        self.score_h = score_h
        super().__init__(**kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        filtered_preds_i = (preds >= self.score_l) <= self.score_h
        filtered_preds = preds[filtered_preds_i]
        if filtered_preds.numel() == 0:
            return


        filtered_target = target[filtered_preds_i]
        super().update(filtered_preds, filtered_target)