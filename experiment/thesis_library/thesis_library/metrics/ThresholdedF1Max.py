from typing import Optional, Union

from anomalib.metrics import BinaryPrecisionRecallCurve, AnomalibMetric
from anomalib.metrics.f1_score import _F1Max
from torch import Tensor


class _ThresholdedF1Max(_F1Max):
    """
    Passes the threshold parameter to the F1Max metric.
    """

    def __init__(
            self,
            thresholds: Optional[Union[int, list[float], Tensor]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Delete precision recall instance without thresholds
        del self.precision_recall_curve

        # Initialize it with Thresholds
        self.precision_recall_curve = BinaryPrecisionRecallCurve(thresholds=thresholds)


class F1Max(AnomalibMetric, _ThresholdedF1Max):
    pass