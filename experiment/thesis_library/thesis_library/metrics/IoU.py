import math
from typing import Optional, Union

import torch
from anomalib.metrics import AnomalibMetric
from anomalib.metrics.binning import thresholds_between_0_and_1, thresholds_between_min_and_max
from anomalib.metrics.plotting_utils import plot_figure
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import Metric
from matplotlib.axes import Axes

"""
Average Intersection over Union over each Image.
This is a simple implementation without argument validation.

Code adapted from ADER (https://github.com/zhangzjn/ADer)

https://github.com/zhangzjn/ADer/blob/main/util/metric.py
"""
class _IoU(Metric):
    confmat: Tensor

    def __init__(
            self,
            thresholds: Optional[Union[int, list[float], Tensor]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        computed_thresholds = thresholds
        if isinstance(thresholds, int):
            computed_thresholds = thresholds_between_0_and_1(thresholds)


        self.register_buffer("thresholds", computed_thresholds, persistent=False)
        self.add_state(
            "confmat", default=torch.zeros(len(computed_thresholds), 2, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states."""
        preds = preds.flatten()
        target = target.flatten()

        self.confmat += _iou_update(preds, target, self.thresholds)

    def compute(self) -> tuple[Tensor, Tensor]:
        """Compute metric."""
        intersections = self.confmat[:, 1]
        unions = self.confmat[:, 0]
        iou = intersections / unions

        return iou, self.thresholds

    def plot(
        self,
        curve: Optional[tuple[Tensor, Tensor]] = None,
        score: Optional[Union[Tensor, bool]] = None,
        ax: Optional[Axes] = None,
    ) -> tuple[Figure, Axis] | None:

        curve_computed = curve or self.compute()
        # switch order so that thresholds are x-Axis
        return plot_figure(
            curve_computed[0],
            curve_computed[1],
            torch.tensor(1),
            (0, 1),
            (0, 1),
            xlabel="thresholds",
            ylabel="IoU",
            loc="lower right",
            title="IoU / threshold"
        )


def _iou_update(
    preds: Tensor,
    target: Tensor,
    thresholds: Tensor,
) -> Tensor:
    """Return the multi-threshold confusion matrix to calculate the IoU values with.
    """
    len_t = len(thresholds)
    target = target == 1
    confmat = thresholds.new_empty((len_t, 2), dtype=torch.int64)
    # Iterate one threshold at a time to conserve memory
    for i in range(len_t):
        preds_t = preds >= thresholds[i]

        # intersection
        confmat[i, 1] = (target & preds_t).sum()
        # union
        confmat[i, 0] = (target | preds_t).sum()
    return confmat


################# Actual metric classes
class _mIoU(_IoU):
    def compute(self) -> Tensor:
        iou, thresholds = super().compute()
        return iou.mean()

class _mIoUMax(_IoU):
    def __init__(self, **kwargs) -> None:
        super().__init__( **kwargs)
        self.threshold: torch.Tensor | None = None

    def compute(self) -> Tensor:
        iou, thresholds = super().compute()
        # Mean of each threshold
        max_thresh_i = iou.argmax()
        # Assign threshold to attribute like F1Max implementation
        self.threshold = thresholds[max_thresh_i]
        return iou[max_thresh_i]


################## Anomalib-Wrappers
class mIoU(AnomalibMetric, _mIoU):
    """Wrapper to add AnomalibMetric functionality to mIoU metric."""
class mIoUMax(AnomalibMetric, _mIoUMax):
    """Wrapper to add AnomalibMetric functionality to mIoU metric."""

################## Tests
def test_iou_calculation():
    # given
    target = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    predicted = torch.tensor([
        [0.1, 0.1, 0.0, 0.0],
        [0.1, 0.8, 0.8, 0.0],
        [0.0, 0.8, 0.8, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    expected_max = 1
    expected_mean = 0.7250

    # when
    iou_instance = _IoU(thresholds=10)
    iou_instance.update(predicted, target)
    actual = iou_instance.compute()[0]

    # then
    assert actual.max() == expected_max
    assert actual.mean() == expected_mean

def test_mIoU_calculation():
    # given
    target = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    predicted = torch.tensor([
        [0.1, 0.1, 0.0, 0.0],
        [0.1, 0.8, 0.8, 0.0],
        [0.0, 0.8, 0.8, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    expected_mean = 0.7250

    # when
    mIou_instance = _mIoU(thresholds=10)
    mIou_instance.update(predicted, target)
    actual = mIou_instance.compute()

    # then
    assert actual == expected_mean
    
def test_mIoUMax_compute_whenMultipleImages_thenMaxThesholdReturned():
    # given
    target = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    predicted1 = torch.tensor([
        [0.1, 0.1, 0.0, 0.0],
        [0.1, 0.2, 0.7, 0.0],
        [0.0, 0.8, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    predicted2 = torch.tensor([
        [0.1, 0.1, 0.0, 0.0],
        [0.1, 0.7, 0.9, 0.0],
        [0.0, 0.7, 0.7, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    delta = 0.005
    expected_max = 1
    expected_threshold = 0.111


    # when
    mIouMax_instance = _mIoUMax(thresholds=10)
    mIouMax_instance.update(predicted1, target)
    mIouMax_instance.update(predicted2, target)
    actual = mIouMax_instance.compute()

    # then
    assert actual == expected_max
    assert math.isclose(mIouMax_instance.threshold, expected_threshold, rel_tol=delta)