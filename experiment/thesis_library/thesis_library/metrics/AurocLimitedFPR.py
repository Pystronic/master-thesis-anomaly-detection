"""
Extension of anomalibs AUROC class to support FPR limits.
"""
import torch
from anomalib.metrics.plotting_utils import plot_figure
from matplotlib.figure import Figure
from torchmetrics.classification.roc import BinaryROC
from torchmetrics.utilities.compute import auc

from anomalib.metrics.base import AnomalibMetric

DEFAULT_MIN_FPR = 0.0
DEFAULT_MAX_FPR = 1.0

class _AUROCLimitFPR(BinaryROC):
    def __init__(
        self,
        min_fpr: float = DEFAULT_MIN_FPR,
        max_fpr: float = DEFAULT_MAX_FPR,
        **kwargs
    ) -> None:
        self.min_fpr = min_fpr
        self.max_fpr = max_fpr
        super().__init__(**kwargs)

    def compute(self) -> torch.Tensor:
        """First compute ROC curve, then compute area under the curve.

        Returns:
            torch.Tensor: Value of the AUROC metric
        """
        tpr: torch.Tensor
        fpr: torch.Tensor

        fpr, tpr = self._compute()
        return auc(fpr, tpr, reorder=True)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with new predictions and targets.

        Need to flatten new values as ROC expects them in this format for binary
        classification.

        Args:
            preds (torch.Tensor): Predictions from the model
            target (torch.Tensor): Ground truth target labels
        """
        super().update(preds.flatten(), target.flatten())

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute false positive rate and true positive rate value pairs.
           Extended from Anomalib-AUROC

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing tensors for FPR
                and TPR values
        """
        tpr: torch.Tensor
        fpr: torch.Tensor
        fpr, tpr, _thresholds = super().compute()

        # Custom filtering of FPR
        if self.min_fpr != DEFAULT_MIN_FPR:
            min_indices = fpr >= self.min_fpr
            fpr = fpr[min_indices]
            tpr = tpr[min_indices]

        if self.max_fpr != DEFAULT_MAX_FPR:
            max_indices = fpr >= self.max_fpr
            fpr = fpr[max_indices]
            tpr = tpr[max_indices]

        return (fpr, tpr)

    def generate_figure(self) -> tuple[Figure, str]:
        """
            Implementation form Anomalib-AUROC
        """
        fpr, tpr = self._compute()
        auroc = self.compute()

        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)
        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        loc = "lower right"
        title = "ROC"

        fig, axis = plot_figure(fpr, tpr, auroc, xlim, ylim, xlabel, ylabel, loc, title)

        axis.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            figure=fig,
        )

        return fig, title


class AUROCLimitFPR(AnomalibMetric, _AUROCLimitFPR):  # type: ignore[misc]
    """Wrapper to add AnomalibMetric functionality to AUROC metric."""
