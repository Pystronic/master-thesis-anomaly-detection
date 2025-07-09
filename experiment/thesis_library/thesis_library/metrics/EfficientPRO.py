import torch
import time
from warnings import warn
from scipy.ndimage import label
from anomalib.data import ImageBatch
from anomalib.metrics import AnomalibMetric, AUPRO as AnomalibAUPRO
from pyaupro import PerRegionOverlap, auc_compute, generate_random_data
from pyaupro._implementation import _get_structure
from torch import Tensor


class _EfficientAUPRO(PerRegionOverlap):
    """
    Add curve calculation to PerRegionOverlap metric.
    """

    def __init__(
        self,
        fpr_limit: float = 0.3,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.register_buffer("fpr_limit", torch.tensor(fpr_limit))

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states. Taken from PerRegionOverlap to fix https://github.com/davnn/pyaupro/issues/7."""
        assert preds.shape == target.shape, "Cannot update when preds.shape != target.shape."

        if self.thresholds is None:
            self.preds.append(preds)
            self.target.append(target)
            return

        # compute the fpr and pro for all preds and target given thresholds
        if (res := _fixed_per_region_overlap_update(preds, target, self.thresholds)) is not None:
            fpr, pro = res
            # weight the fpr and pro contribution given the number of samples
            num_samples = len(preds)
            self.fpr += fpr * num_samples
            self.pro += pro * num_samples
            self.num_updates += num_samples

    def compute(self) -> torch.Tensor:
        fpr, pro = super().compute()

        return auc_compute(fpr, pro, reorder=True, descending=True, limit=self.fpr_limit)


def _fixed_per_region_overlap_update(
    preds: Tensor,
    target: Tensor,
    thresholds: Tensor,
) -> tuple[Tensor, Tensor] | None:
    """Return the false positive rate and per-region overlap for the given thresholds.
       Updated version fixes https://github.com/davnn/pyaupro/issues/7.
    """
    # pre-compute total component areas for region overlap
    structure = _get_structure(target.ndim)
    components, n_components = label(target.numpy(force=True), structure=structure)

    # ensure that there are components available for overlap calculation
    if n_components == 0:
        return warn("No regions found in target for update, ignoring update.", stacklevel=2)

    # convert back to torch and flatten components to vector
    # Fix: Move CPU (caused by from_numpy) to target device
    flat_components = torch.from_numpy(components.ravel()).to(preds.device)

    # only keep true components (non-zero values)
    pos_comp_mask = flat_components > 0
    flat_components = flat_components[pos_comp_mask]
    region_total_area = torch.bincount(flat_components)[1:]

    # pre-compute the negative mask and flatten preds for perf
    negatives = (target == 0).ravel()
    total_negatives = negatives.sum()
    flat_preds = preds.ravel()

    # initialize the result tensors
    len_t = len(thresholds)
    false_positive_rate = thresholds.new_empty(len_t, dtype=torch.float64)
    per_region_overlap = thresholds.new_empty(len_t, dtype=torch.float64)

    # Iterate one threshold at a time to conserve memory
    for t in range(len_t):
        # compute false positive rate
        preds_t = flat_preds >= thresholds[t]
        false_positive_rate[t] = negatives[preds_t].sum()

        # compute per-region overlap
        region_overlap_area = torch.bincount(
            flat_components,
            weights=preds_t[pos_comp_mask],
            minlength=n_components,
        )[1:]
        # faster than region_overlap_area / region_total_area
        region_overlap_area.div_(region_total_area)
        per_region_overlap[t] = torch.mean(region_overlap_area)

    return false_positive_rate / total_negatives, per_region_overlap




class AUPRO(AnomalibMetric, _EfficientAUPRO):  # type: ignore[misc]
    """Wrapper to add AnomalibMetric functionality to AUPRO metric."""


def test_whenEfficientProUser_lessTimeRequiredForCalculation() -> None:
    masks, predictions = generate_random_data(batch_size=10, height=512, width=512, seed=4232)
    image_batch = ImageBatch(
        image=torch.rand([100, 3, 512, 512]),
        anomaly_map=masks,
        gt_mask=predictions.bool()
    )

    anomalib_aupro = AnomalibAUPRO(["anomaly_map", "gt_mask"], num_thresholds=1000)
    efficient_aupro = AUPRO(["anomaly_map", "gt_mask"], thresholds=1000)
    anomalib_aupro.update(image_batch)
    efficient_aupro.update(image_batch)


    anomalib_start = time.process_time()
    anomalib_score = anomalib_aupro.compute()
    anomalib_end = time.process_time()
    efficient_start = time.process_time()
    efficient_score = efficient_aupro.compute()
    efficient_end = time.process_time()

    anomalib_time = anomalib_end - anomalib_start
    efficient_time = efficient_end - efficient_start
    speedup = anomalib_time / efficient_time
    score_delta = abs(anomalib_score - efficient_score)
    print(f"Anomalib time: {anomalib_time}")
    print(f"Efficient time: {efficient_time}")
    print(f"Speedup: {speedup}")
    print(f"Score delta: {score_delta}")
    assert speedup > 1.1
    assert score_delta < 0.005
