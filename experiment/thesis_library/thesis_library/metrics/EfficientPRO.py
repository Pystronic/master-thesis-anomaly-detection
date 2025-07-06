import torch
import time
from anomalib.data import ImageBatch
from anomalib.metrics import AnomalibMetric, AUPRO as AnomalibAUPRO
from pyaupro import PerRegionOverlap, auc_compute, generate_random_data


class _EfficientAUPRO(PerRegionOverlap):

    def __init__(
        self,
        fpr_limit: float = 0.3,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.register_buffer("fpr_limit", torch.tensor(fpr_limit))


    def compute(self) -> torch.Tensor:
        fpr, pro = super().compute()

        return auc_compute(fpr, pro, reorder=True, descending=True, limit=self.fpr_limit)


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
