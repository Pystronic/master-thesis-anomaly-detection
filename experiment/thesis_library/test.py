import numpy as np
import torch
from anomalib.data import ImageBatch

from thesis_library.miad_dataset import MIAD_CATEGORIES
from thesis_library.metrics import image

print(MIAD_CATEGORIES)


# Testing image level metrics [0 = normal, 1 = anomaly]
image_batch = ImageBatch(
    image=torch.rand(4, 3, 224, 224),
    pred_score=torch.from_numpy(np.array([0.1, 0.9, 0.1, 0.9])),
    pred_label=torch.from_numpy(np.array([0, 1, 0, 1])),
    gt_label=torch.from_numpy(np.array([False, False, True, True])),
)

image_metrics = image.get_metrics()
for metric in image_metrics:
    metric.update(image_batch)
    results = metric.compute()
    print(metric.name)
    print(results)