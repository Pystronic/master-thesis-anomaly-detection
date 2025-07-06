import torch
from anomalib.data import ImageBatch

from thesis_library.data.miad_dataset import MIAD_CATEGORIES
from thesis_library.metrics import image, pixel

print(MIAD_CATEGORIES)


# Testing image level metrics [0 = normal, 1 = anomaly]
image_batch = ImageBatch(
    image=torch.rand(4, 3, 224, 224),
    pred_score=torch.tensor([0.1, 0.9, 0.1, 0.9]),
    pred_label=torch.tensor([0, 1, 0, 1]),
    gt_label=torch.tensor([False, False, True, True]),
)

image_metrics = image.get_metrics()
for metric in image_metrics:
    metric.update(image_batch)
    results = metric.compute()
    print(metric.name)
    print(results)


# Testing image level metrics [0 = normal, 1 = anomaly]
image_batch = ImageBatch(
    image=torch.rand(1, 3, 224, 224),
    anomaly_map=torch.tensor([[
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.9, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]]),
    gt_mask=torch.tensor([[
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]]).bool()
)

pixel_metrics = pixel.get_metrics()
for metric in pixel_metrics:
    metric.update(image_batch)
    results = metric.compute()
    print(metric.name)
    print(results)