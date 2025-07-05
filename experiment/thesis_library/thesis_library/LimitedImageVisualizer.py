from typing import TYPE_CHECKING

from anomalib.visualization import ImageVisualizer
from anomalib.data import ImageBatch

# Only import types during type checking to avoid circular imports
if TYPE_CHECKING:
    from lightning.pytorch import Trainer

    from anomalib.models import AnomalibModule

class LimitedImageVisualizer(ImageVisualizer):
    """
    Extension of ImageVisualizer which limits the number of
    visualized images per category to a specified amount.

    This reduces test times massively for big datasets.
    """

    def __init__(
        self,
        image_limit: int,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.image_limit = image_limit

    def on_test_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "AnomalibModule",
        outputs: "ImageBatch",
        batch: "ImageBatch",
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""
        filtered_images = [i for i in batch if img_path_to_number(i.image_path) <= self.image_limit]
        if len(filtered_images) == batch.batch_size:
            super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        elif len(filtered_images) > 0:
            del batch
            super().on_test_batch_end(trainer, pl_module, outputs, ImageBatch.collate(filtered_images), batch_idx, dataloader_idx)

    def on_predict_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "AnomalibModule",
        outputs: "ImageBatch",
        batch: "ImageBatch",
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the predict batch ends."""
        return self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)




def img_path_to_number(path: str) -> int:
    try:
        return int(path[path.rindex('/')+1:path.rindex('.')])
    except:
        return 0