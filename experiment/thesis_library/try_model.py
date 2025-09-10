# Import the required modules
from pathlib import Path

import torch
from anomalib.callbacks import ModelCheckpoint
from anomalib.data import MVTecAD
from anomalib.data.utils import ValSplitMode
from anomalib.deploy import ExportType
from anomalib.metrics import Evaluator
from anomalib.models import Fastflow, Draem, Cfa, AnomalibModule, UniNet, Dinomaly, EfficientAd
from anomalib.engine import Engine
from lightning.pytorch.loggers import CSVLogger
from pandas import DataFrame

from thesis_library.metrics import image, pixel
from thesis_library.metrics.util import  calculate_AD_metrics

# Fix pytorch-lightning bug
# https://github.com/Lightning-AI/pytorch-lightning/issues/17124
def getstate_patch(*_):
    return {}
from torch.utils.data.dataloader import _BaseDataLoaderIter
_BaseDataLoaderIter.__getstate__ = getstate_patch




OUTPUT_PATH = Path("./results")
LOAD_MODEL_PATH = None # OUTPUT_PATH / "weights/torch/EfficientAd.pt"

# Initialize the datamodule, model, and engine

datamodule = MVTecAD(train_batch_size=1, eval_batch_size=1, val_split_mode=ValSplitMode.FROM_TEST, val_split_ratio=0.2)

# Initialize metrics
test_metrics = [
    *image.get_metrics(),
    *pixel.get_metrics()
]
# Reference needed later for workaround

logger = CSVLogger(OUTPUT_PATH / "logs", name="run_log")


# Note: Could work in theory
# def plotting_on_test_epoch_end(
#         self: Evaluator,
#         trainer: Trainer,
#         pl_module: LightningModule,
# ) -> None:
#     """Compute and log test metrics."""
#     del trainer, pl_module  # Unused argument.
#     for metric in self.test_metrics:
#         self.log(metric.name, metric)
#         if getattr(metric, 'plot', None) is not None:
#             try:
#                 fig, axis = metric.plot()
#                 print(f'Plotting {metric.name}')
#                 fig.show()
#             except Exception as e:
#                 print(e)
#                 pass

evaluator = Evaluator(
    test_metrics=test_metrics,
    val_metrics=pixel.get_val_metrics(),
    compute_on_cpu=False,
)
# Bind plotting function instead of original on_test_epoch_end
#setattr(evaluator, 'on_test_epoch_end', plotting_on_test_epoch_end.__get__(evaluator, evaluator.__class__))



checkpoint_callback = ModelCheckpoint(
    dirpath="model/chk",
    filename="best-{epoch:02d}-{PX_F1Max:.3f}",
    monitor="PX_F1Max",
    mode="max",
    save_top_k=3
)

engine = Engine(max_epochs=1,
                max_steps=10,
                callbacks=[checkpoint_callback],
                logger=logger,
                default_root_dir=OUTPUT_PATH,

)


# Reset pre-processor to stop resizing images
pre_processor = AnomalibModule.configure_pre_processor((512, 512))

# Specify backbone and layers


model = Fastflow(evaluator=evaluator, pre_processor=pre_processor)
if LOAD_MODEL_PATH is not None:
    # Load model weight we can trust, since we exported it ourselves
    model.load_state_dict(torch.load(LOAD_MODEL_PATH, weights_only=False), strict=False)



# Train the model
if LOAD_MODEL_PATH is None:
    engine.fit(datamodule=datamodule, model=model)
    engine.export(model, ExportType.TORCH, model_file_name="EfficientAd", export_root=OUTPUT_PATH)

# Returned as single element list
test_result = engine.test(datamodule=datamodule, model=model)[0]

print("##############")
test_result = calculate_AD_metrics(test_result)
print(test_result)

# Export result metrics
result_frame = DataFrame.from_records([dict(test_result)])
result_frame.to_csv(OUTPUT_PATH / "test_results.csv")