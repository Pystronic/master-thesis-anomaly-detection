from typing import Mapping


def calculate_AD_metrics(metrics: Mapping[str, float]) -> Mapping[str, float]:
    mADi = (metrics["IMG_AUROC"] + metrics["IMG_AP"] + metrics["IMG_F1Score"]) / 3
    mADp = (metrics["PX_AUROC"] + metrics["PX_AP"] + metrics["PX_F1Max"]) / 3
    mAD_2_8 = (metrics["PX_0.2_0.8_F1"] + metrics["PX_0.2_0.8_Acc"] + metrics["PX_0.2_0.8_mIoU"] + metrics["PX_mIoUMax"]) / 4
    return {
        **metrics,
        "mADi": mADi,
        "mADp": mADp,
        "mAD_0.2_0.8": mAD_2_8
    }