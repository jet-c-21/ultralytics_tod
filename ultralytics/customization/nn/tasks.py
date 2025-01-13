# coding: utf-8
"""
Author: Jet C.
GitHub: https://github.com/jet-c-21
Create Date: 2025-01-12
"""
from typing import Dict, TYPE_CHECKING, Union

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss, E2EDetectLoss

def detection_model_cal_loss_per_cls(self: DetectionModel, batch, preds=None, existed_result_dict:Dict=None) -> Dict:
    if getattr(self, "criterion", None) is None:
        self.criterion = self.init_criterion()

    preds = self.forward(batch["img"]) if preds is None else preds

    if getattr(self, "end2end", False):
        self.criterion: E2EDetectLoss
        raise NotImplementedError("End2End training is not supported yet.")

    else:
        self.criterion: v8DetectionLoss
