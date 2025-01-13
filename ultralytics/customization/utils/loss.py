# coding: utf-8
"""
Author: Jet C.
GitHub: https://github.com/jet-c-21
Create Date: 2025-01-12
"""
from typing import Dict

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


def v8_detection_loss_cal_loss_per_cls(self: v8DetectionLoss,
                                       preds,
                                       batch,
                                       existed_result_dict: Dict = None, ) -> Dict:
    """
    return:
        {
            <class_index>: {
                        "batch_weighted_loss": torch.Tensor,
                        "detached_loss": torch.Tensor ( loss(box, cls, dfl) ),
                    },
                    ...
        }

    """
    if existed_result_dict is not None:
        result = existed_result_dict
    else:
        result = {}

    # Initialize variables
    feats = preds[1] if isinstance(preds, tuple) else preds
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
        (self.reg_max * 4, self.nc), 1
    )

    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h, w)
    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
