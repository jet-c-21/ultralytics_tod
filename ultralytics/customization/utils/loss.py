# coding: utf-8
"""
Author: Jet C.
GitHub: https://github.com/jet-c-21
Create Date: 2025-01-12
"""
from typing import Union, Tuple, Dict, List, Any

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


def v8_detection_loss_cal_loss_per_cls(self: v8DetectionLoss,
                                       preds: Union[Tuple[torch.Tensor, torch.Tensor], Any],
                                       batch: Union[Dict[str, Union[torch.Tensor, List]], Any],
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

    When `preds` is a Tuple:
        preds[0]:
            in my case, it is a Tensor(16, 13, 5292),
                16 is batch_size,
                13 is ( 9 class + 4 bbox coord [cx, cy, w, h] ),
                5292 is the total number of anchor points or grid cells across all feature map scales.

        preds[1] (feats):
            in my case, it is a List of 3 Tensors,
            each tensor's format is
            (batch_size, number_of_channels, height, width)

            example in my case:
                [
                    Tensor 0: (16, 73, 48, 84), # a P3 feature map at a spatial size of 48(h)x84(w)
                    Tensor 1: (16, 73, 24, 42), # a P5 feature map at a spatial size of 24(h)x42(w)
                    Tensor 2: (16, 73, 12, 21), # a P7 feature map at a spatial size of 12(h)x21(w)
                ] ,

    """
    if existed_result_dict is not None:
        result = existed_result_dict
    else:
        result = {}

    # Initialize variables
    feats = preds[1] if isinstance(preds, tuple) else preds
    """
    
    """

    # # !@#
    # for i, xi in enumerate(feats):
    #     xx = xi.view(feats[0].shape[0], self.no, -1)
    #     print(f"[*DEBUG*] - #x{i} shape: {xx.shape}")


    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], dim=2).split(
        (self.reg_max * 4, self.nc), dim=1
    )
    """
    * feats[0].shape is (16, 73, 48, 84) in my case
    
    * feats[0].shape[0] is batch_size (16 in my case)
    
    * self.no: number of outputs per anchor point
        ultralytics.nn.modules.head.Detect 's reg_max default value is 16
        
        by default we will got:
            self.no = 9 (based on dataset yaml) + 16 (<DetectionModule>.reg_max) * 4 (4 coords for bbox)
                    = 73
    
    * xi.view(feats[0].shape[0], self.no, -1), when i = 0, in my case is (16, 73, 4032)
        [*DEBUG*] - #x0 shape: torch.Size([16, 73, 4032]) # 4032 = 48 * 84 (h * w)
        
        # for i = 1
        [*DEBUG*] - #x1 shape: torch.Size([16, 73, 1008]) # 1008 = 24 * 42 (h * w)
        
        # for i = 2
        [*DEBUG*] - #x2 shape: torch.Size([16, 73, 252]) # 252 = 12 * 21 (h * w)
    
    * torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], dim=2) 
        in my case is (16, 73, 5292)
        5292 = 4032 + 1008 + 252
    
    * reg_max:
        refers to the total number of bins or discrete values predicted for the regression of bounding box parameters:
        - cx (center x-coordinate of the box).
        - cy (center y-coordinate of the box).
        - w (width of the box).
        - h (height of the box).
    
    * pred_scores in my case is Tensor(16, 9, 5292) # (batch_size, number_of_classes, 5292)
    
    * pred_distri in my case is Tensor(16, 64, 5292) # (batch_size, reg_max for cx cy w h, 5292)
    """

    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    """
    * why we need to permute at here?
        - Class loss (e.g., Cross-Entropy) typically expects predictions 
          in the shape (batch_size, num_anchors, num_classes) — hence the permute for `pred_scores`
        - Bounding box regression often requires distributions 
          in the shape (batch_size, num_anchors, num_bins) — hence the permute for `pred_distri`
    
    * .contiguous() Method
        The `.contiguous()` method creates a new tensor with the same data 
        but stored in a contiguous memory layout. 
        If the tensor is already contiguous, `.contiguous()` has no effect.
    
    """

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h, w)
    """
    feats[0].shape[2:] is (48, 84) in my case
    self.stride[0] is 8 in my case
    in my case, imgsz is Tensor([384, 672], device='cuda:0')
    """

    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
    """
    * self.stride in my case is Tensor([8., 16., 32.], device='cuda:0')
        we can take reference from `ultralytics/cfg/models/v8/yolov8.yaml`
        - # 3-P3/8
        - # 5-P4/16
        - # 7-P5/32
    """

    # Targets
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), dim=1)
    """
    * batch["batch_idx"]: the key also means image index in the batch
        - if the length of batch["batch_idx"] is 167, it means there are 167 bbox in the batch.
        - the value range of batch["batch_idx"] is corresponding to the batch size (image count).
          in my case, the value range is 0-15 because the batch size is 16.
    
    each entry in targets is a Tensor(6, ) in format of [batch_idx, cls, cx, cy, w, h]
    in my case, the shape of targets is (167, 6)
    """

    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    """
    after preprocess, the shape of targets is in format:
        [batch_size, max_bbox_count__across_all_images_in_this_batch, target_attribute_cls_x1_y1_x2_y2]   
        in my case the shape of target is (16, 22, 5)
    """

    gt_labels, gt_bboxes = targets.split((1, 4), dim=2)  # cls, xyxy
    """
    in my case the shape of gt_labels is (16, 22, 1) and gt_bboxes is (16, 22, 4)
    """

    mask_gt = gt_bboxes.sum(dim=2, keepdim=True).gt_(0.0)
    """
    in my case the shape of mask_gt is (16, 22, 1), 
    to resolve different shape between gt_labels and gt_bboxes.
    """

    # Decode predicted bboxes
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

    # Assign targets to anchors
    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)

    # 

