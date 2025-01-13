import json
from typing import Dict, Tuple
import numpy as np
import torch

from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


def _batches_are_equal(batch1: Dict, batch2: Dict) -> bool:
    """
    this function is for debug

    Check if two batches are equal, including nested structures, tensors, and arrays.

    Args:
        batch1 (Dict): The first batch dictionary to compare.
        batch2 (Dict): The second batch dictionary to compare.

    Returns:
        bool: True if the batches are equal, False otherwise.
    """
    if batch1.keys() != batch2.keys():
        return False

    for key in batch1:
        val1, val2 = batch1[key], batch2[key]

        # Check if both are tensors
        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if not torch.equal(val1, val2):
                return False

        # Check if both are NumPy arrays
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                return False

        # Recursively check dictionaries
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if not _batches_are_equal(val1, val2):
                return False

        # Check lists
        elif isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                return False
            if not all(_batches_are_equal({i: v1}, {i: v2}) for i, (v1, v2) in enumerate(zip(val1, val2))):
                return False

        # Check tuples
        elif isinstance(val1, tuple) and isinstance(val2, tuple):
            if len(val1) != len(val2):
                return False
            if not all(_batches_are_equal({i: v1}, {i: v2}) for i, (v1, v2) in enumerate(zip(val1, val2))):
                return False

        # Fallback to direct comparison for other types
        else:
            if val1 != val2:
                return False

    return True


def _preds_are_equal(pred1: Tuple, pred2: Tuple) -> bool:
    """
    Check if two predictions are equal.

    Args:
        pred1 (Tuple): The first prediction tuple to compare.
        pred2 (Tuple): The second prediction tuple to compare.

    Returns:
        bool: True if the predictions are equal, False otherwise.
    """
    if len(pred1) != len(pred2):
        return False

    for val1, val2 in zip(pred1, pred2):
        # Check if both are tensors
        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if not torch.equal(val1, val2):
                return False

        # Check if both are NumPy arrays
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                return False

        # Recursively compare nested tuples
        elif isinstance(val1, tuple) and isinstance(val2, tuple):
            if not _preds_are_equal(val1, val2):
                return False

        # Check lists
        elif isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                return False
            if not all(_preds_are_equal((v1,), (v2,)) for v1, v2 in zip(val1, val2)):
                return False

        # Recursively compare dictionaries
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if val1.keys() != val2.keys():
                return False
            for key in val1:
                if not _preds_are_equal((val1[key],), (val2[key],)):
                    return False

        # Fallback to direct comparison for other types
        else:
            if val1 != val2:
                return False

    return True


@smart_inference_mode()
def base_validator_call(self: BaseValidator, trainer=None, model=None):
    """
    runtime override function for `ultralytics.engine.validator.BaseValidator.__call__()`

    Executes validation process, running inference on dataloader and computing performance metrics.
    """
    _msg = f"running customization BaseValidator.__call__() with class instance: {self}"
    LOGGER.info(_msg)

    self.training = trainer is not None
    augment = self.args.augment and (not self.training)
    if self.training:
        self.device = trainer.device
        self.data = trainer.data
        # force FP16 val during training
        self.args.half = self.device.type != "cpu" and trainer.amp
        model = trainer.ema.ema or trainer.model
        model = model.half() if self.args.half else model.float()
        # self.model = model
        self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
        self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
        model.eval()
    else:
        callbacks.add_integration_callbacks(self)
        model = AutoBackend(
            weights=model or self.args.model,
            device=select_device(self.args.device, self.args.batch),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
        )
        # self.model = model
        self.device = model.device  # update device
        self.args.half = model.fp16  # update half
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_imgsz(self.args.imgsz, stride=stride)
        if engine:
            self.args.batch = model.batch_size
        elif not pt and not jit:
            self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
            LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

        if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
            self.data = check_det_dataset(self.args.data)
        elif self.args.task == "classify":
            self.data = check_cls_dataset(self.args.data, split=self.args.split)
        else:
            raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
        if not pt:
            self.args.rect = False
        self.stride = model.stride  # used in get_dataloader() for padding
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

        model.eval()
        model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

    self.run_callbacks("on_val_start")
    dt = (
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
    )
    bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
    self.init_metrics(de_parallel(model))
    self.jdict = []  # empty before each val
    loss_per_cls = {}
    for batch_i, batch in enumerate(bar):
        self.run_callbacks("on_val_batch_start")
        self.batch_i = batch_i
        # Preprocess
        with dt[0]:
            batch = self.preprocess(batch)

        # Inference
        with dt[1]:
            preds = model(batch["img"], augment=augment)

        # Loss
        with dt[2]:
            if self.training:
                # import copy
                # orig_batch = copy.deepcopy(batch)
                # orig_preds = copy.deepcopy(preds)

                _detached_all_cls_loss = model.loss(batch, preds)[1]
                print(f"\n[*DEBUG*] - #{batch_i} _detached_all_cls_loss: {_detached_all_cls_loss}\n")
                self.loss += _detached_all_cls_loss

                if not hasattr(model, "cal_loss_per_cls"):
                    raise RuntimeError(
                        f"function attr: cal_loss_per_cls should be assigned to {model.__class__.__name__} in runtime.")

                # assert _batches_are_equal(batch, orig_batch)
                # assert _preds_are_equal(preds, orig_preds)

                # state 2 of batch and preds
                model.cal_loss_per_cls(batch, preds, existed_result_dict=loss_per_cls)

        # Postprocess
        with dt[3]:
            preds = self.postprocess(preds)

        self.update_metrics(preds, batch)
        if self.args.plots and batch_i < 3:
            self.plot_val_samples(batch, batch_i)
            self.plot_predictions(batch, preds, batch_i)

        self.run_callbacks("on_val_batch_end")
    stats = self.get_stats()
    self.check_stats(stats)
    self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
    self.finalize_metrics()
    self.print_results()
    self.run_callbacks("on_val_end")
    if self.training:
        model.float()
        results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
        return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
    else:
        LOGGER.info(
            "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                *tuple(self.speed.values())
            )
        )
        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / "predictions.json"), "w") as f:
                LOGGER.info(f"Saving {f.name}...")
                json.dump(self.jdict, f)  # flatten and save
            stats = self.eval_json(stats)  # update stats
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats
