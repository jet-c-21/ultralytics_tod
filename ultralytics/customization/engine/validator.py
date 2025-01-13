import json

import torch

from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


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
                self.loss += model.loss(batch, preds)[1]
                pass

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
