# coding: utf-8
"""
Author: Jet C.
GitHub: https://github.com/jet-c-21
Create Date: 2024-12-21
"""
# >>> Dynamic Changing `sys.path` in Runtime by Adding Project Directory to Path >>>
import pathlib
import sys

THIS_FILE_PATH = pathlib.Path(__file__).absolute()
THIS_FILE_PARENT_DIR = THIS_FILE_PATH.parent
PROJECT_DIR = THIS_FILE_PARENT_DIR.parent.parent
sys.path.append(str(PROJECT_DIR))
print(f"[*INFO*] - append directory to path: {PROJECT_DIR}")
# <<< Dynamic Changing `sys.path` in Runtime by Adding Project Directory to Path <<<


from ultralytics.customization.utils.checks import basic_environment_check

from ultralytics.customization.engine.validator import base_validator_call
from ultralytics.customization.utils.loss import v8_detection_loss_cal_loss_per_cls

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics import YOLO

from third_party_packages.ichase_utils.dataset_tool import YOLODataset
from third_party_packages.ichase_utils.file_tool import get_file_content_hash

basic_environment_check()

DetectionValidator._original_call = DetectionValidator.__call__
DetectionValidator.__call__ = base_validator_call
v8DetectionLoss.cal_loss_per_cls = v8_detection_loss_cal_loss_per_cls


def main():
    ds_yaml = PROJECT_DIR / "USA-KY001-1_sampled.yaml"
    assert ds_yaml.is_file(), f"Dataset file not found: {ds_yaml}"

    ds_dir = PROJECT_DIR / "train-datasets" / f"{ds_yaml.stem}"
    assert ds_dir.is_dir(), f"Dataset directory not found: {ds_dir}"

    yolo_ds = YOLODataset(ds_dir)
    print(f"[*INFO*] - {ds_dir.name} yolo dataset info:\n{yolo_ds.info_df}\n")
    print(f"[*INFO*] - all_image_paths_val_hash: {yolo_ds.get_all_images_hash()}")
    print(f"[*INFO*] - all_label_paths_val_hash: {yolo_ds.get_all_label_paths_val_hash()}\n")

    epochs = 100
    batch = 8
    lr0 = 1e-5
    random_state = 369

    model = YOLO("yolov8n.yaml")

    model.train(
        data=ds_yaml,
        epochs=epochs,
        lr0=lr0,
        batch=batch,
        seed=random_state,
        verbose=True,
    )


if __name__ == '__main__':
    main()
