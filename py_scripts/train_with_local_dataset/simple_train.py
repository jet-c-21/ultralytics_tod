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

# basic environment check
import torch
import ultralytics

print(f"[*INFO*] - GPU available: {torch.cuda.is_available()}")
print(f"[*INFO*] - imported ultralytics path: {ultralytics.__file__}")
print(f"[*INFO*] - imported ultralytics version: {ultralytics.__version__}")

from ultralytics import YOLO

from third_party_packages.ichase_utils.dataset_tool import YOLODataset


def main():
    ds_yaml = PROJECT_DIR / "USA-KY001-1_sampled.yaml"
    assert ds_yaml.is_file(), f"Dataset file not found: {ds_yaml}"

    ds_dir = PROJECT_DIR / "train-datasets" / f"{ds_yaml.stem}"
    assert ds_dir.is_dir(), f"Dataset directory not found: {ds_dir}"

    yolo_ds = YOLODataset(ds_dir)
    print(f"[*INFO*] - {ds_dir.name} yolo dataset info:\n{yolo_ds.info_df}\n")

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
    )


if __name__ == '__main__':
    main()
