# coding: utf-8
"""
Author: Jet C.
GitHub: https://github.com/jet-c-21
Create Date: 4/16/23

load p series:
https://github.com/ultralytics/ultralytics/issues/981
test on 2023-12-24

"""

import datetime
import math
import multiprocessing as mp
from pprint import pprint as pp

from ultralytics import YOLO

try:
    import ultralytics

    msg = f"Using Custom Ultralytics Version: {ultralytics.SUB_VERSION}"
    print(msg)
except:
    pass

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# last_pt_path = '/usr/src/ultralytics/runs/detect/train6/weights/last.pt'
# pt_path = 'GPL.0.0.0_EPOCH-300.pt' # existed pt
# model = YOLO('yolov8n.yaml').load(pt_path)

# model = YOLO('yolov8x.pt', task='detect')

# model_yaml_name = 'yolov8n-gplNano.yaml'
model_yaml_path = "yolov8n.yaml"

# model = YOLO(gpl_open_src_pt_path, task='detect')
# model = YOLO('yolov8n-p6.yaml', task='detect').load(gpl_open_src_pt_path)
# model = YOLO(str(model_yaml_path), task='detect').load(gpl_open_src_pt_path)
model = YOLO(str(model_yaml_path), task="detect")  # train a brand-new model

data_yaml = "B003-1_DC-3_SEED-369.yaml"
# data_yaml_name = 'E001-1_DC-69_SEED-369_2023-12-20-18-24-23.yaml'

if __name__ == "__main__":
    training_task = "GPL Nano Model Training"
    print(model.info(detailed=True))

    st = datetime.datetime.now()
    # Train the model
    # model -> Trainer -> ultralytics.nn.tasks.DetectionModel
    model.train(
        # resume=True,
        # augment=False,
        data=str(data_yaml),
        epochs=150,
        patience=300,
        imgsz=640,
        workers=mp.cpu_count(),
        batch=32,
        lr0=1e-3,
        # batch=16,
        # lr0=(1e-3) * math.sqrt(0.5),
        optimizer="auto",
        rect=True,
        plots=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.3,  # deg
        translate=0.15,
        scale=0.15,
        shear=1.0,
        flipud=0,
        fliplr=0.5,
        mosaic=1.0,
        perspective=0.0002,
        cache=True,
    )

    ed = datetime.datetime.now()
    cost_time = ed - st
    print(cost_time)
