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
from aibr_dsb.ichase_utils import var as VAR
from aibr_dsb.ichase_utils.var import aibr as AIBR
from aibr_dsb.ichase_utils.notify_tool.line import send_message_by_line_notify
from aibr_dsb.ichase_utils.file_tool import dt_timedelta_to_day_hms_str, chmod_777_r

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# last_pt_path = '/usr/src/ultralytics/runs/detect/train6/weights/last.pt'
# pt_path = 'GPL.0.0.0_EPOCH-300.pt' # existed pt
# model = YOLO('yolov8n.yaml').load(pt_path)

# model = YOLO('yolov8x.pt', task='detect')

model_yaml_name = 'yolov8n-gplNano.yaml'
# model_yaml_name = 'yolov8n.yaml'
model_yaml_path = AIBR.ULTRALYTICS_MODEL_CFG_DIR / model_yaml_name

gpl_open_src_pt_path = AIBR.MODEL_WEIGHTS_ENGINES_DIR / 'GPL-OPEN-SRC_V1_VmAP-50-322.pt'
# model = YOLO(gpl_open_src_pt_path, task='detect')
# model = YOLO('yolov8n-p6.yaml', task='detect').load(gpl_open_src_pt_path)
# model = YOLO(str(model_yaml_path), task='detect').load(gpl_open_src_pt_path)
model = YOLO(str(model_yaml_path), task='detect')  # train a brand-new model

data_yaml_name = 'gpl-field_YOLO-DS_2023-12.yaml'
# data_yaml_name = 'E001-1_DC-69_SEED-369_2023-12-20-18-24-23.yaml'
data_yaml = VAR.PROJECT_DIR / data_yaml_name

if __name__ == '__main__':
    training_task = 'GPL Nano Model Training'

    msg = f"[{training_task}] - {AIBR.EMOJI.ROCKET} [Start] {AIBR.EMOJI.ROCKET}"
    print(msg)
    send_message_by_line_notify(msg)

    print(model.info(detailed=True))

    st = datetime.datetime.now()
    # Train the model
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
        optimizer='auto',
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
    cost_time_str = dt_timedelta_to_day_hms_str(cost_time)

    msg = f"[{training_task}] - {AIBR.EMOJI.MODEL_TRAIN_FIN} [FINISH] {AIBR.EMOJI.MODEL_TRAIN_FIN}\n" \
          f"Training Cost Time: {cost_time_str}"
    print(msg)
    send_message_by_line_notify(msg)

    chmod_777_r(VAR.PROJECT_DIR / 'runs')
