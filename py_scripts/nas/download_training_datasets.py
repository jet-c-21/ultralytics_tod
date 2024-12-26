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


from third_party_packages.ichase_utils.ftp import get_vital_data_nas_client
from third_party_packages.ichase_utils.ichase_data.aibr import AIBRDeviceData


def main():
    # aibr_ftp_tool = AIBRFTPTool(log_path=None)
    # print(aibr_ftp_tool)

    location_id = "USA-KY001-1"

    add_bg_images = False

    vd_nas = get_vital_data_nas_client()
    aibr_dd = AIBRDeviceData(location_id=location_id, nas=vd_nas)
    aibr_dd.pull_data_and_build_training_ds(
        data_count=200,
        add_bg_images=add_bg_images,
        specific_ds_name=f"{location_id}_sampled",
    )


if __name__ == '__main__':
    main()
