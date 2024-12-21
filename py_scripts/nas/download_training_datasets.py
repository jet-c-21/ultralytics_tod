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

from thirdparty_packages.ichase_utils import var as VAR
from thirdparty_packages.ichase_utils.ftp.aibr import AIBRFTPTool

def main():
    pass

if __name__ == '__main__':
    main()