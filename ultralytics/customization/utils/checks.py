# coding: utf-8
"""
Author: Jet C.
GitHub: https://github.com/jet-c-21
Create Date: 13/01/2025
"""


def basic_environment_check():
    """
    this function is an extra function not for overriden
    """
    import torch
    import ultralytics

    print(f"[*INFO*] - GPU available: {torch.cuda.is_available()}")
    print(f"[*INFO*] - imported ultralytics path: {ultralytics.__file__}")
    print(f"[*INFO*] - imported ultralytics version: {ultralytics.__version__}")
