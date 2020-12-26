# -*- coding: utf-8 -*-
# @Time    : 2020/10/5
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os

_sod_root = "/home/lart/Datasets/Saliency/RGBSOD"

ECSSD_te_paths = dict(
    root=os.path.join(_sod_root, "ECSSD"),
    image=dict(path=os.path.join(_sod_root, "ECSSD", "Image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_sod_root, "ECSSD", "Mask"), suffix=".png"),
)
DUTOMRON_te_paths = dict(
    root=os.path.join(_sod_root, "DUT-OMRON"),
    image=dict(path=os.path.join(_sod_root, "DUT-OMRON", "Image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_sod_root, "DUT-OMRON", "Mask"), suffix=".png"),
)
HKUIS_te_paths = dict(
    root=os.path.join(_sod_root, "HKU-IS"),
    image=dict(path=os.path.join(_sod_root, "HKU-IS", "Image"), suffix=".png"),
    mask=dict(path=os.path.join(_sod_root, "HKU-IS", "Mask"), suffix=".png"),
)
PASCALS_te_paths = dict(
    root=os.path.join(_sod_root, "PASCAL-S"),
    image=dict(path=os.path.join(_sod_root, "PASCAL-S", "Image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_sod_root, "PASCAL-S", "Mask"), suffix=".png"),
)
SOC_te_paths = dict(
    root=os.path.join(_sod_root, "SOC/Test"),
    image=dict(path=os.path.join(_sod_root, "SOC/Test", "Image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_sod_root, "SOC/Test", "Mask"), suffix=".png"),
)
DUTS_te_paths = dict(
    root=os.path.join(_sod_root, "DUTS/Test"),
    image=dict(path=os.path.join(_sod_root, "DUTS/Test", "Image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_sod_root, "DUTS/Test", "Mask"), suffix=".png"),
)
DUTS_tr_paths = dict(
    root=os.path.join(_sod_root, "DUTS/Train"),
    image=dict(path=os.path.join(_sod_root, "DUTS/Train", "Image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_sod_root, "DUTS/Train", "Mask"), suffix=".png"),
)
