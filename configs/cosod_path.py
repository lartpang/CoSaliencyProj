# -*- coding: utf-8 -*-
# @Time    : 2020/12/3
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os

_CoSOD_ROOT = "/home/lart/Datasets/Saliency/CoSOD"

COCO9213 = dict(
    root=os.path.join(_CoSOD_ROOT, "COCO9213-os"),
    image=dict(path=os.path.join(_CoSOD_ROOT, "COCO9213-os", "img"), suffix=".png"),
    mask=dict(path=os.path.join(_CoSOD_ROOT, "COCO9213-os", "gt"), suffix=".png"),
)
CoCA = dict(
    root=os.path.join(_CoSOD_ROOT, "CoCA"),
    image=dict(path=os.path.join(_CoSOD_ROOT, "CoCA", "image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_CoSOD_ROOT, "CoCA", "binary"), suffix=".png"),
    bbox=dict(path=os.path.join(_CoSOD_ROOT, "CoCA", "bbox"), suffix=".txt"),
    instance=dict(path=os.path.join(_CoSOD_ROOT, "CoCA", "instance"), suffix=".png"),
)
CoSal2015 = dict(
    root=os.path.join(_CoSOD_ROOT, "CoSal2015"),
    image=dict(path=os.path.join(_CoSOD_ROOT, "CoSal2015", "Image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_CoSOD_ROOT, "CoSal2015", "GroundTruth"), suffix=".png"),
)
CoSOD3k = dict(
    root=os.path.join(_CoSOD_ROOT, "CoSOD3k"),
    image=dict(path=os.path.join(_CoSOD_ROOT, "CoSOD3k", "Image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_CoSOD_ROOT, "CoSOD3k", "GroundTruth"), suffix=".png"),
    bbox=dict(path=os.path.join(_CoSOD_ROOT, "CoSOD3k", "BoundingBox"), suffix=".txt"),
    instance=dict(path=os.path.join(_CoSOD_ROOT, "CoSOD3k", "SegmentationObject"), suffix=".png"),
)
iCoSeg = dict(
    root=os.path.join(_CoSOD_ROOT, "iCoSeg"),
    image=dict(path=os.path.join(_CoSOD_ROOT, "iCoSeg", "Image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_CoSOD_ROOT, "iCoSeg", "GroundTruth"), suffix=".png"),
)
ImagePair = dict(
    root=os.path.join(_CoSOD_ROOT, "ImagePair"),
    image=dict(path=os.path.join(_CoSOD_ROOT, "ImagePair", "Image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_CoSOD_ROOT, "ImagePair", "GroundTruth"), suffix=".png"),
)
MSRC = dict(
    root=os.path.join(_CoSOD_ROOT, "MSRC"),
    image=dict(path=os.path.join(_CoSOD_ROOT, "MSRC", "Image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_CoSOD_ROOT, "MSRC", "GroundTruth"), suffix=".png"),
)
WICOS = dict(
    root=os.path.join(_CoSOD_ROOT, "WICOS"),
    image=dict(path=os.path.join(_CoSOD_ROOT, "WICOS", "Image"), suffix=".jpg"),
    mask=dict(path=os.path.join(_CoSOD_ROOT, "WICOS", "GroundTruth"), suffix=".png"),
)
