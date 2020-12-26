# -*- coding: utf-8 -*-
# @Time    : 2020/8/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import json
import os

import cv2
import mmcv
import numpy as np
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())


def read_data_dict_from_dir(dir_path: dict) -> dict:
    img_dir = dir_path["image"]["path"]
    img_suffix = dir_path["image"]["suffix"]

    if dir_path.get("mask"):
        has_mask_data = True
        mask_dir = dir_path["mask"]["path"]
        mask_suffix = dir_path["mask"]["suffix"]
    else:
        has_mask_data = False

    if dir_path.get("edge"):
        has_edge_data = True
        edge_dir = dir_path["edge"]["path"]
        edge_suffix = dir_path["edge"]["suffix"]
    else:
        has_edge_data = False

    if dir_path.get("hotspot"):
        has_hs_data = True
        hs_dir = dir_path["hotspot"]["path"]
        hs_suffix = dir_path["hotspot"]["suffix"]
    else:
        has_hs_data = False

    if dir_path.get("cam"):
        has_cam_data = True
        cam_dir = dir_path["cam"]["path"]
        cam_suffix = dir_path["cam"]["suffix"]
    else:
        has_cam_data = False

    total_image_path_list = []
    total_mask_path_list = []
    total_edge_path_list = []
    total_hs_path_list = []
    total_cam_path_list = []

    name_list_from_img_dir = [x[:-4] for x in os.listdir(img_dir)]
    if has_mask_data:
        name_list_from_mask_dir = [x[:-4] for x in os.listdir(mask_dir)]
        image_name_list = sorted(list(set(name_list_from_img_dir).intersection(set(name_list_from_mask_dir))))
    else:
        image_name_list = name_list_from_img_dir
    for idx, image_name in enumerate(image_name_list):
        total_image_path_list.append(dict(path=os.path.join(img_dir, image_name + img_suffix), idx=idx))
        if has_mask_data:
            total_mask_path_list.append(dict(path=os.path.join(mask_dir, image_name + mask_suffix), idx=idx))
        if has_edge_data:
            total_edge_path_list.append(dict(path=os.path.join(edge_dir, image_name + edge_suffix), idx=idx))
        if has_hs_data:
            total_hs_path_list.append(dict(path=os.path.join(hs_dir, image_name + hs_suffix), idx=idx))
        if has_cam_data:
            total_cam_path_list.append(dict(path=os.path.join(cam_dir, image_name + cam_suffix), idx=idx))

    return dict(
        root=dir_path["root"],
        image=total_image_path_list,
        mask=total_mask_path_list,
        edge=total_edge_path_list,
        hs=total_hs_path_list,
        cam=total_cam_path_list,
    )


def read_data_list_form_txt(path: str) -> list:
    line_list = []
    with open(path, encoding="utf-8", mode="r") as f:
        line = f.readline()
        while line:
            line_list.append(line.strip())
            line = f.readline()
    return line_list


def read_data_dict_from_json(json_path: str) -> dict:
    with open(json_path, mode="r", encoding="utf-8") as openedfile:
        data_info = json.load(openedfile)
    return data_info


def read_color_array(path: str):
    assert path.endswith(".jpg") or path.endswith(".png")
    bgr_array = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    return rgb_array


def _flow_to_direction_and_magnitude(flow, unknown_thr=1e6):
    """Convert flow map to RGB image.

    Args:
        flow (ndarray): Array of optical flow.
        unknown_thr (str): Values above this threshold will be marked as
            unknown and thus ignored.

    Returns:
        ndarray: RGB image that can be visualized.
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    color_wheel = mmcv.make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) | (np.abs(dy) > unknown_thr)
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    flow_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(flow_magnitude > np.finfo(float).eps):
        max_rad = np.max(flow_magnitude)
        dx /= max_rad
        dy /= max_rad

    flow_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    flow_direction = np.arctan2(-dy, -dx) / np.pi  # -1,1

    bin_real = (flow_direction + 1) / 2 * (num_bins - 1)  # [0,num_bins-1)
    bin_left = np.floor(bin_real).astype(int)
    bin_right = (bin_left + 1) % num_bins
    w = (bin_real - bin_left.astype(np.float32))[..., None]
    flow_img = (1 - w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]
    direction_map = flow_img.copy()
    small_ind = flow_magnitude <= 1
    flow_img[small_ind] = 1 - flow_magnitude[small_ind, None] * (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75
    flow_img[ignore_inds, :] = 0

    return dict(flow=flow_img, direction=direction_map, magnitude=flow_magnitude)


def read_flow_array(path: str, return_info, to_normalize=False):
    """
    :param path:
    :param return_info:
    :param to_normalize:
    :return: 0~1
    """
    assert path.endswith(".flo")
    flow_array = mmcv.flowread(path)
    split_flow = _flow_to_direction_and_magnitude(flow_array)
    if not isinstance(return_info, (tuple, list)):
        return_info = [return_info]

    return_array = dict()
    for k in return_info:
        data_array = split_flow[k]
        if k == "magnitude" and to_normalize:
            data_array = (data_array - data_array.min()) / (data_array.max() - data_array.min())
        return_array[k] = data_array
    return return_array


def read_binary_array(path: str, to_normalize: bool = False, thr: float = -1) -> np.ndarray:
    """
    1. read the binary image with the suffix `.jpg` or `.png`
        into a grayscale ndarray
    2. (to_normalize=True) rescale the ndarray to [0, 1]
    3. (thr >= 0) binarize the ndarray with `thr`
    4. return a gray ndarray (np.float32)
    """
    assert path.endswith(".jpg") or path.endswith(".png")
    gray_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if to_normalize:
        gray_array = gray_array.astype(np.float32)
        gray_array_min = gray_array.min()
        gray_array_max = gray_array.max()
        if gray_array_max != gray_array_min:
            gray_array = (gray_array - gray_array_min) / (gray_array_max - gray_array_min)
        else:
            gray_array /= 255

    if thr >= 0:
        gray_array = (gray_array > thr).astype(np.float32)

    return gray_array
