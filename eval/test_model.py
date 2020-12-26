# -*- coding: utf-8 -*-
# @Time    : 2020/11/25
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import os

import cv2
import numpy as np
import torch
import ttach as tta
from tqdm import tqdm

from data.utils import read_binary_array
from utils.misc import construct_print
from utils.recorder import CustomizedTimer, GroupedMetricRecorder
from utils.tools_funcs import clip_to_normalize, save_array_as_image

_DEVICES = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_aug(model, data, transforms):
    if transforms is None:
        model_output = model(data=data)
        outputs = model_output["seg"]
    else:
        auged_outputs = []
        for transformer in transforms:
            # augment image
            aug_data = {name: transformer.augment_image(data_item) for name, data_item in data.items()}
            # pass to model
            model_output = model(data=aug_data)
            # reverse augmentation
            deaug_logits = transformer.deaugment_mask(model_output["seg"])
            # save results
            auged_outputs.append(deaug_logits)
        # reduce results as you want, e.g mean/max/min
        outputs = torch.mean(torch.stack(auged_outputs, dim=0), dim=0)
    return outputs


@CustomizedTimer(cus_msg="Test")
@torch.no_grad()
def test(model, data_loader, save_path, use_tta=False, clip_range=None):
    model.eval()
    group_metric_recorder = GroupedMetricRecorder()

    if use_tta:
        construct_print("We will use Test Time Augmentation!")
        transforms = tta.Compose(
            [  # 2*3
                tta.HorizontalFlip(),
                tta.Scale(scales=[0.75, 1, 1.5], interpolation="bilinear", align_corners=False),
            ]
        )
    else:
        transforms = None

    tqdm_iter = tqdm(enumerate(data_loader), total=len(data_loader), leave=False, ncols=80)
    for batch_id, batch in tqdm_iter:
        tqdm_iter.set_description(f"te=>{batch_id + 1} ")

        images = batch["image"].to(_DEVICES, non_blocking=True)
        seg_logits = test_aug(model=model, data=dict(image=images), transforms=transforms)
        seg_probs = seg_logits.sigmoid().squeeze(1).cpu().detach().numpy()
        # float32 N,H,W (C has been removed)

        hs, ws = batch["image_info"]["ori_shape"]
        for i, seg_pred in enumerate(seg_probs):
            mask_path = batch["image_info"]["mask_path"][i]
            mask_array = read_binary_array(mask_path)
            mask_h, mask_w = mask_array.shape
            image_h, image_w = hs[i].item(), ws[i].item()

            seg_pred = cv2.resize(seg_pred, dsize=(image_w, image_h), interpolation=cv2.INTER_LINEAR)  # 0~1
            if (image_h, image_w) != (mask_h, mask_w):
                tqdm.write(
                    f"!!! image {(image_h, image_w)} and mask {(mask_h, mask_w)} have different size:" f" {mask_path}"
                )
                # h, w 这里以mask为准
                seg_pred = seg_pred[:mask_h, :mask_w]
                assert seg_pred.shape == mask_array.shape

            if clip_range is not None:
                seg_pred = clip_to_normalize(seg_pred, clip_range=clip_range)

            group_name = batch["image_info"]["group_name"][i]
            if save_path:  # 这里的save_path包含了数据集名字
                save_path_with_group_name = os.path.join(save_path, group_name)
                pred_name = batch["image_info"]["mask_name"][i]
                save_array_as_image(data_array=seg_pred, save_name=pred_name, save_dir=save_path_with_group_name)

            seg_pred = (seg_pred * 255).astype(np.uint8)
            group_metric_recorder.update(group_name=group_name, pre=seg_pred, gt=mask_array, gt_path=mask_path)
    fixed_seg_results = group_metric_recorder.show()
    return fixed_seg_results


@CustomizedTimer(cus_msg="Val")
@torch.no_grad()
def val(model, data_loader, clip_range=None):
    model.eval()
    group_metric_recorder = GroupedMetricRecorder()

    tqdm_iter = tqdm(enumerate(data_loader), total=len(data_loader), leave=False, ncols=80)
    for batch_id, batch in tqdm_iter:
        tqdm_iter.set_description(f"te=>{batch_id + 1} ")

        images = batch["image"].to(_DEVICES, non_blocking=True)
        seg_logits = test_aug(model=model, data=dict(image=images), transforms=None)
        seg_probs = seg_logits.sigmoid().squeeze(1).cpu().detach().numpy()
        # float32 N,H,W (C has been removed)

        hs, ws = batch["image_info"]["ori_shape"]
        for i, seg_pred in enumerate(seg_probs):
            mask_path = batch["image_info"]["mask_path"][i]
            mask_array = read_binary_array(mask_path)
            mask_h, mask_w = mask_array.shape
            image_h, image_w = hs[i].item(), ws[i].item()

            seg_pred = cv2.resize(seg_pred, dsize=(image_w, image_h), interpolation=cv2.INTER_LINEAR)  # 0~1
            if (image_h, image_w) != (mask_h, mask_w):
                tqdm.write(
                    f"!!! image {(image_h, image_w)} and mask {(mask_h, mask_w)} have different size:" f" {mask_path}"
                )
                # h, w 这里以mask为准
                seg_pred = seg_pred[:mask_h, :mask_w]
                assert seg_pred.shape == mask_array.shape

            if clip_range is not None:
                seg_pred = clip_to_normalize(seg_pred, clip_range=clip_range)

            group_name = batch["image_info"]["group_name"][i]
            seg_pred = (seg_pred * 255).astype(np.uint8)
            group_metric_recorder.update(group_name=group_name, pre=seg_pred, gt=mask_array, gt_path=mask_path)
    fixed_seg_results = group_metric_recorder.show()
    return fixed_seg_results
