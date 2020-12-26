# -*- coding: utf-8 -*-
import os
from datetime import datetime
from pprint import pprint

import cv2
import numpy as np
import torch
import ttach as tta
from tqdm import tqdm

import network as network_lib
from config import arg_config, proj_root
from data.load_rgb_with_group import create_loader
from data.utils import read_binary_array
from utils.misc import construct_print, initialize_seed_cudnn
from utils.recorder import MetricRecorder
from utils.tools_funcs import clip_to_normalize, resume_checkpoint, save_array_as_image

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


@torch.no_grad()
def test(model, data_loader, save_path):
    model.eval()
    cal_total_seg_metrics = MetricRecorder()

    if arg_config["use_tta"]:
        construct_print("We will use Test Time Augmentation!")
        transforms = tta.Compose(
            [  # 2*3
                tta.HorizontalFlip(),
                tta.Scale(scales=[0.75, 1, 1.5], interpolation="bilinear", align_corners=False),
            ]
        )
    else:
        transforms = None

    tqdm_iter = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    for batch_id, batch in tqdm_iter:
        tqdm_iter.set_description(f"te=>{batch_id + 1} ")

        images = batch["image"].to(_DEVICES, non_blocking=True)
        seg_logits = test_aug(model=model, data=dict(image=images), transforms=transforms)
        seg_probs = seg_logits.sigmoid().squeeze(1).cpu().detach().numpy()

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

            if arg_config["clip_range"] is not None:
                seg_pred = clip_to_normalize(seg_pred, clip_range=arg_config["clip_range"])

            if save_path:  # 这里的save_path包含了数据集名字
                save_path_with_group_name = os.path.join(save_path, batch["image_info"]["group_name"][i])
                pred_name = batch["image_info"]["mask_name"][i]
                save_array_as_image(data_array=seg_pred, save_name=pred_name, save_dir=save_path_with_group_name)

            seg_pred = (seg_pred * 255).astype(np.uint8)
            # 先对group内部计算平均指标，之后再整体平均，这和直接整体平均结果是一样的，但是并不利于统计分析不同组内部的情况
            # TODO: 有必要对其改进
            cal_total_seg_metrics.step(seg_pred, mask_array, mask_path)
    fixed_seg_results = cal_total_seg_metrics.get_results()
    return fixed_seg_results


def main():
    construct_print(f"{datetime.now()}: Initializing...")
    construct_print(f"Project Root: {proj_root}")
    pprint(arg_config)

    initialize_seed_cudnn(seed=0, use_cudnn_benchmark=False)

    network_realname = arg_config["model"]
    if hasattr(network_lib, network_realname):
        model = getattr(network_lib, network_realname)().to(_DEVICES)
    else:
        raise Exception("Please add the network into the __init__.py.")

    resume_checkpoint(exp_name="", load_path=params_path, model=model, mode="onlynet")
    with open(recorder_txt_path, encoding="utf-8", mode="a") as f:
        f.write(f"{params_path}" + "\n")

    for data_name, data_path in arg_config["data"]["te"]:
        construct_print(f"Testing with testset: {data_name}")
        te_loader = create_loader(
            data_path=data_path,
            in_size=arg_config["in_size"]["te"],
            batch_size=arg_config["batch_size"],
            num_workers=arg_config["num_workers"],
            base_seed=arg_config["base_seed"],
            training=False,
            shuffle=False,
        )
        pred_save_path = os.path.join(save_root, data_name)
        seg_results = test(model=model, save_path=pred_save_path, data_loader=te_loader)
        msg = f"Results on the testset({data_name}:'{data_path['root']}'):\n{seg_results}"
        print(msg)
        with open(recorder_txt_path, encoding="utf-8", mode="a") as f:
            f.write(msg + "\n")
    construct_print(f"{datetime.now()}: End training...")


if __name__ == "__main__":
    params_path = ""
    save_root = os.path.join(os.path.dirname(os.path.dirname(params_path)), "test")

    recorder_txt_path = os.path.join(os.path.dirname(os.path.dirname(params_path)), "re_eval.txt")
    with open(recorder_txt_path, encoding="utf-8", mode="a") as f:
        f.write(f"{datetime.now()}\n")
    main()
