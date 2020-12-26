# -*- coding: utf-8 -*-
# @Time    : 2020/11/6
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import copy
from datetime import datetime
from pprint import pprint

import torch
from better_config import arg_config, backbone_config, proj_root
from tqdm import tqdm

import network as network_lib
from data.load_rgb import create_loader
from utils.misc import construct_print, initialize_seed_cudnn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, data_loader):
    model.eval()
    tqdm_iter = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    elapsed_time_s_list = []
    for test_batch_id, test_data in tqdm_iter:
        tqdm_iter.set_description(f"te=>{test_batch_id + 1} ")
        with torch.no_grad():
            curr_jpegs = test_data["image"].to(DEVICE, non_blocking=True)

            # https://pytorch.org/docs/stable/notes/cuda.html#asynchronous
            # -execution
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            model_output = model(data=dict(image=curr_jpegs))
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start_event.elapsed_time(end_event)
            elapsed_time_s_list.append(elapsed_time_ms / 1000)
    return len(elapsed_time_s_list) / sum(elapsed_time_s_list)


def main():
    construct_print(f"{datetime.now()}: Initializing...")
    construct_print(f"Project Root: {proj_root}")
    pprint(arg_config)

    initialize_seed_cudnn(seed=0, use_cudnn_benchmark=False)

    network_realname = arg_config["model"]
    if hasattr(network_lib, network_realname):
        model_cfg = copy.deepcopy(arg_config["model_cfg"])
        backbone_name = model_cfg["backbone"]
        model_cfg["backbone_cfg"] = backbone_config[backbone_name]
        model = getattr(network_lib, network_realname)(model_cfg).to(DEVICE)
    else:
        raise Exception("Please add the network into the __init__.py.")

    for data_name, data_path in arg_config["data"]["te_data_list"].items():
        te_loader, te_length = create_loader(data_path=data_path, training=False, shuffle=False)
        construct_print(f"Testing with testset: {data_name} with {te_length} samples")
        fps = test(model=model, data_loader=te_loader)
        construct_print(f"{fps} FPS")


if __name__ == "__main__":
    main()
