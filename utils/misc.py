# -*- coding: utf-8 -*-
import copy
import json
import os
import random
import shutil
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch


def prepare_for_running(user_cfg):
    initialize_seed_cudnn(
        seed=0,
        use_cudnn_benchmark=False if user_cfg.train_config.use_mstrain else user_cfg.train_config.use_cudnn_benchmark,
    )
    pre_mkdir(path_config=user_cfg.path)
    pre_copy(
        main_file_path=user_cfg.path.main,
        all_config=user_cfg,
        proj_root=user_cfg.proj_root,
    )


def set_seed_for_lib(seed):
    random.seed(seed)
    np.random.seed(seed)
    # 为了禁止hash随机化，使得实验可复现。
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def initialize_seed_cudnn(seed, use_cudnn_benchmark):
    assert isinstance(use_cudnn_benchmark, bool) and isinstance(seed, int)
    set_seed_for_lib(seed)
    torch.backends.cudnn.enabled = True
    if use_cudnn_benchmark:
        construct_print("We will use `torch.backends.cudnn.benchmark`")
    else:
        construct_print("We will not use `torch.backends.cudnn.benchmark`")
    torch.backends.cudnn.benchmark = use_cudnn_benchmark
    torch.backends.cudnn.deterministic = True


def pre_mkdir(path_config):
    # 提前创建好记录文件，避免自动创建的时候触发文件创建事件
    check_mkdir(path_config["pth_log"])
    make_log(path_config["te_log"], f"=== te_log {datetime.now()} ===")
    make_log(path_config["tr_log"], f"=== tr_log {datetime.now()} ===")

    # 提前创建好存储预测结果和存放模型以及tensorboard的文件夹
    check_mkdir(path_config["save"])
    check_mkdir(path_config["pth"])


def pre_copy(main_file_path, proj_root, all_config):
    path_config = all_config["path"]
    with open(path_config["all_cfg"], encoding="utf-8", mode="w") as f:
        json.dump(all_config, f, indent=2)
    shutil.copy(f"{proj_root}/config.py", path_config["cfg_copy"])
    shutil.copy(main_file_path, path_config["trainer_copy"])


def clip_normalize_scale(array, clip_min=0, clip_max=250, new_min=0, new_max=255):
    array = np.clip(array, a_min=clip_min, a_max=clip_max)
    array = (array - array.min()) / (array.max() - array.min())
    array = array * (new_max - new_min) + new_min
    return array


def check_mkdir(dir_name, delete_if_exists=False):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        if delete_if_exists:
            construct_print(f"{dir_name} will be re-created!!!")
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)


def check_if_file_exists(file_path):
    if not (os.path.exists(file_path) and os.path.isfile(file_path)):
        raise FileNotFoundError


def make_log(path, context):
    with open(path, "a") as log:
        log.write(f"{context}\n")


def construct_print(out_str: str, total_length: int = 80):
    if len(out_str) >= total_length:
        out_str = "[ ==>>\n" + out_str + "\n <<== ]"
    else:
        space_str = " " * ((total_length - len(out_str)) // 2 - 6)
        out_str = "[ ->> " + space_str + out_str + space_str + " <<- ]"
    print(out_str)


def construct_path(proj_root: str, exp_name: str) -> dict:
    ckpt_path = os.path.join(proj_root, "output")

    pth_log_path = os.path.join(ckpt_path, exp_name)
    tb_path = os.path.join(pth_log_path, "tb")
    save_path = os.path.join(pth_log_path, "pre")
    pth_path = os.path.join(pth_log_path, "pth")

    final_full_model_path = os.path.join(pth_path, "checkpoint_final.pth")
    final_state_path = os.path.join(pth_path, "state_final.pth")

    tr_log_path = os.path.join(pth_log_path, f"tr_{str(datetime.now())[:10]}.txt")
    te_log_path = os.path.join(pth_log_path, f"te_{str(datetime.now())[:10]}.txt")
    cfg_copy_path = os.path.join(pth_log_path, f"cfg_{str(datetime.now())}.txt")
    cfg_all_path = os.path.join(pth_log_path, f"cfg_{str(datetime.now())}.json")
    trainer_copy_path = os.path.join(pth_log_path, f"trainer_{str(datetime.now())}.txt")

    path_config = {
        "ckpt_path": ckpt_path,
        "pth_log": pth_log_path,
        "tb": tb_path,
        "save": save_path,
        "pth": pth_path,
        "final_full_net": final_full_model_path,
        "final_state_net": final_state_path,
        "tr_log": tr_log_path,
        "te_log": te_log_path,
        "cfg_copy": cfg_copy_path,
        "all_cfg": cfg_all_path,
        "trainer_copy": trainer_copy_path,
    }

    return path_config


def construct_exp_name(arg_dict: dict, extra_dicts: list):
    # bs_16_lr_0.05_e30_noamp_2gpu_noms_352
    focus_item = OrderedDict(
        {
            "in_size": "s",
            "batch_size": "bs",
            "epoch_num": "e",
            "use_amp": "amp",
            "lr": "lr",
            "lr_strategy": "lt",
            "optimizer": "op",
            "strategy": "ot",
            "use_mstrain": "ms",
            "info": "info",
        }
    )
    copy_arg_dict = copy.deepcopy(arg_dict)
    copy_extra_dicts = copy.deepcopy(extra_dicts)

    exp_name = f"{copy_arg_dict['model']}"
    for k, v in focus_item.items():
        if (item := copy_arg_dict.get(k, "NotExist")) == "NotExist":
            extra_dict = {}
            for temp_dict in copy_extra_dicts:
                extra_dict.update(temp_dict)
            if (item := extra_dict.get(k, "NotExist")) == "NotExist":
                raise ValueError(f"{item}: {k}")

        if k == "in_size" and isinstance(item, dict):
            tr_item = item["tr"]
            if tr_item.get("hw", False):
                item = tr_item["hw"]
            elif tr_item.get("h", False) and tr_item.get("w", False):
                item = f"{tr_item['h']}X{tr_item['w']}"
            else:
                raise ValueError
        if k == "batch_size" and isinstance(item, dict):
            item = item["tr"]

        if isinstance(item, bool):
            item = "Y" if item else "N"
        elif isinstance(item, (list, tuple)):
            item = "Y" if item else "N"  # 只是判断是否非空
        elif isinstance(item, str):
            if not item:
                continue
            if "_" in item:
                item = item.replace("_", "")
        elif item is None:
            item = "N"

        if isinstance(item, str):
            item = item.lower()
        exp_name += f"_{v.upper()}{item}"
    return exp_name


if __name__ == "__main__":
    print("=" * 8)
    out_str = "lartpang"
    construct_print(out_str, total_length=8)
