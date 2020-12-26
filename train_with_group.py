# -*- coding: utf-8 -*-
import os
from datetime import datetime
from pprint import pprint

import torch
from torch.cuda import amp

import network as network_lib
from config import arg_config, loss_config, optimizer_config, proj_root, scheduler_config
from data.load_rgb_with_group import create_loader
from eval.test_model import test, val
from loss import get_loss_combination_with_cfg
from utils.misc import (
    construct_exp_name,
    construct_path,
    construct_print,
    initialize_seed_cudnn,
    make_log,
    pre_copy,
    pre_mkdir,
)
from utils.recorder import AvgMeter, CustomizedTimer, TBRecorder
from utils.tools_funcs import make_optim_with_cfg, make_scheduler_with_cfg, resume_checkpoint, save_checkpoint

_DEVICES = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def total_loss(seg_logits, seg_gts):
    loss_list = []
    loss_str_list = []

    seg_logits = seg_logits.float()

    for loss_name, loss_func in loss_funcs.items():
        loss = loss_func(seg_logits=seg_logits, seg_gts=seg_gts)
        loss_list.append(loss)
        loss_str_list.append(f"{loss_name}: {loss.item():.5f}")

    train_loss = sum(loss_list)
    return train_loss, loss_str_list


@CustomizedTimer(cus_msg="Train An Epoch")
def train_epoch(curr_epoch):
    construct_print(f"Exp_Name: {exp_name}")

    loss_record = AvgMeter()
    num_iter_per_epoch = len(tr_loader)
    for curr_iter_in_epoch, data in enumerate(tr_loader):
        curr_iter = curr_epoch * num_iter_per_epoch + curr_iter_in_epoch

        curr_jpegs = data["image"].to(_DEVICES, non_blocking=True)
        curr_masks = data["mask"].to(_DEVICES, non_blocking=True)
        with amp.autocast(enabled=arg_config["use_amp"]):
            preds = model(data=dict(image=curr_jpegs))
            seg_logits = preds["seg"]

        loss, loss_item_list = total_loss(seg_logits=seg_logits, seg_gts=curr_masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        iter_loss = loss.item()
        batch_size = curr_jpegs.size(0)
        loss_record.update(iter_loss, batch_size)

        if arg_config["tb_update"] > 0 and curr_iter % arg_config["tb_update"] == 0:
            tb_recorder.record_curve("loss_avg", loss_record.avg, curr_iter)
            tb_recorder.record_curve("iter_loss", iter_loss, curr_iter)
            tb_recorder.record_curve("lr", optimizer.param_groups, curr_iter)
            tb_recorder.record_image("img", curr_jpegs, curr_iter)
            tb_recorder.record_image("msk", curr_masks, curr_iter)
            tb_recorder.record_image("seg", seg_logits.sigmoid(), curr_iter)

        if arg_config["print_freq"] > 0 and curr_iter % arg_config["print_freq"] == 0:
            lr_string = ",".join([f"{x:.7f}" for x in scheduler.get_last_lr()])
            log = (
                f"[{curr_iter_in_epoch}:{num_iter_per_epoch},{curr_iter}:"
                f"{num_iter},"
                f"{curr_epoch}:{end_epoch}][{list(curr_jpegs.shape)}]"
                f"[Lr:{lr_string}]"
                f"[M:{loss_record.avg:.5f},C:{iter_loss:.5f}]"
                f"{loss_item_list}"
            )
            print(log)
            make_log(path_dict["tr_log"], log)

        if scheduler_usebatch:
            scheduler.step()


def train(val_loader=None):
    for curr_epoch in range(start_epoch, end_epoch):
        if val_loader is not None:
            seg_results = val(model=model, data_loader=val_loader)
            msg = f"Epoch: {curr_epoch}, Results on the valset:\n{seg_results}"
            print(msg)
            make_log(path_dict["te_log"], msg)

        model.train()
        train_epoch(curr_epoch=curr_epoch)

        # 根据周期修改学习率
        if not scheduler_usebatch:
            scheduler.step()

        # 每个周期都进行保存测试，保存的是针对第curr_epoch+1周期的参数
        save_checkpoint(
            exp_name=exp_name,
            model=model,
            current_epoch=curr_epoch + 1,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            full_net_path=path_dict["final_full_net"],
            state_net_path=path_dict["final_state_net"],
            total_epoch=end_epoch,
            save_num_models=arg_config["save_num_models"],
        )


construct_print(f"{datetime.now()}: Initializing...")
construct_print(f"Project Root: {proj_root}")
pprint(arg_config, indent=2, width=99)

# construct exp_name
exp_name = construct_exp_name(arg_dict=arg_config, extra_dicts=[optimizer_config, scheduler_config])
path_dict = construct_path(proj_root=proj_root, exp_name=exp_name)

initialize_seed_cudnn(
    seed=0,
    use_cudnn_benchmark=False if arg_config["use_mstrain"] else arg_config["use_cudnn_benchmark"],
)
pre_mkdir(path_config=path_dict)
pre_copy(
    main_file_path=__file__,
    all_config=dict(args=arg_config, path=path_dict, opti=optimizer_config, sche=scheduler_config),
    proj_root=proj_root,
)

if arg_config["tb_update"] > 0:
    tb_recorder = TBRecorder(path_dict["tb"])

tr_loader, tr_length = create_loader(
    data_path=arg_config["data"]["tr"],
    in_size=arg_config["in_size"]["tr"],
    batch_size=arg_config["batch_size"]["tr"],
    num_workers=arg_config["num_workers"],
    base_seed=arg_config["base_seed"],
    training=True,
    shuffle=True,
    use_mstrain=arg_config["use_mstrain"],
    get_length=True,
)
construct_print(f"Total length of the trainset is {tr_length}")

end_epoch = arg_config["epoch_num"]
num_iter = end_epoch * len(tr_loader)

network_realname = arg_config["model"]
if hasattr(network_lib, network_realname):
    model = getattr(network_lib, network_realname)().to(_DEVICES)
else:
    raise Exception("Please add the network into the __init__.py.")

loss_funcs = get_loss_combination_with_cfg(loss_cfg=loss_config)
optimizer = make_optim_with_cfg(model=model, optimizer_cfg=optimizer_config)
scheduler_usebatch = scheduler_config["sche_usebatch"]
scheduler = make_scheduler_with_cfg(
    optimizer=optimizer, total_num=num_iter if scheduler_usebatch else end_epoch, scheduler_cfg=scheduler_config
)

# Creates a GradScaler once at the beginning of training.
scaler = amp.GradScaler(enabled=arg_config["use_amp"])

if arg_config["resume"]:
    start_epoch = resume_checkpoint(
        exp_name=exp_name,
        load_path=path_dict["final_full_net"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        mode="all",
        force_load=True,
    )
else:
    start_epoch = 0

if start_epoch != end_epoch:
    if arg_config["has_val"]:
        val_data_name, val_data_path = arg_config["data"]["val"]
        construct_print(f"We will validate the model per epoch with {val_data_name}.")
        val_loader = create_loader(
            data_path=val_data_path,
            in_size=arg_config["in_size"]["val"],
            batch_size=arg_config["batch_size"]["val"],
            num_workers=arg_config["num_workers"],
            base_seed=arg_config["base_seed"],
            training=False,
            shuffle=False,
        )
    else:
        construct_print("We will not validate the model.")
        val_loader = None
    train(val_loader=val_loader)
else:
    if not arg_config["resume"]:
        resume_checkpoint(exp_name=exp_name, load_path=path_dict["final_state_net"], model=model, mode="onlynet")

if arg_config["has_test"]:
    for te_data_name, te_data_path in arg_config["data"]["te"]:
        construct_print(f"Testing with testset: {te_data_name}")
        te_loader = create_loader(
            data_path=te_data_path,
            in_size=arg_config["in_size"]["te"],
            batch_size=arg_config["batch_size"]["te"],
            num_workers=arg_config["num_workers"],
            base_seed=arg_config["base_seed"],
            training=False,
            shuffle=False,
        )
        pred_save_path = os.path.join(path_dict["save"], te_data_name)
        seg_results = test(model=model, save_path=pred_save_path, data_loader=te_loader)
        msg = f"Results on the testset({te_data_name}:'{te_data_path['root']}'):\n{seg_results}"
        print(msg)
        make_log(path_dict["te_log"], msg)

construct_print(f"{datetime.now()}: End training...")
