# -*- coding: utf-8 -*-
import os

from configs import cosod_path

proj_root = os.path.dirname(__file__)

cosod_data = dict(
    tr=[
        ("coco9213", cosod_path.COCO9213),
    ],
    val=("msrc", cosod_path.MSRC),
    te=[
        ("msrc", cosod_path.MSRC),
        ("coca", cosod_path.CoCA),
        ("cosal2015", cosod_path.CoSal2015),
        ("cosod3k", cosod_path.CoSOD3k),
        ("icoseg", cosod_path.iCoSeg),
        ("imagepair", cosod_path.ImagePair),
        ("wicos", cosod_path.WICOS),
    ],
)

arg_config = dict(
    # 常用配置
    resume=False,  # 是否需要恢复模型
    info="wogroup",
    data=cosod_data,
    model="MINet_VGG16",
    save_num_models=1,
    has_val=True,
    has_test=True,
    use_amp=True,
    use_tta=False,
    use_mstrain=False,
    base_seed=0,
    use_cudnn_benchmark=False,
    in_size=dict(
        tr=dict(hw=320, extra_scales=[1.5, 1.25]),
        val=dict(hw=320),
        te=dict(hw=320),
    ),
    batch_size=dict(  # int or None => will load all data as a batch
        tr=8,
        val=8,
        te=8,
    ),
    clip_range=(0, 1),
    epoch_num=60,  # 训练周期
    num_workers=4,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    tb_update=50,  # >0 则使用tensorboard
    print_freq=50,  # >0, 保存迭代过程中的信息
)

loss_config = dict(
    bce=True,
    iou=False,
    weighted_iou=False,
    mae=False,
    mse=False,
    ssim=False,
)

optimizer_config = dict(
    lr=0.001,
    # ['trick', 'r3', 'all', 'finetune'],
    strategy="trick",
    optimizer="sgd",
    optimizer_candidates=dict(
        sgd=dict(momentum=0.9, weight_decay=5e-4, nesterov=False),
        adamw=dict(weight_decay=5e-4, eps=1e-8),
    ),
)

scheduler_config = dict(
    sche_usebatch=True,
    lr_strategy="poly",
    scheduler_candidates=dict(
        clr=dict(min_lr=0.001, max_lr=0.01, step_size=2000, mode="exp_range"),
        linearonclr=dict(),
        cos=dict(warmup_length=1, min_coef=0.025, max_coef=1),
        poly=dict(warmup_length=1, lr_decay=0.9, min_coef=0.025),
        step=dict(milestones=[30, 45, 55], gamma=0.1),
    ),
)
