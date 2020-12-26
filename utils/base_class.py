# -*- coding: utf-8 -*-
# @Time    : 2020/9/18
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import torch.nn as nn

from backbone.origin.from_origin import Backbone_R50_Custumed, Backbone_R101_Custumed
from utils.misc import construct_print


class BasicEncoder(nn.Module):
    def __init__(self, in_channel, model):
        super(BasicEncoder, self).__init__()
        self.encoders = nn.ModuleList(model(in_channel))

    def forward(self, x):
        outs = []
        for en in self.encoders:
            x = en(x)
            outs.append(x)
        return outs


class BaseModel(nn.Module):
    def __init__(self, backbone_info: dict):
        super(BaseModel, self).__init__()
        self.pretrain_path = backbone_info.get("pretrain_path")

        if backbone_info["backbone"] == "resnet":
            depth = backbone_info["backbone_cfg"].get("depth", 50)
            assert depth in [50, 101]
            if depth == 101:
                resnet = Backbone_R101_Custumed
            else:
                resnet = Backbone_R50_Custumed
            self.shared_encoder = BasicEncoder(in_channel=3, model=resnet)
        else:
            raise NotImplementedError

        if backbone_info["freeze_bn"]:
            self.freeze_bn()

    def freeze_bn(self):
        construct_print("We will freeze all BN layers.")
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_grouped_params(self):
        """
        配合tools_funcs.make_optim_with_cfg `optimizer_strategy == "finetune"`使用
        Returns:
            params_groups = dict(pretrained=backbone, retrained=head)
        """
        backbone = []
        head = []
        for name, params_tensor in self.named_parameters():
            if name.startswith("encoder"):
                backbone.append(params_tensor)
            else:
                head.append(params_tensor)
        params_groups = dict(pretrained=backbone, retrained=head)
        return params_groups
