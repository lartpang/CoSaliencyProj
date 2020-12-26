# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import model_zoo


def load_pretrained_params(model: nn.Module, pretrained_dict: dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    if list(pretrained_dict.keys())[0].startswith("module."):
        pretrained_dict = {
            k[7:]: v
            for k, v in pretrained_dict.items()
            if (k[7:] in model_dict) and (v.size() == model_dict[k[7:]].size())
        }
    else:
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if (k in model_dict) and (v.size() == model_dict[k].size())
        }
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
