# -*- coding: utf-8 -*-
# @Time    : 2020/11/8
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
from torch import nn

from module.BaseModule import BasicSegHead_Vgg, V16BasicEncoderTransLayer


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.encoder = V16BasicEncoderTransLayer(out_c=64)
        self.seg_head = BasicSegHead_Vgg(mid_c=64)

    def forward(self, data):
        feats = self.encoder(data["image"])
        seg_logits = self.seg_head(feats)
        return dict(seg=seg_logits)

    def get_grouped_params(self):
        """
        配合tools_funcs.make_optim_with_cfg `optimizer_strategy == "finetune"`使用
        Returns:
            params_groups = dict(pretrained=backbone, retrained=head)
        """
        backbone = []
        head = []
        for name, params_tensor in self.named_parameters():
            if name.startswith("encoder.encoders"):
                backbone.append(params_tensor)
            else:
                head.append(params_tensor)
        params_groups = dict(pretrained=backbone, retrained=head)
        return params_groups


if __name__ == "__main__":
    model = FPN()
    print([k for k, v in model.named_parameters()])
