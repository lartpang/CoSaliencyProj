# -*- coding: utf-8 -*-
# @Time    : 2020/11/7
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import torch
from torch import nn

from backbone.origin.from_origin import Backbone_R50_Custumed, Backbone_V16_Custumed
from module.BaseBlocks import BasicConv2d
from utils.tensor_ops import cus_sample, upsample_add


class R50BasicEncoder(nn.Module):
    def __init__(self):
        super(R50BasicEncoder, self).__init__()
        self.encoders = nn.ModuleList(Backbone_R50_Custumed(3))

    def forward(self, x):
        outs = []
        for en in self.encoders:
            x = en(x)
            outs.append(x)
        return outs


class V16BasicEncoder(nn.Module):
    def __init__(self):
        super(V16BasicEncoder, self).__init__()
        self.encoders = nn.ModuleList(Backbone_V16_Custumed(3))

    def forward(self, x):
        outs = []
        for en in self.encoders:
            x = en(x)
            outs.append(x)
        return outs


class BasicTransLayer(nn.Module):
    def __init__(self, out_c):
        super(BasicTransLayer, self).__init__()
        self.c5_down = nn.Conv2d(2048, out_c, 1)
        self.c4_down = nn.Conv2d(1024, out_c, 1)
        self.c3_down = nn.Conv2d(512, out_c, 1)
        self.c2_down = nn.Conv2d(256, out_c, 1)
        self.c1_down = nn.Conv2d(64, out_c, 1)

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 5
        c1, c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        outs = [c5, c4, c3, c2, c1]
        return outs


class BasicSegHead_Res(nn.Module):
    def __init__(self, mid_c):
        super(BasicSegHead_Res, self).__init__()
        self.p5_d5 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.p4_d4 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.p3_d3 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.p2_d2 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.p1_d1 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.seg_conv = nn.Sequential(BasicConv2d(mid_c, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))

    def forward(self, xs):
        assert len(xs) == 5
        p1, p2, p3, p4, p5 = xs

        d5 = self.p5_d5(p5)
        d4 = self.p4_d4(upsample_add(d5, p4))
        d3 = self.p3_d3(upsample_add(d4, p3))
        d2 = self.p2_d2(upsample_add(d3, p2))
        d1 = self.p1_d1(upsample_add(d2, p1))
        d0 = cus_sample(d1, scale_factor=2)

        s_0 = self.seg_conv(d0)
        return s_0


class R50BasicEncoderTransLayer(nn.Module):
    def __init__(self, in_c=3, out_c=64):
        super(R50BasicEncoderTransLayer, self).__init__()
        self.encoders = nn.ModuleList(Backbone_R50_Custumed(in_c))
        out_c_encoders = [64, 256, 512, 1024, 2048]
        self.translayers = nn.ModuleList()
        for mid_c in out_c_encoders:
            self.translayers.append(nn.Conv2d(mid_c, out_c, 1))

    def forward(self, x):
        outs = []
        for i, (en, trans) in enumerate(zip(self.encoders, self.translayers)):
            x = en(x)
            o = trans(x)
            outs.append(o)
        return outs


class V16BasicEncoderTransLayer(nn.Module):
    def __init__(self, in_c=3, out_c=64):
        super(V16BasicEncoderTransLayer, self).__init__()
        self.encoders = nn.ModuleList(Backbone_V16_Custumed(in_c))
        out_c_encoders = [64, 128, 256, 512, 512]
        self.translayers = nn.ModuleList()
        for mid_c in out_c_encoders:
            self.translayers.append(nn.Conv2d(mid_c, out_c, 1))

    def forward(self, x):
        outs = []
        for i, (en, trans) in enumerate(zip(self.encoders, self.translayers)):
            x = en(x)
            o = trans(x)
            outs.append(o)
        return outs


class BasicSegHead_Vgg(nn.Module):
    def __init__(self, mid_c):
        super(BasicSegHead_Vgg, self).__init__()
        self.p5_d5 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.p4_d4 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.p3_d3 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.p2_d2 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.p1_d1 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.seg_conv = nn.Sequential(BasicConv2d(mid_c, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))

    def forward(self, xs):
        assert len(xs) == 5
        p1, p2, p3, p4, p5 = xs

        d5 = self.p5_d5(p5)
        d4 = self.p4_d4(upsample_add(d5, p4))
        d3 = self.p3_d3(upsample_add(d4, p3))
        d2 = self.p2_d2(upsample_add(d3, p2))
        d1 = self.p1_d1(upsample_add(d2, p1))

        s_0 = self.seg_conv(d1)
        return s_0


if __name__ == "__main__":
    model = V16BasicEncoderTransLayer()
    in_data = torch.randn(4, 3, 224, 224)
    print([x.shape for x in model(in_data)])
