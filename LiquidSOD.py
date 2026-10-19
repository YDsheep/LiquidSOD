import torch
import torch.nn as nn
from models.VMamba.classification.models import build_vssm_model
from models.VMamba.classification.config import get_config
from options import opt
from models.HA import HA
import os
from models.Dec3L import Dec, DecS
# from models.Dec import Dec
from models.GNN_L2 import MultiScaleGraphFusion, MultiScaleLiquidFusion, MambaFU, SnakeSpatialFusion
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DWConv(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1):
        super(DWConv, self).__init__()
        self.out_planes = out_planes
        self.dwconv = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, padding=dilation,
                                groups=in_planes, dilation=dilation)
        self.pconv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pconv(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class ECA(nn.Module):
    """ 高效通道注意力 """

    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

class Refine(nn.Module):

    def __init__(self, num, dims):
        super(Refine, self).__init__()
        self.dim = dims//num
        self.num = num

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num)
        ])

        # self.dwconv = nn.ModuleList(
        #     DWConv(self.dim, self.dim, kernel_size=5) for _ in range(num)
        # )

        self.conv_fu = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(dims)
        )
        self.eca = ECA(dims)

    def forward(self, x):
        residual = x
        batchsize, num_channels, height, width = x.data.size()
        x = x.view(batchsize, self.num, num_channels // self.num, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x_sh = x.view(batchsize, -1, height, width)
        x_n = torch.chunk(x_sh, self.num, dim=1)

        out_list = []
        for i in range(self.num):
            localf = self.convs[i](x_n[i])
            # globalf = self.dwconv[i](x_n[i])
            # processed = localf + globalf
            processed = localf
            out_list.append(processed)

        out = torch.cat(out_list, dim=1)
        out = self.conv_fu(out)
        out = self.eca(out)
        final = out + residual

        return final

class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel, use_eca=False):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )
        self.eca = ECA(out_channel) if use_eca else nn.Identity()

    def forward(self, x):
        x = self.reduce(x)
        return self.eca(x)


class SEM(nn.Module):
    def __init__(self, c_in, c_feat, c_atten):
        super(SEM, self).__init__()
        self.c_feat = c_feat
        self.c_atten = c_atten
        self.conv_feat = nn.Conv2d(c_in, c_feat, kernel_size=1)
        self.conv_atten = nn.Conv2d(c_in, c_atten, kernel_size=1)

    def forward(self, input):
        # input: [B, C, H, W]
        b, c, h, w = input.size()

        feat = self.conv_feat(input).view(b, self.c_feat, -1)  # [B, c_feat, HW]
        atten = self.conv_atten(input).view(b, self.c_atten, -1)  # [B, c_atten, HW]
        atten = F.softmax(atten, dim=-1)

        descriptors = torch.bmm(feat, atten.permute(0, 2, 1))

        return descriptors  # [B, c_feat, c_atten]


class SDM(nn.Module):
    def __init__(self, c_in, c_feat, c_atten):
        super(SDM, self).__init__()
        self.c_atten = c_atten
        self.c_feat = c_feat

        self.conv_atten = nn.Conv2d(c_in, c_atten, kernel_size=1)
        self.out_conv = nn.Conv2d(c_feat, c_in, kernel_size=1)

    def forward(self, input, descriptors):
        """
        input: 当前特征图 [B, C, H, W] (作为 Query)
        descriptors: 全局语义字典 [B, c_feat, c_atten] (来自 SEM)
        """
        b, c, h, w = input.size()

        atten_vectors = self.conv_atten(input).view(b, self.c_atten, -1)  # [B, c_atten, HW]
        atten_vectors = F.softmax(atten_vectors, dim=1)

        # (c_feat, c_atten) * (c_atten, HW) -> (c_feat, HW)
        out_feat = torch.bmm(descriptors, atten_vectors)

        out_feat = out_feat.view(b, self.c_feat, h, w)

        return self.out_conv(out_feat)

# LiquidSOD
class LiquidSOD(nn.Module):
    def __init__(self, channel=48):
        super(LiquidSOD, self).__init__()
        
        # todo 1 Backbone model

        self.vmamba_config = get_config(opt)
        self.encoder = build_vssm_model(self.vmamba_config)
        if opt.pre:
              if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                self.encoder.load_state_dict(checkpoint['model'], strict=False)
                print("=> loaded checkpoint")

        self.reduce_sal1 = Reduction(96, channel)
        self.reduce_sal2 = Reduction(96, channel)
        self.reduce_sal3 = Reduction(192, channel)
        self.reduce_sal4 = Reduction(384, channel)
        self.reduce_sal5 = Reduction(768, channel)

        self.muti3 = MambaFU(channel, channel)
        self.muti2 = MultiScaleGraphFusion(channel, channel, channel, 64)
        self.muti1 = SnakeSpatialFusion(channel, channel)


        self.re1 = Refine(2, channel)
        self.re2 = Refine(2, channel)
        self.re3 = Refine(2, channel)
        self.re4 = Refine(2, channel)
        self.re5 = Refine(2, channel)

        self.ms_liquid = MultiScaleLiquidFusion(
                channel_list=[channel, channel, channel, channel, channel],
                node_dim=channel,
                node_num=64,
                out_layer_idx=-1
            )

        self.decoder = Dec(channel, channel, channel)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x_rgb):

        x1_rgb, x2_rgb, x3_rgb, x4_rgb, x5_rgb = self.encoder(x_rgb)
        x1_rgb = x1_rgb.permute(0, 3, 1, 2)
        x2_rgb = x2_rgb.permute(0, 3, 1, 2)
        x3_rgb = x3_rgb.permute(0, 3, 1, 2)
        x4_rgb = x4_rgb.permute(0, 3, 1, 2)
        x5_rgb = x5_rgb.permute(0, 3, 1, 2)

        x_sal1 = self.reduce_sal1(x1_rgb)
        x_sal2 = self.reduce_sal2(x2_rgb)
        x_sal3 = self.reduce_sal3(x3_rgb)
        x_sal4 = self.reduce_sal4(x4_rgb)
        x_sal5 = self.reduce_sal5(x5_rgb)

        sal5 = self.re5(x_sal5)
        sal4 = self.re4(x_sal4)
        sal3 = self.re3(x_sal3)
        sal2 = self.re2(x_sal2)
        sal1 = self.re1(x_sal1)

        liquid_feat = self.ms_liquid([sal1, sal2, sal3, sal4, sal5])
        sal5 = liquid_feat


        x1f = self.muti1(sal2, sal1)
        x3f = self.muti3(sal5, sal4)
        x2f = self.muti2(x3f, sal3, x1f)

        x3_pred, x2_pred, x1_pred = self.decoder(x1f, x2f, x3f)

        return x3_pred, x2_pred, x1_pred


    
    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
