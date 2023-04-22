import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Qconv import QuaternionConv as QConv
from model.Qconv import QuaternionLinear as QLinear

# from model.ViTmodule.mobilevit_block import MobileViTBlockv2, MobileViTBlockv3
import torchvision.transforms.functional as TF
from model.MobVIT import QMobViTBlock, MobileViTBlockv3, QDilatedInception


# channes
class _Residual_Block(nn.Module):
    # 16         16+16       1
    def __init__(self, in_channels, out_channels, groups=1, wide_width=True, downsample=False, upsample=False):
        super().__init__()

        self.downsample = downsample
        self.upsample = upsample
        self.Normer = True if (out_channels > 16) else False

        middle_channels = (in_channels if in_channels > out_channels else out_channels) if wide_width else out_channels

        self.conv1 = QConv(in_channels, middle_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        # if middle_channels < 4:
        #     num_groups = 1
        # else:
        #     num_groups = int(middle_channels / 4)
        # self.GN1 = nn.GroupNorm(num_groups, middle_channels, eps=1e-05, affine=True)
        self.relu1 = nn.LeakyReLU(0.02, inplace=True)
        self.conv2 = QConv(middle_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.SE = QSELayer(out_channels)

        if in_channels != out_channels:
            self.translation = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False, groups=groups)
        else:
            self.translation = None
        if self.upsample is True:
            self.UPConv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1,
                                             output_padding=1)

    def forward(self, x):
        if self.upsample:
            x = self.UPConv(x)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        identity = x

        out = self.conv1(x)
        # out = self.GN1(out)
        # if self.Normer:
        #     out = self.BN2(out)
        out = self.relu1(out)
        # out = self.BN2(out)
        out = self.conv2(out)
        out = self.BN2(out)
        out = self.SE(out)

        if self.translation is not None:
            identity = self.translation(identity)

        out += identity

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        return out


# channes
class _Residual_Block_REAL(nn.Module):
    # 16         16+16       1
    def __init__(self, in_channels, out_channels, groups=1, wide_width=True, downsample=False, upsample=False):
        super().__init__()

        self.downsample = downsample
        self.upsample = upsample
        self.Normer = True if (out_channels > 16) else False

        middle_channels = (in_channels if in_channels > out_channels else out_channels) if wide_width else out_channels

        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, stride=1, padding=1, bias=False,
                               groups=groups)
        # if middle_channels < 4:
        #     num_groups = 1
        # else:
        #     num_groups = int(middle_channels / 4)
        # self.GN1 = nn.GroupNorm(num_groups, middle_channels, eps=1e-05, affine=True)
        self.relu1 = nn.LeakyReLU(0.02, inplace=True)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                               groups=groups)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.SE = QSELayer(out_channels)

        if in_channels != out_channels:
            self.translation = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False, groups=groups)
        else:
            self.translation = None
        if self.upsample is True:
            self.UPConv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1,
                                             output_padding=1)

    def forward(self, x):
        if self.upsample:
            x = self.UPConv(x)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        identity = x

        out = self.conv1(x)
        # out = self.GN1(out)
        # if self.Normer:
        #     out = self.BN2(out)
        out = self.relu1(out)
        # out = self.BN2(out)
        out = self.conv2(out)
        out = self.BN2(out)
        out = self.SE(out)

        if self.translation is not None:
            identity = self.translation(identity)

        out += identity

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        return out

        #  残差块      4       16       16+16                                   downsample: i>3? True: false


def make_layer(block, blocks, in_channels, out_channels, groups=1, norm_layer=nn.BatchNorm2d, downsample=False,
               upsample=False):
    assert blocks >= 1
    layers = []
    # 16         16+16       1     nn.BatchNorm2d
    layers.append(block(in_channels, out_channels, groups, norm_layer, downsample=downsample, upsample=upsample))
    for i in range(1, blocks):
        layers.append(block(out_channels, out_channels, groups, norm_layer))

    return nn.Sequential(*layers)


# channes
class QDResidual_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.QDilatedInception = QDilatedInception(in_channel=in_channels, out_channel=in_channels)
        self.Residual_Block1 = _Residual_Block_REAL(in_channels=in_channels * 4, out_channels=in_channels)
        self.Residual_Block2 = _Residual_Block_REAL(in_channels=in_channels, out_channels=out_channels)

        # num_groups = int(in_channels / 16)
        # self.GN1 = nn.GroupNorm(num_groups, in_channels, eps=1e-05, affine=True)
        self.relu1 = nn.LeakyReLU(0.02, inplace=True)

    def forward(self, x):
        # identity = x
        x = self.QDilatedInception(x)
        x = self.Residual_Block1(x)
        # x = self.GN1(x)
        x = self.relu1(x)
        out = self.Residual_Block2(x)
        if self.in_channels == self.out_channels:
            out = out + x

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if channel < 16:
            reduction = 1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class QSELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(QSELayer, self).__init__()
        if reduction < 8:
            reduction = 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            QLinear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            QLinear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        if channel <= 16:
            reduction = 1
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class QMixUP(nn.Module):
    def __init__(self, channel):
        super(QMixUP, self).__init__()
        self.Att1 = CBAMLayer(channel)
        self.Att2 = CBAMLayer(channel)

    def forward(self, x_skip, x_unet):
        x1 = self.Att1(x_skip)
        x2 = self.Att2(x_unet)
        out = x1 + x2
        return out


# class QMixUP(nn.Module):
#     def __init__(self, channel):
#         super(QMixUP, self).__init__()
#         self.QSE1 = QSELayer(channel)
#         self.QSE2 = QSELayer(channel)

#     def forward(self, x_skip, x_unet):
#         x1 = self.QSE1(x_skip)
#         x2 = self.QSE2(x_unet)
#         out = x1 + x2
#         return out


class QNLNet(nn.Module):
    def __init__(self, in_chs, num_blocks, mid_chs, out_chs):
        super(QNLNet, self).__init__()
        BefNetList = []
        # BefNetList.append(QDResidual_Block(in_channels=in_chs, out_channels=in_chs),)
        BefNetList.append(QDResidual_Block(in_channels=in_chs, out_channels=mid_chs))
        for i in range(num_blocks):
            BefNetList.append(_Residual_Block(in_channels=mid_chs, out_channels=mid_chs))
        self.BefNet = nn.Sequential(*BefNetList)

        # QNLR Net
        self.QNLRBlosks = nn.Sequential(
            QDResidual_Block(in_channels=mid_chs, out_channels=mid_chs),
            QMobViTBlock(opts=None, in_channels=mid_chs, attn_unit_dim=mid_chs if mid_chs > 32 else 32),
            # QDResidual_Block(in_channels=mid_chs, out_channels=mid_chs),
            QMobViTBlock(opts=None, in_channels=mid_chs, attn_unit_dim=mid_chs if mid_chs > 32 else 32),
            _Residual_Block_REAL(in_channels=mid_chs, out_channels=mid_chs),
        )

        # QLDR Net
        self.QLDRBlocks = nn.Sequential(
            QDResidual_Block(in_channels=mid_chs, out_channels=mid_chs),
            _Residual_Block_REAL(in_channels=mid_chs, out_channels=mid_chs),
            _Residual_Block_REAL(in_channels=mid_chs, out_channels=mid_chs),
            QDResidual_Block(in_channels=mid_chs, out_channels=mid_chs),
            _Residual_Block_REAL(in_channels=mid_chs, out_channels=mid_chs),
            _Residual_Block_REAL(in_channels=mid_chs, out_channels=mid_chs),
            QDResidual_Block(in_channels=mid_chs, out_channels=mid_chs),
            _Residual_Block_REAL(in_channels=mid_chs, out_channels=mid_chs),
            _Residual_Block_REAL(in_channels=mid_chs, out_channels=mid_chs),
        )

        self.aggCNN = QMixUP(mid_chs)
        self.aggVIT = QMixUP(mid_chs)
        # After RESNet
        AftNetList = []
        for i in range(num_blocks):
            AftNetList.append(_Residual_Block_REAL(in_channels=mid_chs, out_channels=mid_chs))
        AftNetList.append(QDResidual_Block(in_channels=mid_chs, out_channels=out_chs), )
        # AftNetList.append(QDResidual_Block(in_channels=out_chs, out_channels=out_chs), )

        self.AftNet = nn.Sequential(*AftNetList)

    def forward(self, x):
        x = self.BefNet(x)
        res2 = x
        x_QNLR = self.QNLRBlosks(x)
        x_QLDR = self.QLDRBlocks(x)
        x = self.aggCNN(res2, x_QNLR)
        x = self.aggVIT(x, x_QLDR)
        # x = res2 + x_QNLR - x_QLDR
        out = self.AftNet(x)
        return out


class QSkipNet(nn.Module):
    def __init__(self, in_chs=16, num_scales=3, delta_chs=16):
        super(QSkipNet, self).__init__()
        self.in_chs = in_chs
        self.num_scales = num_scales
        self.delta_chs = delta_chs

        self.QVIT = nn.ModuleDict()
        self.UP = nn.ModuleDict()
        self.CompressBlock = nn.ModuleDict()
        self.DilateBlock = nn.ModuleDict()

        for i in range(self.num_scales):
            self.QVIT['{}'.format(i)] = QMobViTBlock(opts=None, in_channels=in_chs,
                                                     attn_unit_dim=in_chs * (self.num_scales - i))
            self.CompressBlock['{}'.format(i)] = QDResidual_Block(in_channels=in_chs, out_channels=self.in_chs)
            self.DilateBlock['{}'.format(i)] = nn.Sequential(
                QDResidual_Block(in_channels=self.in_chs * (4 - i), out_channels=in_chs),
                _Residual_Block(in_channels=in_chs, out_channels=in_chs),
                _Residual_Block(in_channels=in_chs, out_channels=in_chs),
                _Residual_Block(in_channels=in_chs, out_channels=in_chs),
            )

            self.UP['{}'.format(i)] = nn.ConvTranspose2d(in_channels=self.in_chs * (4 - i),
                                                         out_channels=self.in_chs * (4 - i), kernel_size=3,
                                                         stride=2, padding=1, output_padding=1)
            in_chs = in_chs + delta_chs

        self.CompressBlock['{}'.format(self.num_scales)] = QDResidual_Block(in_channels=in_chs,
                                                                            out_channels=self.in_chs)
        self.UP['{}'.format(self.num_scales)] = nn.ConvTranspose2d(in_channels=self.in_chs, out_channels=self.in_chs,
                                                                   kernel_size=3,
                                                                   stride=2, padding=1, output_padding=1)

    def forward(self, ResList):
        NLList = []
        Up_fea = self.CompressBlock['{}'.format(self.num_scales)](ResList[self.num_scales])
        Up_fea = self.UP['{}'.format(self.num_scales)](Up_fea)
        for i in range(self.num_scales):
            index = self.num_scales - i - 1
            Vit_fea = self.QVIT['{}'.format(index)](ResList[index])
            QCNN_fea = self.CompressBlock['{}'.format(index)](ResList[index])
            Up_fea = torch.cat([Up_fea, QCNN_fea], dim=1)
            QCNN_fea = self.DilateBlock['{}'.format(index)](Up_fea)
            Up_fea = self.UP['{}'.format(index)](Up_fea)
            NLList.append(ResList[index] + Vit_fea - QCNN_fea)
        NLList = NLList[::-1]
        return NLList


class QUNet(nn.Module):
    def __init__(self, num_blocks=4, in_chs=16, num_layers=4, delta_chs=16, num_scales=3):
        super(QUNet, self).__init__()
        self.in_chs = in_chs
        self.num_blocks = num_blocks
        self.num_scales = num_scales

        self.Qhead = QConv(in_channels=4, out_channels=in_chs, kernel_size=3, stride=1, padding=1)

        self.CNNenc = nn.ModuleDict()
        # self.QMixup = nn.ModuleDict()
        self.decDEC = nn.ModuleDict()

        self.QNLNet = QSkipNet(in_chs=in_chs, num_scales=self.num_scales, delta_chs=delta_chs)

        for i in range(num_blocks):  # 残差块           4        16    16+16      i>3? True: false
            self.CNNenc['enc{}'.format(i)] = make_layer(block=_Residual_Block, blocks=num_layers,
                                                        in_channels=in_chs, out_channels=in_chs + delta_chs,
                                                        downsample=i < num_scales)
            in_chs = in_chs + delta_chs

        for i in range(num_blocks):
            # self.QMixup['{}'.format(i)] = QMixUP(in_chs)
            self.decDEC['dec{}'.format(i)] = make_layer(block=_Residual_Block, blocks=num_layers,
                                                        in_channels=in_chs, out_channels=in_chs - delta_chs,
                                                        upsample=i >= (num_blocks - num_scales))
            in_chs = in_chs - delta_chs
        # self.QMixup['{}'.format(self.num_blocks)] = QMixUP(self.in_chs)

        self.Qtail = QConv(in_channels=in_chs, out_channels=4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        DEC支路
        """

        Res = []
        Res_NLnet = []

        x_4chs = x
        x = self.Qhead(x)
        Res_NLnet.append(x)
        for i in range(self.num_blocks):
            x = self.CNNenc['enc{}'.format(i)](x)
            Res.append(x)
            if i < self.num_scales:
                Res_NLnet.append(x)

        Res = Res[::-1]
        NLList = self.QNLNet(Res_NLnet)

        for i in range(self.num_blocks):
            if i > (self.num_blocks - self.num_scales):
                x = x + NLList[self.num_blocks - i]
                # x = self.QMixup['{}'.format(i)](x, NLList[self.num_blocks - i])
            else:
                x = x + Res[i]
                # x = self.QMixup['{}'.format(i)](x, Res[i])
            x = self.decDEC['dec{}'.format(i)](x)
        # x = self.QMixup['{}'.format(self.num_blocks)](x, NLList[0])
        x = x + NLList[0]
        x = self.Qtail(x)
        out = x + x_4chs

        return out


class QDerainNet(nn.Module):
    def __init__(self, ):
        super(QDerainNet, self).__init__()
        self.head = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.QNLNet0 = QNLNet(in_chs=48 + 4, num_blocks=4, mid_chs=64, out_chs=4)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.QNLNet1 = QNLNet(in_chs=32 + 4, num_blocks=3, mid_chs=48, out_chs=48)
        self.Up1 = nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.QNLNet2 = QNLNet(in_chs=4, num_blocks=2, mid_chs=32, out_chs=32)
        self.Up2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.SE1 = QMixUP(channel=4)
        self.tail = nn.Sequential(
            QConv(4, 16, kernel_size=3, stride=1, padding=1),
            QConv(16, 4, kernel_size=3, stride=1, padding=1),
        )

        self.QUnet = QUNet()

    def forward(self, x):
        r = self.head(x)
        x = torch.cat([r, x], dim=1)  # channels=4
        x_init = x
        x1 = self.pool1(x)
        x2 = self.pool1(x1)
        # x3 = self.pool1(x2)

        # x3 = self.QNLNet3(x3)
        # x3 = self.Up3(x3)

        # x2 = torch.cat([x2, x3], dim=1)
        x2 = self.QNLNet2(x2)
        x2 = self.Up2(x2)

        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.QNLNet1(x1)
        x1 = self.Up1(x1)

        x = torch.cat([x, x1], dim=1)
        x = self.QNLNet0(x)  # chs = 4

        x = self.SE1(x, x_init)
        res = x
        x = self.tail(x)
        x = res + x
        out_QUnet = self.QUnet(x_init)
        x = out_QUnet + x

        out = x[:, 1:, :, :]
        return out


def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params


if __name__ == '__main__':
    inp = torch.rand([2, 3, 480, 480])
    print(inp.shape)
    model = QDerainNet()
    # print(model)
    cnn_paras_count(model)
    out = model(inp)
    print(out.shape)


