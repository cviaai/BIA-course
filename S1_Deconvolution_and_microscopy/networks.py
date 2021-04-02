import torch
import torch.nn as nn
import torch.nn.functional as F

# U-NET
class double_conv(nn.Module):
    '''(conv => Normalization => ReLU) * 2
    Normalization to be defined when calling a model'''

    def __init__(self, in_ch, out_ch, mode):
        super(double_conv, self).__init__()

        if mode=='instance':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        if mode=='batch':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        if mode=='none':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, mode):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, mode)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, mode):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, mode)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, mode, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, mode)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

###########__UNet__##################
class UNet(nn.Module):
    def __init__(self, mode, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 12, mode)
        self.down1 = down(12, 24, mode)
        self.down2 = down(24, 48, mode)
        self.down3 = down(48, 96, mode)
        self.down4 = down(96, 96, mode)
        self.up1 = up(192, 48, mode)
        self.up2 = up(96, 24, mode)
        self.up3 = up(48, 12, mode)
        self.up4 = up(24, 12, mode)

        self.outc = outconv(12, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x


###########__KPN__###################
class KPN(nn.Module):
    def __init__(self, mode, N=8, K=3):
        super(KPN, self).__init__()
        self.inc = inconv(1, 12, mode)
        self.down1 = down(12, 24, mode)
        self.down2 = down(24, 48, mode)
        self.down3 = down(48, 96, mode)
        self.down4 = down(96, 96, mode)
        self.up1 = up(192, 48, mode)
        self.up2 = up(96, 24, mode)
        self.up3 = up(48, 12, mode)
        self.up4 = up(24, 12, mode)
        self.outc = outconv(12, N*K*K)
        self.N = N
        self.K = K

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        x10 = self.outc(x9)
        x11 = x10.mean(dim=-1).mean(dim=-1)
        x12 = x11.reshape(x10.shape[0], self.N, self.K, self.K)  # B x D x H_g x W_g
        x = x12.unsqueeze(2)  # B x D x C x H_g x W_g
        return x

