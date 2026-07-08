import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.fft

from torch.fft import irfft2
from torch.fft import rfft2

def rfft(x, d):
    t = rfft2(x, dim=(-d, -1))
    return torch.stack((t.real, t.imag), -1)
def irfft(x, d, signal_sizes):
    return irfft2(torch.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=(-d, -1))
class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(FFC, self).__init__()

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=kernel_size, stride=stride, padding=padding, groups=1, bias=False)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = rfft(x, 2)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        output = irfft(ffted, 2, signal_sizes=r_size[2:])

        return output
class ConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x_out = self.act(x)
        return x_out
class UpConvAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x_out = self.act(x)
        return x_out
class OutConvBlock(nn.Module):
    def __init__(self, in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x_out = self.act(x)
        return x_out
class DLKConvAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.regularconv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.largeconv = nn.Conv2d(in_channels, out_channels, 5, 1, 2)
        self.oneconv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.fftconv = FFC(in_channels, out_channels)
        self.act = nn.GELU()
    def forward(self, x):
        regularx = self.regularconv(x)
        largex = self.largeconv(x)
        onex = self.oneconv(x)
        fftconv = self.fftconv(x)
        x_out = regularx + largex + onex + fftconv + x
        x_out = self.act(x_out)
        return x_out
class FEncoder_DLK(nn.Module):
    def __init__(self, in_channels=1, channel_num=16):
        super().__init__()
        self.conv_11 = ConvAct(in_channels, channel_num)
        self.conv_12 = ConvAct(channel_num, channel_num)
        self.conv_21 = ConvAct(channel_num, channel_num * 2)
        self.conv_22 = DLKConvAct(channel_num * 2, channel_num * 2)
        self.conv_31 = ConvAct(channel_num * 2, channel_num * 4)
        self.conv_32 = DLKConvAct(channel_num * 4, channel_num * 4)
        self.conv_41 = ConvAct(channel_num * 4, channel_num * 8)
        self.conv_42 = DLKConvAct(channel_num * 8, channel_num * 8)
        self.conv_51 = ConvAct(channel_num * 8, channel_num * 16)
        self.conv_52 = DLKConvAct(channel_num * 16, channel_num * 16)
        self.downsample = nn.MaxPool2d(3, stride=2, padding=1)
    def forward(self, x_in):
        x_1 = self.conv_11(x_in)
        x_1 = self.conv_12(x_1)
        x = self.downsample(x_1)
        x_2 = self.conv_21(x)
        x_2 = self.conv_22(x_2)
        x = self.downsample(x_2)
        x_3 = self.conv_31(x)
        x_3 = self.conv_32(x_3)
        x = self.downsample(x_3)
        x_4 = self.conv_41(x)
        x_4 = self.conv_42(x_4)
        x = self.downsample(x_4)
        x_5 = self.conv_51(x)
        x_5 = self.conv_52(x_5)
        return [x_1, x_2, x_3, x_4, x_5]
def get_winsize(x_size, window_size):
    use_window_size = list(window_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
    return tuple(use_window_size)
def get_aff(xk, yk):
    xk = xk.permute(0, 2, 1)  # b, c, n
    yk = yk.permute(0, 2, 1)  # b, c, n
    affinity = xk.transpose(1, 2) @ yk
    maxes = torch.max(affinity, dim=1, keepdim=True)[0]
    x_exp = torch.exp(affinity - maxes)
    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
    affinity = x_exp / x_exp_sum
    affinity = affinity.permute(0, 2, 1)
    return affinity
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], C)
    return windows
def window_reverse(windows, window_size, dims):
    B, H, W = dims
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
class Correlation2D(nn.Module):
    def __init__(self, dim, window_size=[2, 2]):
        super().__init__()
        self.channel = window_size[0] * window_size[1]
        self.window_size = window_size
    def forward(self, x_in, y_in):
        b, c, h, w = x_in.shape
        n = h * w
        x = x_in.permute(0, 2, 3, 1)
        y = y_in.permute(0, 2, 3, 1)
        window_size = get_winsize((h, w), self.window_size)
        pad_l = pad_t = 0
        pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        y = nnf.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape
        dims = [b, hp, wp]
        x_windows = window_partition(x, window_size)
        y_windows = window_partition(y, window_size)
        affinity = get_aff(x_windows, y_windows)
        affinity = affinity.view(-1, *(window_size + (self.channel,)))
        affinity = window_reverse(affinity, window_size, dims)
        if pad_r > 0 or pad_b > 0:
            affinity = affinity[:, :h, :w, :].contiguous()
        affinity = affinity.view(b, n, self.channel).reshape(b, n, self.channel, 1).transpose(2, 3) # b, n, 1, 8
        out = affinity.reshape(b, h, w, self.channel).permute(0, 3, 1, 2)
        return out
class CorrModule_win(nn.Module):
    def __init__(self, channel_num, window_size=[7, 7]):
        super().__init__()
        self.corr = Correlation2D(channel_num, window_size=window_size)
        self.conv = ConvAct(channel_num * 2 + window_size[0]*window_size[1], channel_num)
    def forward(self, mov, fix):
        cost_volume = self.corr(mov, fix)
        x = torch.cat((mov, cost_volume, fix), dim=1)
        x_out = self.conv(x)
        return x_out

class UNet_full(nn.Module):
    def __init__(self, input_nc=1, channel_num=16):
        super().__init__()
        self.fencoder = FEncoder_DLK(in_channels=input_nc, channel_num=channel_num)

        self.corr_1 = ConvAct(channel_num * 2, channel_num * 1)
        self.corr_2 = CorrModule_win(channel_num * 2, window_size=[3, 3])
        self.corr_3 = CorrModule_win(channel_num * 4, window_size=[3, 3])
        self.corr_4 = CorrModule_win(channel_num * 8, window_size=[3, 3])
        self.corr_5 = CorrModule_win(channel_num * 16, window_size=[3, 3])

        self.conv_1 = ConvAct(channel_num * 1 * 2, channel_num * 1)
        self.conv_2 = ConvAct(channel_num * 2 * 2, channel_num * 2)
        self.conv_3 = ConvAct(channel_num * 4 * 2, channel_num * 4)
        self.conv_4 = ConvAct(channel_num * 8 * 2, channel_num * 8)

        self.upsample_1 = UpConvAct(channel_num * 2, channel_num * 1)
        self.upsample_2 = UpConvAct(channel_num * 4, channel_num * 2)
        self.upsample_3 = UpConvAct(channel_num * 8, channel_num * 4)
        self.upsample_4 = UpConvAct(channel_num * 16, channel_num * 8)

        self.outlayer = OutConvBlock()

    def forward(self, moving, fixed):
        mov_1, mov_2, mov_3, mov_4, mov_5 = self.fencoder(moving)
        fix_1, fix_2, fix_3, fix_4, fix_5 = self.fencoder(fixed)

        c_5 = self.corr_5(fix_5, mov_5)

        dconv_5_up = self.upsample_4(c_5)
        c_4 = self.corr_4(fix_4, mov_4)
        cat = torch.cat([c_4, dconv_5_up], dim=1)
        dconv_4 = self.conv_4(cat)

        dconv_4_up = self.upsample_3(dconv_4)
        c_3 = self.corr_3(fix_3, mov_3)
        cat = torch.cat([c_3, dconv_4_up], dim=1)
        dconv_3 = self.conv_3(cat)

        dconv_3_up = self.upsample_2(dconv_3)
        c_2 = self.corr_2(fix_2, mov_2)
        cat = torch.cat([c_2, dconv_3_up], dim=1)
        dconv_2 = self.conv_2(cat)

        dconv_2_up = self.upsample_1(dconv_2)
        c_1 = self.corr_1(torch.cat([fix_1, mov_1], dim=1))
        cat = torch.cat([c_1, dconv_2_up], dim=1)
        dconv_1 = self.conv_1(cat)

        out = self.outlayer(dconv_1)

        return out
