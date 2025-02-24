import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.input_conv.apply(weights_init('xavier'))

    def forward(self, input):
        output = self.input_conv(input)
        return output


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        h = self.conv(input)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h


# class PConvUNet(nn.Module):
#     def __init__(self, layer_size=7, input_channels=3+3, output_channels=3, upsampling_mode='nearest'):
#         super().__init__()
#         self.freeze_enc_bn = False
#         self.upsampling_mode = upsampling_mode
#         self.layer_size = layer_size
#         self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-5')
#         self.enc_2 = PCBActiv(64, 128, sample='down-3')
#         self.enc_3 = PCBActiv(128, 256, sample='down-3')
#         self.enc_4 = PCBActiv(256, 512, sample='down-3')
#         # self.enc_5 = PCBActiv(512, 512, sample='down-3')
#         # self.enc_6 = PCBActiv(512, 512, sample='down-3')
#         # self.enc_7 = PCBActiv(512, 512, sample='down-3')
#         # self.dec_7 = PCBActiv(512 + 512, 512, activ='leaky')
#         # self.dec_6 = PCBActiv(512 + 512, 512, activ='leaky')
#         # self.dec_5 = PCBActiv(512 + 512, 512, activ='leaky')
#         for i in range(4, self.layer_size):
#             name = 'enc_{:d}'.format(i + 1)
#             setattr(self, name, PCBActiv(512, 512, sample='down-3'))
#
#         for i in range(4, self.layer_size):
#             name = 'dec_{:d}'.format(i + 1)
#             setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
#         self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
#         self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
#         self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
#         self.dec_1 = PCBActiv(64 + input_channels, output_channels, bn=False, activ=None, conv_bias=True)
#
#     def forward(self, input):
#         h_dict = {}  # for the output of enc_N
#         h_dict['h_0']= input
#         h_key_prev = 'h_0'
#         for i in range(1, self.layer_size + 1):
#             l_key = 'enc_{:d}'.format(i)
#             h_key = 'h_{:d}'.format(i)
#             h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])
#             h_key_prev = h_key
#
#         h_key = 'h_{:d}'.format(self.layer_size)
#         h= h_dict[h_key]
#
#         for i in range(self.layer_size, 0, -1):
#             enc_h_key = 'h_{:d}'.format(i - 1)
#             dec_l_key = 'dec_{:d}'.format(i)
#             b, c, ww, hh = h_dict[enc_h_key].shape
#             h = F.interpolate(h, size=(ww, hh), mode=self.upsampling_mode)
#
#             h = torch.cat([h, h_dict[enc_h_key]], dim=1)
#             h = getattr(self, dec_l_key)(h)
#
#         return h


class PConvUNet_Enc_hdr(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        # self.enc_5 = PCBActiv(512, 512, sample='down-3')
        # self.enc_6 = PCBActiv(512, 512, sample='down-3')
        # self.enc_7 = PCBActiv(512, 512, sample='down-3')
        # self.dec_7 = PCBActiv(512 + 512, 512, activ='leaky')
        # self.dec_6 = PCBActiv(512 + 512, 512, activ='leaky')
        # self.dec_5 = PCBActiv(512 + 512, 512, activ='leaky')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

    def forward(self, input):
        h_dict = {}
        h_dict['h_0'] = input
        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h = h_dict[h_key]

        return h, h_dict


class PConvUNet_Dec_hdr(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, output_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            # if i == (self.layer_size - 1):
            #     setattr(self, name, PCBActiv(512 + 512 + 512, 512, activ='leaky'))
            # else:
            setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, output_channels, bn=False, activ=None, conv_bias=True)

    def forward(self, h, h_dict):
        h_dict_hdr = {}
        h_dict_hdr['h_{:d}'.format(self.layer_size)] = h
        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            b, c, ww, hh = h_dict[enc_h_key].shape
            h = F.interpolate(h, size=(ww, hh), mode=self.upsampling_mode)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h = getattr(self, dec_l_key)(h)

            h_dict_hdr[enc_h_key] = h

        return h, h_dict_hdr


class PConvUNet_Enc_ldr(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-5')
        self.enc_2 = PCBActiv(64 * 2, 128, sample='down-3')
        self.enc_3 = PCBActiv(128 * 2, 256, sample='down-3')
        self.enc_4 = PCBActiv(256 * 2, 512, sample='down-3')
        # self.enc_5 = PCBActiv(512, 512, sample='down-3')
        # self.enc_6 = PCBActiv(512, 512, sample='down-3')
        # self.enc_7 = PCBActiv(512, 512, sample='down-3')
        # self.dec_7 = PCBActiv(512 + 512, 512, activ='leaky')
        # self.dec_6 = PCBActiv(512 + 512, 512, activ='leaky')
        # self.dec_5 = PCBActiv(512 + 512, 512, activ='leaky')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512 * 2, 512, sample='down-3'))

    def forward(self, input, h_dict_hdr):
        h_dict = {}
        h_dict['h_0'] = input
        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            if i == 1:
                h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])
            else:
                h_dict[h_key] = getattr(self, l_key)(torch.cat((h_dict[h_key_prev], h_dict_hdr[h_key_prev]), dim=1))
            # h_dict[h_key] = getattr(self, l_key)(torch.cat((h_dict[h_key_prev], h_dict_hdr[h_key_prev]), dim=1))
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h = h_dict[h_key]

        return h, h_dict


class PConvUNet_Dec(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, output_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            # if i == (self.layer_size-1):
            setattr(self, name, PCBActiv(512 + 512 * 2, 512, activ='leaky'))
            # else:
            #     setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256 * 2, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128 * 2, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64 * 2, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, output_channels, bn=False, activ=None, conv_bias=True)

    def forward(self, h, h_dict, h_dict_hdr):
        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            b, c, ww, hh = h_dict[enc_h_key].shape
            h = F.interpolate(h, size=(ww, hh), mode=self.upsampling_mode)
            if (i - 1) == 0:
                h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            else:
                h = torch.cat([h, h_dict[enc_h_key], h_dict_hdr[enc_h_key]], dim=1)
            h = getattr(self, dec_l_key)(h)

        return h
