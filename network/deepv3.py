"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import logging
from numpy import var
import torch
from torch import nn
from network import Resnet
from network.mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights
from network.AdaIN import AdaIN
from network.class_dsu import class_dsu
import random
import torchvision.models as models

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-101', args=None):
        super(DeepV3Plus, self).__init__()
        self.args = args
        self.trunk = trunk
        self.AdaIN = AdaIN(self.args)  #[B,C]
        self.pertubration = class_dsu(num_classes)
        self.num_classes = num_classes
        channel_1st = 3
        channel_2nd = 64
        channel_3rd = 256
        channel_4th = 512
        prev_final_channel = 1024
        final_channel = 2048

        if trunk == 'resnet-50':
            resnet = Resnet.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101':
            resnet = Resnet.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool, self.fc= \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool, resnet.fc
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        os = 16


        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.dsn = nn.Sequential(
            nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
            Norm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        initialize_weights(self.AdaIN)
        initialize_weights(self.dsn)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5

    def forward(self, x, gt=None, style_hallucination=False, out_prob=False, return_style_features=False, fixedAdain=False, cdsu=False):
    
        x_size = x.size()  
        
        if isinstance(return_style_features, list) and len(return_style_features):
            multi_feature = True
        else:
            multi_feature = False

        # Iterative train segmentation model and IAM module 
        if fixedAdain:  # fixed Adain, train segmentation model
            self.layer0.requires_grad = True
            self.layer1.requires_grad = True
            self.layer2.requires_grad = True
            self.layer3.requires_grad = True
            self.layer4.requires_grad = True
            self.AdaIN.requires_grad = False
            self.aspp.requires_grad = True
            self.bot_aspp.requires_grad = True
            self.bot_fine.requires_grad = True
            self.final1.requires_grad = True
            self.final2.requires_grad = True
        else:   # fixed segmentation model, train Adain
            self.layer0.requires_grad = False
            self.layer1.requires_grad = False
            self.layer2.requires_grad = False
            self.layer3.requires_grad = False
            self.layer4.requires_grad = False
            self.AdaIN.requires_grad = True
            self.aspp.requires_grad = False
            self.bot_aspp.requires_grad = False
            self.bot_fine.requires_grad = False
            self.final1.requires_grad = False
            self.final2.requires_grad = False

        f_style = {}
        f_ori = x
        f_adain = x
        
        # image-level augmentation
        if style_hallucination:
            f_adain, ori_style, aug_style = self.AdaIN(x)
            x = torch.cat((x, f_adain),dim=0)

        if fixedAdain:
            del f_adain, f_ori
        
        x = self.layer0(x)   

        # class-level augmentation
        if cdsu:
            aug_f = x[x_size[0]:, :, :, :].clone()
            ori_f = x[:x_size[0], :, :, :].clone()
            local_aug_f = self.pertubration(ori_f, gt)
            x = torch.cat((ori_f, aug_f, local_aug_f),dim=0)
 
        x = self.layer1(x)  
        low_level = x

        x = self.layer2(x)  

        x = self.layer3(x)  
        f_style['layer3'] = x
        aux_out = x

        x = self.layer4(x)  
        f_style['layer4'] = x


        if return_style_features and (not style_hallucination):
            output = {
                'features': f_style
            }                
            return output

        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final1(dec0)
        dec2 = self.final2(dec1)
        main_out = Upsample(dec2, x_size[2:])
        if fixedAdain:
            aux_out = self.dsn(aux_out)
            output = {
                'main_out': main_out,
                'aux_out': aux_out,
                'features': f_style
            }                
            return output
        else:
            output = {
                'main_out': main_out,
                'ori_features':ori_style,
                'adain_features':aug_style
            }    
            return output


def DeepR50V3PlusD(args, num_classes):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', args=args)


def DeepR101V3PlusD(args, num_classes):
    """
    Resnet 101 Based Network, the origin 7x7
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3Plus(num_classes, trunk='resnet-101', args=args)
