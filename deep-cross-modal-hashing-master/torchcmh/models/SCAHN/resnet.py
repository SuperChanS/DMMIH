from __future__ import absolute_import
from __future__ import division

from torchcmh.models import BasicModule

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

import torch
from torch import nn

from .pyramidpooling import PyramidPooling
from .weight_attention import WeightAttention
from ..AGed.networks import FPA, PAN
import math

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.leak_relu = nn.LeakyReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        # out = F.tanh(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(BasicModule):
    """Residual network.
    Reference:
        He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(self, bit, block, layers,
                 last_stride=2,
                 fusion_num=4,
                 **kwargs):
        self.inplanes = 64
        self.module_name = 'ASCHN_resnet'
        self.fusion_num = fusion_num
        super(ResNet, self).__init__()
        self.feature_dims = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion, 512 * block.expansion]

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier1 = nn.Linear(128, bit)
        self.classifier2 = nn.Linear(256, bit)
        self.classifier3 = nn.Linear(512, bit)
        self.classifier4 = nn.Linear(1024, bit)
        self.classifierr = nn.Linear(2048, bit)

        self.BN1 = nn.BatchNorm1d(128)
        self.BN2 = nn.BatchNorm1d(256)
        self.BN3 = nn.BatchNorm1d(512)
        self.BN4 = nn.BatchNorm1d(1024)
        self.BNr = nn.BatchNorm1d(2048)
        self.spp1 = PyramidPooling([1,1])

        self._init_params()

        self.weight = WeightAttention(bit=bit, ms_num=2)
        #双线性池化
        # resnet 50, ...
        self.proj1 = nn.Conv2d(in_channels=128, out_channels=2048, kernel_size=1, stride=1)
        self.proj2 = nn.Conv2d(in_channels=256, out_channels=2048, kernel_size=1, stride=1)
        self.proj12 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1, stride=1)
        self.proj3 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1)
        self.proj4 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1)
        self.fc_concat = torch.nn.Linear(bit*2, bit)

        self.softmax = nn.LogSoftmax(dim=1)
        self.avgpool = nn.AvgPool2d(1)
        self.ca = ChannelAttention(2048)
        self.sa = SpatialAttention()

        # self.pan = PAN()
        # self.fpa1 = FPA(64)
        # self.fpa2 = FPA(128)
        # self.fpa3 = FPA(256)
        # self.fpa4 = FPA(512)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _construct_hash_layer(self, bit):
        for feature_dim in self.feature_dims:
            self.classifiers.append(nn.Linear(feature_dim, bit))

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __feature_maps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def forward(self, x: torch.Tensor):
        f = self.__feature_maps(x)
        f1 = self.layer1(f)
        v1 = self.spp1(f1)
        f2 = self.layer2(f1)
        v2 = self.spp1(f2)
        f3 = self.layer3(f2)
        v3 = self.spp1(f3)
        f4 = self.layer4(f3)
        v4 = self.spp1(f4)
        batch_size = v1.size(0)

        v_t1 = v1.unsqueeze(2).unsqueeze(3)
        v_t2 = v2.unsqueeze(2).unsqueeze(3)
        v_t3 = v3.unsqueeze(2).unsqueeze(3)
        v_t4 = v4.unsqueeze(2).unsqueeze(3)
        v_temp1 = self.proj1(v_t1)
        v_temp2 = self.proj2(v_t2)
        inter_1 = v_temp1 * v_temp2
        v_temp12 = self.proj12(inter_1)

        v_temp3 = self.proj3(v_t3)
        inter_2 = v_temp12 * v_temp3
        v_temp23 = self.proj12(inter_2)

        v_temp4 = self.proj4(v_t4)
        inter_3 = v_temp23 * v_temp4

        inter1 = self.avgpool(inter_3).view(batch_size,-1)
        resultt1 = torch.nn.functional.normalize(torch.sign(inter1) * torch.sqrt(torch.abs(inter1) + 1e-10))
        # print(resultt1.shape)
        resultt1 = resultt1.unsqueeze(2).unsqueeze(3)
        out1 = self.ca(resultt1) * resultt1
        out1 = self.sa(out1) * out1
        out1 = out1.view(batch_size,-1)
        result1 = self.classifierr(out1)
        # print(result1.shape)

        v_temp11 = self.proj4(v_t4)
        v_temp22 = self.proj3(v_t3)
        inter_11 = v_temp11 * v_temp22
        v_temp1122 = self.proj12(inter_11)

        v_temp33 = self.proj2(v_t2)
        inter_22 = v_temp1122 * v_temp33
        v_temp2233 = self.proj12(inter_22)

        v_temp44 = self.proj1(v_t1)
        inter_33 = v_temp2233 * v_temp44
        inter2 = self.avgpool(inter_33).view(batch_size, -1)
        resultt2 = torch.nn.functional.normalize(torch.sign(inter2) * torch.sqrt(torch.abs(inter2) + 1e-10))
        resultt2 = resultt2.unsqueeze(2).unsqueeze(3)
        out = self.ca(resultt2) * resultt2
        out = self.sa(out) * out
        out = out.view(batch_size,-1)
        result2 = self.classifierr(out)

        middle_hash = [result1, result2]
        y = self.weight(*middle_hash)
        # print(y.shape)
        # result = torch.cat((result1, result2), 1)
        # result = self.fc_concat(result)
        # f2 = self.layer2(f1_temp)
        # f2_temp = self.fpa2(f2)
        # v2 = self.global_avgpool(f2_temp)
        # f3 = self.layer3(f2_temp)
        # f3_temp = self.fpa3(f3)
        # v3 = self.global_avgpool(f3_temp)
        # f4 = self.layer4(f3_temp)
        # f4_temp = self.fpa4(f4)
        # v4 = self.global_avgpool(f4_temp)
        ###原来的
        # f = self.__feature_maps(x)
        # f1 = self.layer1(f)
        # v1 = self.global_avgpool(f1)
        # f2 = self.layer2(f1)
        # v2 = self.global_avgpool(f2)
        # print(v2.shape)
        # f3 = self.layer3(f2)
        # v3 = self.global_avgpool(f3)
        # f4 = self.layer4(f3)
        # v4 = self.global_avgpool(f4)

        # v1 = v1.view(v1.size(0), -1)
        # v2 = v2.view(v2.size(0), -1)
        # v3 = v3.view(v3.size(0), -1)
        # v4 = v4.view(v4.size(0), -1)

        # v1 = self.BN1(v1)
        # v2 = self.BN2(v2)
        # v3 = self.BN3(v3)
        # v4 = self.BN4(v4)
        #
        # y1 = self.classifier1(v1)  # type: torch.Tensor
        # y2 = self.classifier2(v2)  # type: torch.Tensor
        # y3 = self.classifier3(v3)  # type: torch.Tensor
        # y4 = self.classifier4(v4)  # type: torch.Tensor
        #
        # y1 = torch.tanh(y1)
        # y2 = torch.tanh(y2)
        # y3 = torch.tanh(y3)
        # y4 = torch.tanh(y4)
        #
        # middle_hash = [y1, y2, y3, y4]
        # middle_hash = middle_hash[4 - self.fusion_num:]
        #
        # y = self.weight(*middle_hash)


        if self.training is False:
            return y

        return middle_hash, y
    #middle_hash


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""


def resnet18(num_classes, fusion_num=4, pretrained=True, **kwargs):
    model = ResNet(
        bit=num_classes,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fusion_num=fusion_num,
        **kwargs
    )
    model.module_name = "ASCHN_resnet18"
    if pretrained:
        model.init_pretrained_weights(model_urls['resnet18'])
    return model


def resnet34(num_classes, fusion_num=4, pretrained=True, **kwargs):
    model = ResNet(
        bit=num_classes,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fusion_num=fusion_num,
        **kwargs
    )
    model.module_name = "ASCHN_resnet34"
    if pretrained:
        model.init_pretrained_weights(model_urls['resnet34'])
    return model


def resnet50(num_classes, fusion_num=4, pretrained=True, **kwargs):
    model = ResNet(
        bit=num_classes,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fusion_num=fusion_num,
        **kwargs
    )
    model.module_name = "ASCHN_resnet50"
    if pretrained:
        model.init_pretrained_weights(model_urls['resnet50'])
    return model


def resnet101(num_classes, fusion_num=4, pretrained=True, **kwargs):
    model = ResNet(
        bit=num_classes,
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        last_stride=2,
        fusion_num=fusion_num,
        **kwargs
    )
    model.module_name = "ASCHN_resnet101"
    if pretrained:
        model.init_pretrained_weights(model_urls['resnet101'])
    return model


def resnet152(num_classes, fusion_num=4, pretrained=True, **kwargs):
    model = ResNet(
        bit=num_classes,
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        last_stride=2,
        fusion_num=fusion_num,
        **kwargs
    )
    model.module_name = "ASCHN_resnet152"
    if pretrained:
        model.init_pretrained_weights(model_urls['resnet152'])
    return model