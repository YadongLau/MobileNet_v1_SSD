import torch
import torch.nn as nn
from torchsummary import summary


class BottleNeck(nn.Module):
    def __init__(self,input_channels, output_channels,strides,expansion):
        super(BottleNeck,self).__init__()
    # def bottleneck(input_channels, output_channels, strides,expansion):
        self.l_1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=strides, bias=False)
        self.l_2 = nn.BatchNorm2d(output_channels)
        self.l_3 = nn.ReLU(inplace=True)
        self.l_4 = nn.Conv2d(output_channels, out_channels=output_channels, kernel_size=3, stride=strides * expansion, bias=False)
        self.l_5 = nn.BatchNorm2d(64)
        self.l_6 = nn.ReLU(inplace=True)
        self.l_7 = nn.Conv2d(output_channels, out_channels=output_channels*4, kernel_size=1, stride=strides, bias=False)
        self.l_8 = nn.BatchNorm2d(64)
        #self.layers = [l_1, l_2, l_3, l_4, l_5, l_6, l_7, l_8]

    def forward(self, x):
        out = self.l_1(x)
        out = self.l_2(out)
        out = self.l_3(out)
        out = self.l_4(out)
        out = self.l_5(out)
        out = self.l_6(out)
        out = self.l_7(out)
        out = self.l_8(out)

        return out


def identity_block(in_channel, app):
    tlayers = []
    tlayers += [nn.Conv2d(in_channels=in_channel, out_channels=in_channel*4, kernel_size=1, stride=1*app)]
    tlayers += [nn.BatchNorm2d(in_channel*4)]
    return tlayers


def resnet50(i):
    in_channels = i
    layers = []

    # =========================================================
    #                      conv1_x
    # =========================================================
    layers += [nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)]
    layers += [nn.BatchNorm2d(64)]
    layers += [nn.ReLU(inplace=True)]

    # =========================================================
    #                      conv2_x    3
    # =========================================================
    layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=2)
    layers += identity_block(in_channel=64, app=2)
    layers += [nn.ReLU(inplace=True)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += [nn.ReLU(inplace=True)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += [nn.ReLU(inplace=True)]

    # =========================================================
    #                      conv3_x       4
    # =========================================================
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=2)
    layers += identity_block(in_channel=64, app=2)
    layers += [nn.ReLU(inplace=True)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += [nn.ReLU(inplace=True)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += [nn.ReLU(inplace=True)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += [nn.ReLU(inplace=True)]

    # =========================================================
    #                      conv4_x       6
    # =========================================================
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += identity_block(in_channel=64, app=1)
    layers += [nn.ReLU(inplace=True)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += [nn.ReLU(inplace=True)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += [nn.ReLU(inplace=True)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += [nn.ReLU(inplace=True)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += [nn.ReLU(inplace=True)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += [nn.ReLU(inplace=True)]

    # =========================================================
    #                      conv5_x       3
    # =========================================================
    layers += BottleNeck(input_channels=64, output_channels=64, strides=2, expansion=2)
    layers += identity_block(in_channel=64, app=2)
    layers += [nn.ReLU(inplace=True)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += [nn.ReLU(inplace=True)]
    layers += BottleNeck(input_channels=64, output_channels=64, strides=1, expansion=1)
    layers += [nn.ReLU(inplace=True)]

    return layers



