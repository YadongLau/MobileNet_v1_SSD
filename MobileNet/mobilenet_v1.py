import torch.nn as nn
from torchsummary import summary


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


# def conv_dw(inp, oup, stride):
#     return nn.Sequential(
#         # dw
#         nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#         nn.BatchNorm2d(inp),
#         nn.ReLU(inplace=True),
#
#         # pw
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True),
#     )

# def conv_dw(inp, oup, stride):
#         # dw
#         [nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#         nn.BatchNorm2d(inp),
#         nn.ReLU(inplace=True),
#         # pw
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True)]

def mobilenet_v1(i):
    layers = []
    # conv_bn
    layers += [nn.Conv2d(i, 32, kernel_size=3, stride=2, bias=False)]
    layers += [nn.BatchNorm2d(32)]
    layers += [nn.ReLU(inplace=True)]
# 1
    layers += [nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False)]
    layers += [nn.BatchNorm2d(32)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(32, 64, 1, 1, 0, bias=False)]
    layers += [nn.BatchNorm2d(64)]
    layers += [nn.ReLU(inplace=True)]
    # 2
    layers += [nn.Conv2d(64, 64, 3, 2, 1, groups=64, bias=False)]
    layers += [nn.BatchNorm2d(64)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(64, 128, 1, 1, 0, bias=False)]
    layers += [nn.BatchNorm2d(128)]
    layers += [nn.ReLU(inplace=True)]

    # 3
    layers += [nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False)]
    layers += [nn.BatchNorm2d(128)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(128, 128, 1, 1, 0, bias=False)]
    layers += [nn.BatchNorm2d(128)]
    layers += [nn.ReLU(inplace=True)]

    # 4
    layers += [nn.Conv2d(128, 128, 3, 2, 1, groups=128, bias=False)]
    layers += [nn.BatchNorm2d(128)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(128, 256, 1, 1, 0, bias=False)]
    layers += [nn.BatchNorm2d(256)]
    layers += [nn.ReLU(inplace=True)]

    # 5
    layers += [nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False)]
    layers += [nn.BatchNorm2d(256)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(256, 256, 1, 1, 0, bias=False)]
    layers += [nn.BatchNorm2d(256)]
    layers += [nn.ReLU(inplace=True)]

    # 6
    layers += [nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False)]
    layers += [nn.BatchNorm2d(256)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(256, 512, 1, 1, 0, bias=False)]
    layers += [nn.BatchNorm2d(512)]
    layers += [nn.ReLU(inplace=True)]

    # 7
    layers += [nn.Conv2d(512, 512, 3, 2, 0, groups=512, bias=False)]
    layers += [nn.BatchNorm2d(512)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, 1, 1, 0, bias=False)]
    layers += [nn.BatchNorm2d(512)]
    layers += [nn.ReLU(inplace=True)]

    # 8
    layers += [nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False)]
    layers += [nn.BatchNorm2d(512)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, 1, 1, 0, bias=False)]
    layers += [nn.BatchNorm2d(512)]
    layers += [nn.ReLU(inplace=True)]

    # 9
    layers += [nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False)]
    layers += [nn.BatchNorm2d(512)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, 1, 1, 0, bias=False)]
    layers += [nn.BatchNorm2d(512)]
    layers += [nn.ReLU(inplace=True)]

    # 10
    # layers += [nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False)]
    # layers += [nn.BatchNorm2d(512)]
    # layers += [nn.ReLU(inplace=True)]
    # layers += [nn.Conv2d(512, 512, 1, 1, 0, bias=False)]
    # layers += [nn.BatchNorm2d(512)]
    # layers += [nn.ReLU(inplace=True)]

    # 11
    layers += [nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False)]
    layers += [nn.BatchNorm2d(512)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, 1, 1, 0, bias=False)]
    layers += [nn.BatchNorm2d(512)]
    layers += [nn.ReLU(inplace=True)]

    # 12
    layers += [nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False)]
    layers += [nn.BatchNorm2d(512)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 1024, 1, 1, 0, bias=False)]
    layers += [nn.BatchNorm2d(1024)]
    layers += [nn.ReLU(inplace=True)]
    #
    # # 13
    layers += [nn.Conv2d(1024, 1024, 1, 1, 1, groups=1024, bias=False)]
    # layers += [nn.BatchNorm2d(1024)]
    # layers += [nn.ReLU(inplace=True)]
    # layers += [nn.Conv2d(1024, 1024, 1, 1, 0, bias=False)]
    # layers += [nn.BatchNorm2d(1024)]
    # layers += [nn.ReLU(inplace=True)]

    # conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    # layers += [conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers




#
# class MobileNetV1(nn.Module):
#     def __init__(self, ch_in, n_classes):
#         super(MobileNetV1, self).__init__()
#
#         def conv_bn(inp, oup, stride):
#             return nn.Sequential(
#                 nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True)
#                 )
#
#         def conv_dw(inp, oup, stride):
#             return nn.Sequential(
#                 # dw
#                 nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#                 nn.BatchNorm2d(inp),
#                 nn.ReLU(inplace=True),
#
#                 # pw
#                 nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True),
#                 )
#
#         self.model = nn.Sequential(
#             conv_bn(ch_in, 32, 2),  # 3
#             conv_dw(32, 64, 1),     # 9
#             conv_dw(64, 128, 2),    # 15
#             conv_dw(128, 128, 1),   # 21
#             conv_dw(128, 256, 2),   # 27
#             conv_dw(256, 256, 1),   # 33
#             # conv_dw(256, 512, 2),
#             # conv_dw(512, 512, 1),
#             # conv_dw(512, 512, 1),
#             # conv_dw(512, 512, 1),
#             # conv_dw(512, 512, 1),
#             # conv_dw(512, 512, 1),
#             # conv_dw(512, 1024, 2),
#             # conv_dw(1024, 1024, 1),
#             # nn.AdaptiveAvgPool2d(1)
#         )
#         # self.fc = nn.Linear(1024, n_classes)
#
#     def forward(self, x):
#         x = self.model(x)
#         x = x.view(-1, 1024)
#         # x = self.fc(x)
#         return x
#
# if __name__=='__main__':
#     # model check
#     model = MobileNetV1(ch_in=3, n_classes=1000)
#     summary(model, input_size=(3, 300, 300), device='cpu')