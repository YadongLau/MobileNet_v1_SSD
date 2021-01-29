import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.config import Config
from nets.ssd_layers import Detect
from nets.ssd_layers import L2Norm,PriorBox
from nets.vgg import vgg as add_vgg
import numpy as np
from math import sqrt as sqrt
from MobileNet.mobilenet_v1 import mobilenet_v1 as add_mnet

class SSD(nn.Module):
    def __init__(self, phase, base, extras, head, num_classes,feature_map_size):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = Config
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.priorbox = PriorBox(self.cfg, feature_map_size)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.Boxes = self.Cal_Priors()
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if phase == 'test' or 'trace':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 20, 0.01, 0.45)
    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # 获得conv4_3的内容
        # for k in range(23):
        #     x = self.vgg[k](x)
        for k in range(39):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # 获得fc7的内容
        # for k in range(23, len(self.vgg)):
        #     x = self.vgg[k](x)
        # sources.append(x)
        for k in range(39, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # 获得后面的内容
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # 添加回归层和分类层
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 进行resize
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1,
                self.num_classes)),  # conf preds
                self.priors
            )

        elif self.phase == "trace":
           output = torch.cat(
                (loc.view(loc.size(0), -1, 4), self.softmax(conf.view(conf.size(0),-1, self.num_classes))), 2)
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def Cal_Priors(self):
        mean = []
        for k, f in enumerate(self.priorbox.feature_maps):
            x, y = np.meshgrid(np.arange(f), np.arange(f))
            x = x.reshape(-1)
            y = y.reshape(-1)
            for i, j in zip(y, x):
                f_k = self.priorbox.image_size / self.priorbox.steps[k]
                # 计算网格的中心
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 求短边
                s_k = self.priorbox.min_sizes[k] / self.priorbox.image_size
                mean += [cx, cy, s_k, s_k]

                # 求长边
                s_k_prime = sqrt(s_k * (self.priorbox.max_sizes[k] / self.priorbox.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 获得长方形
                for ar in self.priorbox.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # 获得所有的先验框
        output = torch.Tensor(mean).view(-1, 4)

        if self.priorbox.clip:
            output.clamp_(max=1, min=0)
        return Variable(output)

    def Cal_CofLoc(self,loc_data,conf_data,prior_data):
        loc_data = loc_data.cpu()
        conf_data = conf_data.cpu()
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, 20, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)
        # 对每一张图片进行处理
        for i in range(num):
            # 对先验框解码获得预测框
            decoded_boxes = self.Cal_decode(loc_data[i], prior_data, self.detect.variance)
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                # 对每一类进行非极大抑制
                c_mask = conf_scores[cl].gt(0.01)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # 进行非极大抑制
                ids, count = self.Cal_nms(boxes, scores, 0.45, 20)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        # flt = output.contiguous().view(num, -1, 5)
        # _, idx = flt[:, :, 0].sort(1, descending=True)
        # _, rank = idx.sort(1)
        # flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

    def Cal_decode(self,loc, priors, variances):
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def Cal_nms(self,boxes, scores, overlap=0.5, top_k=200):
        keep = scores.new(int(scores.size(0))).zero_().long()
        if boxes.numel() == 0:
            return keep
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)
        v, idx = scores.sort(0)
        idx = idx[-top_k:]


        count = 0
        while idx.numel() > 0:
            i = idx[-1]
            keep[count] = i
            count += 1
            if idx.size(0) == 1:
                break
            idx = idx[:-1]
            w = torch.clamp(torch.index_select(x2, 0, idx), max=float(x2[i])) \
                - torch.clamp(torch.index_select(x1, 0, idx), min=float(x1[i]))
            h = torch.clamp(torch.index_select(y2, 0, idx), max=float(y2[i])) \
                - torch.clamp(torch.index_select(y1, 0, idx), min=float(y1[i]))
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            inter = w * h
            rem_areas = torch.index_select(area, 0, idx)
            union = (rem_areas - inter) + area[i]
            IoU = inter / union
            idx = idx[IoU.le(overlap)]
        return keep, count


def add_extras(i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i

    # Block 6
    # 19,19,1024 -> 10,10,512
    layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

    # Block 7
    # 10,10,512 -> 5,5,256
    layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    # Block 8
    # 5,5,256 -> 3,3,256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    
    # Block 9
    # 3,3,256 -> 1,1,256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    return layers


mbox = [4, 6, 6, 6, 4, 4]


def get_ssd(phase,num_classes, feature_map_size):

    # vgg, extra_layers = add_vgg(3), add_extras(1024)
    mnet, extra_layers = add_mnet(3), add_extras(1024)
    loc_layers = []
    conf_layers = []
    # vgg_source = [21, -2]
    vgg_source = [39, 69]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(mnet[v].out_channels,
                                 mbox[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(mnet[v].out_channels,
                        mbox[k] * num_classes, kernel_size=3, padding=1)]
                        
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                  * num_classes, kernel_size=3, padding=1)]

    SSD_MODEL = SSD(phase, mnet, extra_layers, (loc_layers, conf_layers), num_classes, feature_map_size)
    return SSD_MODEL
