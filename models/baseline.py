import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np

import cv2

from models.resnet import resnet50
from utils.calc_acc import calc_acc

from layers import TripletLoss
from layers import CenterTripletLoss
from layers import CenterLoss
from layers import cbam
from layers import NonLocalBlockND

class Baseline(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, pattern_attention=False, modality_attention=0, mutual_learning=False, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.pattern_attention = pattern_attention
        self.modality_attention = modality_attention
        self.mutual_learning = mutual_learning

        self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride, modality_attention=modality_attention)

        self.base_dim = 2048
        self.dim = 0
        self.part_num = kwargs.get('num_parts', 0)

        if pattern_attention:
            self.base_dim = 2048
            self.dim = 2048
            self.part_num = kwargs.get('num_parts', 6)
            self.spatial_attention = nn.Conv2d(self.base_dim, self.part_num, kernel_size=1, stride=1, padding=0, bias=True)
            torch.nn.init.constant_(self.spatial_attention.bias, 0.0)
            self.activation = nn.Sigmoid()
            self.weight_sep = kwargs.get('weight_sep', 0.1)

        if mutual_learning:
            self.visible_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.infrared_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)

            self.visible_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.visible_classifier_.weight.requires_grad_(False)
            self.visible_classifier_.weight.data = self.visible_classifier.weight.data

            self.infrared_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.infrared_classifier_.weight.requires_grad_(False)
            self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data

            self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
            self.weight_sid = kwargs.get('weight_sid', 0.5)
            self.weight_KL = kwargs.get('weight_KL', 2.0)
            self.update_rate = kwargs.get('update_rate', 0.2)
            self.update_rate_ = self.update_rate

        print("output feat length:{}".format(self.base_dim + self.dim * self.part_num))
        self.bn_neck = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        nn.init.constant_(self.bn_neck.bias, 0) 
        self.bn_neck.bias.requires_grad_(False)

        if kwargs.get('eval', False):
            return

        self.classification = kwargs.get('classification', False)
        self.triplet = kwargs.get('triplet', False)
        self.center_cluster = kwargs.get('center_cluster', False)
        self.center_loss = kwargs.get('center', False)
        self.margin = kwargs.get('margin', 0.3)

        if self.classification:
            self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num , num_classes, bias=False)
        if self.mutual_learning or self.classification:
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.triplet:
            self.triplet_loss = TripletLoss(margin=self.margin)
        if self.center_cluster:
            k_size = kwargs.get('k_size', 8)
            self.center_cluster_loss = CenterTripletLoss(k_size=k_size, margin=self.margin)
        if self.center_loss:
            self.center_loss = CenterLoss(num_classes, self.base_dim + self.dim * self.part_num)

    def forward(self, inputs, labels=None, **kwargs):
        loss_reg = 0
        loss_center = 0
        modality_logits = None
        modality_feat = None

        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)
        # CNN
        global_feat = self.backbone(inputs)

        b, c, w, h = global_feat.shape

        if self.pattern_attention:
            masks = global_feat
            masks = self.spatial_attention(masks)
            masks = self.activation(masks)

            feats = []
            for i in range(self.part_num):
                mask = masks[:, i:i+1, :, :]
                feat = mask * global_feat

                feat = F.avg_pool2d(feat, feat.size()[2:])
                feat = feat.view(feat.size(0), -1)

                feats.append(feat)

            global_feat = F.avg_pool2d(global_feat, global_feat.size()[2:])
            global_feat = global_feat.view(global_feat.size(0), -1)

            feats.append(global_feat)
            feats = torch.cat(feats, 1)

            if self.training:
                masks = masks.view(b, self.part_num, w*h)
                loss_reg = torch.bmm(masks, masks.permute(0, 2, 1))
                loss_reg = torch.triu(loss_reg, diagonal = 1).sum() / (b * self.part_num * (self.part_num - 1) / 2)

        else:
            feats = F.avg_pool2d(global_feat, global_feat.size()[2:])
            feats = feats.view(feats.size(0), -1)

        if not self.training:
            feats = self.bn_neck(feats)
            return feats
        else:
            return self.train_forward(feats, labels, loss_reg, sub, **kwargs)

    def train_forward(self, feat, labels, loss_reg, sub, **kwargs):
        epoch = kwargs.get('epoch')
        metric = {}
        if self.pattern_attention and loss_reg != 0 :
            loss = loss_reg.float() * self.weight_sep
            metric.update({'p-reg': loss_reg.data})
        else:
            loss = 0

        if self.triplet:
            triplet_loss, _, _ = self.triplet_loss(feat.float(), labels)
            loss += triplet_loss
            metric.update({'tri': triplet_loss.data})

        if self.center_loss:
            center_loss = self.center_loss(feat.float(), labels)
            loss += center_loss
            metric.update({'cen': center_loss.data})

        if self.center_cluster:
            center_cluster_loss, _, _ = self.center_cluster_loss(feat.float(), labels)
            loss += center_cluster_loss
            metric.update({'cc': center_cluster_loss.data})

        feat = self.bn_neck(feat)

        if self.classification:
            logits = self.classifier(feat)
            cls_loss = self.id_loss(logits.float(), labels)
            loss += cls_loss
            metric.update({'acc': calc_acc(logits.data, labels), 'ce': cls_loss.data})

        if self.mutual_learning:
            # cam_ids = kwargs.get('cam_ids')
            # sub = (cam_ids == 3) + (cam_ids == 6)
            
            logits_v = self.visible_classifier(feat[sub == 0])
            v_cls_loss = self.id_loss(logits_v.float(), labels[sub == 0])
            loss += v_cls_loss * self.weight_sid
            logits_i = self.infrared_classifier(feat[sub == 1])
            i_cls_loss = self.id_loss(logits_i.float(), labels[sub == 1])
            loss += i_cls_loss * self.weight_sid

            logits_m = torch.cat([logits_v, logits_i], 0).float()
            with torch.no_grad():
                self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) \
                                                 + self.infrared_classifier.weight.data * self.update_rate
                self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) \
                                                 + self.visible_classifier.weight.data * self.update_rate

                logits_v_ = self.infrared_classifier_(feat[sub == 0])
                logits_i_ = self.visible_classifier_(feat[sub == 1])

                logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()
            logits_m = F.softmax(logits_m, 1)
            logits_m_ = F.log_softmax(logits_m_, 1)
            mod_loss = self.KLDivLoss(logits_m_, logits_m) 

            loss += mod_loss * self.weight_KL + (v_cls_loss + i_cls_loss) * self.weight_sid
            metric.update({'ce-v': v_cls_loss.data})
            metric.update({'ce-i': i_cls_loss.data})
            metric.update({'KL': mod_loss.data})

        return loss, metric
