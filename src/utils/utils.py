import torch
import numpy as np
from torch import nn

def calculate_iou(pred, target):
    """计算二值掩码IoU"""
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    iou = intersection / union
    return iou

def calculate_acc(pred, target):
    """计算准确率"""
    return (pred == target).sum() / target.numel()


def calculate_precision_recall_f1(pred, target):
    """计算精确率，召回率，F1"""
    TP = (pred * target).sum()
    FP = ((pred == 1) & (target == 0)).sum()
    FN = ((pred == 0) & (target == 1)).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall , f1


def init_weights(m):
    """初始化模型权重"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)