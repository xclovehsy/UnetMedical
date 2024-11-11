import os
import sys
sys.path.append('../')

import torch
from torch import nn
import numpy as np
from model.unet import UNet
from utils.utils import *
import matplotlib.pylab as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Test:

    def __init__(self, config, logger, data_manager):
        self.config = config
        self.logger = logger

        self.unet = UNet(in_channels=1, out_channels=1).to(device)
        self.unet.load_state_dict(torch.load(config.test_checkpoint)['model_state'])

        self.test_iter = data_manager.get_test_dataloader()

    
    # def evaluate(self):
    #     """评估模型"""
    #     self.unet.eval()
    #     TP, FP, FN = 0, 0, 0
    #     intersection, union = 0, 0
    #     correct, total = 0, 0
    #     start = time.time()
        
    #     with torch.no_grad():
    #         for img, mask in self.valid_dataloader:
    #             img, mask = img.to(device), mask.to(device)
    #             logits = self.unet(img)  
    #             pred = (torch.sigmoid(logits) > 0.5)

    #             TP += (pred * mask).sum()
    #             FP += ((pred == 1) & (mask == 0)).sum()
    #             FN += ((pred == 0) & (mask == 1)).sum()

    #             tmp = torch.sum(pred * mask)
    #             intersection += tmp
    #             union += torch.sum(pred) + torch.sum(mask) - tmp

    #             correct += (pred == mask).sum()
    #             total += img.numel()
        
    #     precision = TP / (TP + FP)
    #     recall = TP / (TP + FN)
    #     f1 = 2 * (precision * recall) / (precision + recall)
    #     accuracy = correct / total
    #     iou = intersection / union
    #     end = time.time()
    #     return iou, accuracy, precision, recall, f1, end-start

    def test(self):
        """测试模型"""
        
        self.unet.train()
        with torch.no_grad():
            for img, mask in self.test_iter:
                img, mask = img.to(device), mask.to(device)
                logits = self.unet(img)  
                pred = (torch.sigmoid(logits) > 0.5)

                plt.figure(figsize=(20, 20), dpi=80)
                for i in range(self.config.batch_size):
                    ax = plt.subplot(3, 4, i + 1)
                    ax.imshow(img[i].permute(1, 2, 0).cpu())
                    ax = plt.subplot(3, 4, i + 1 + 4)
                    ax.imshow(mask[i].permute(1, 2, 0).cpu())
                    ax = plt.subplot(3, 4, i + 1 + 8)
                    ax.imshow(pred[i].permute(1, 2, 0).cpu())
                
                plt.show()
                
                

                break

        