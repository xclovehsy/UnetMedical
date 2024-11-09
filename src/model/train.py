import os
import sys
sys.path.append('../')

import time
import torch
from torch import nn
import numpy as np
from model.unet import UNet
from utils.utils import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Train:

    def __init__(self, config, logger, data_manager):
        self.config = config
        self.logger = logger

        self.unet = UNet(in_channels=1, out_channels=1).to(device)
        self.unet.apply(init_weights)

        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=config.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_dataloader = data_manager.get_train_dataloader()
        self.valid_dataloader = data_manager.get_valid_dataloader()

        self.writer = SummaryWriter(config.tensorboard_dir)
    
    def evaluate(self):
        """评估模型"""
        self.unet.eval()
        TP, FP, FN = 0, 0, 0
        intersection, union = 0, 0
        correct, total = 0, 0
        start = time.time()
        
        with torch.no_grad():
            for img, mask in self.valid_dataloader:
                img, mask = img.to(device), mask.to(device)
                logits = self.unet(img)  
                pred = (torch.sigmoid(logits) > 0.5)

                TP += (pred * mask).sum()
                FP += ((pred == 1) & (mask == 0)).sum()
                FN += ((pred == 0) & (mask == 1)).sum()

                tmp = torch.sum(pred * mask)
                intersection += tmp
                union += torch.sum(pred) + torch.sum(mask) - tmp

                correct += (pred == mask).sum()
                total += img.numel()
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = correct / total
        iou = intersection / union
        end = time.time()
        return iou, accuracy, precision, recall, f1, end-start

    def train(self):
        """训练模型"""

        global_step = 0
        iou_prev = 0
        dataloader_size = len(self.train_dataloader)
        
        for epoch in range(1, self.config.epochs + 1):
            tr_loss = 0
            
            # Training
            self.unet.train()
            for step, (img, mask) in enumerate(self.train_dataloader):
                img, mask = img.to(device), mask.to(device)

                logits = self.unet(img)  
                loss = self.criterion(logits, mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                global_step += 1
                tr_loss += loss.item()
                if step % self.config.log_step == 0:
                    acc = calculate_acc(logits, mask)
                    precision, recall, f1 = calculate_precision_recall_f1(logits, mask)
                    iou = calculate_iou(logits, mask)

                    self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                    self.writer.add_scalar('Train/Accuracy', acc, global_step)
                    self.writer.add_scalar('Train/Precision', precision, global_step)
                    self.writer.add_scalar('Train/Recall', recall, global_step)
                    self.writer.add_scalar('Train/F1', f1, global_step)
                    self.writer.add_scalar('Train/IoU', iou, global_step)

                    self.logger.info(f"Epoch:{epoch}-{step}/{dataloader_size}, Loss:{loss.item():.6f}, IoU:{iou*100:.2f}")
    
            self.logger.info(f"Epoch:{epoch} completed, Total training's Loss: {tr_loss:.6f}")

            # Evaluating
            iou, accuracy, precision, recall, f1, spend = self.evaluate()
            self.logger.info(f"Epoch:{epoch}, IoU:{iou*100:.2f}, Acc:{accuracy*100:.2f}, Precision: {precision*100:.2f}, Recall: {recall*100:.2f}, F1: {f1*100:.2f} on Vaild Dataset, Spend: {spend/60.0:.2f} minutes for evaluation")
            self.writer.add_scalar('Vaild/Accuracy', accuracy, epoch)
            self.writer.add_scalar('Vaild/Precision', precision, epoch)
            self.writer.add_scalar('Vaild/Recall', recall, epoch)
            self.writer.add_scalar('Vaild/F1', f1, epoch)
            self.writer.add_scalar('Vaild/IoU', iou, epoch)

            # Save latest mode parameter
            torch.save({'epoch': epoch, 'model_state': self.unet.state_dict(), 'valid_iou': iou, 'valid_acc': accuracy, 'valid_f1': f1, 'global_step': global_step}, os.path.join(self.config.checkpoints_dir, f'latest_unet_checkpoint.pt'))
            
            # Save best model parameter
            if iou > iou_prev:
                torch.save({'epoch': epoch, 'model_state': self.unet.state_dict(), 'valid_iou': iou, 'valid_acc': accuracy, 'valid_f1': f1}, os.path.join(self.config.checkpoints_dir, f'best_unet_checkpoint.pt'))
                iou_prev = iou

        