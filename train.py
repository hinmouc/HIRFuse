# -*-coding:utf-8 -*-

# File       : trains.py
# Author     : hingmauc
# Time       : 2024/7/20 19:15
# Description：

import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.losses import Losses1, Losses2
from data.datasets  import VIRDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
    ---------------- model ---------------------
'''
from net.model import Encoder, Decoder, IterativeFusion
class HIRFuse(nn.Module):
    def __init__(self,
                 dim=64,
                 heat_res=(128,128),
                 depth=3):
        super(HIRFuse, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)

        self.iterative_fusion = IterativeFusion(in_channels=dim,heat_res=heat_res).to(device)

    def forward(self, data_VIS, data_IR):
        feature_VI= self.encoder(data_VIS)
        feature_IR= self.encoder(data_IR)

        fusion_feature = self.iterative_fusion(feature_VI, feature_IR)

        data_Fuse = self.decoder(data_VIS, fusion_feature)

        return data_Fuse


'''
    ---------------- train ---------------------
'''
class Trainer:
    def __init__(self, model, train_loader, learning_rate=1e-4, weight_decay=0,
                 optim_step=20, optim_gamma=0.5, clip_grad_norm_value = 0.01):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.clip_grad_norm_value = clip_grad_norm_value

        self.optimizers = [
            torch.optim.Adam(self.model.encoder.parameters(), lr=learning_rate, weight_decay=weight_decay),
            torch.optim.Adam(self.model.decoder.parameters(), lr=learning_rate, weight_decay=weight_decay),
            torch.optim.Adam(self.model.iterative_fusion.parameters(), lr=learning_rate, weight_decay=weight_decay)
        ]

        self.schedulers = [
            torch.optim.lr_scheduler.StepLR(self.optimizers[0], step_size=optim_step, gamma=optim_gamma),
            torch.optim.lr_scheduler.StepLR(self.optimizers[1], step_size=optim_step, gamma=optim_gamma),
            torch.optim.lr_scheduler.StepLR(self.optimizers[2], step_size=optim_step, gamma=optim_gamma)
        ]

        self.phase = 'phase1'
        self.losses_stage1 = Losses1()
        self.losses_stage2 = Losses2()
        self.num_epochs = 2
        self.epoch_gap = 1

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1} of {self.num_epochs}")
            for i, data in enumerate(self.train_loader):

                data_VIS, data_IR = data
                data_VIS = data_VIS.cuda()
                data_IR = data_IR.cuda()

                self._zero_grad()

                if epoch < self.epoch_gap:
                    self.phase = 'phase1'
                    feature_VIS = self.model.encoder(data_VIS)
                    feature_IR = self.model.encoder(data_IR)
                    data_VIS_hat = self.model.decoder(data_VIS, feature_VIS)
                    data_IR_hat = self.model.decoder(data_IR, feature_IR)

                    loss = self.losses_stage1(data_VIS, data_IR, data_VIS_hat, data_IR_hat)
                else:
                    self.phase = 'phase2'
                    data_Fuse = self.model(data_VIS, data_IR)
                    loss = self.losses_stage2(data_Fuse, data_VIS, data_IR)

                print("%s Epoch: [%2d/%2d] [%4d/%4d]" \
                      % ("HIRFuse", (epoch + 1), self.num_epochs, i + 1, self.train_loader.__len__()))

                loss.backward()
                self._step_optimizers(self.phase)
                self._step_schedulers(self.phase)

            if epoch == self.num_epochs - 1:
                self.save_checkpoint()

    def _zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def _step_optimizers(self, phase):
        if phase == 'phase1':
            for optimizer in self.optimizers[:2]:
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], self.clip_grad_norm_value, norm_type=2)
                optimizer.step()
        else:
            for optimizer in self.optimizers:
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], self.clip_grad_norm_value, norm_type=2)
                optimizer.step()

    def _step_schedulers(self, phase, val_loss=None):
        if phase == 'phase1':
            for scheduler in self.schedulers[:2]:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        else:
            for scheduler in self.schedulers:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        self._check_and_fix_lr()

    def _check_and_fix_lr(self):
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                if param_group['lr'] < 1e-6:
                    param_group['lr'] = 1e-6

    def save_checkpoint(self):
        folder_path = f'checkpoint/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        checkpoint = {
            'Encoder': self.model.encoder.state_dict(),
            'Decoder': self.model.decoder.state_dict(),
            'IterativeFuseLayer': self.model.iterative_fusion.state_dict()
        }
        file_path = os.path.join(folder_path, f"checkpoint.pth")
        torch.save(checkpoint, file_path)


def main():
    vi_paths = '''...'''
    ir_paths = '''...'''

    transform = transforms.Compose([
        transforms.RandomResizedCrop((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    train_dataset = VIRDataset(vi_paths, ir_paths, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=4,
                              shuffle=True, num_workers=0)

    model = HIRFuse().to(device)
    trainer = Trainer(model, train_loader=train_loader)
    trainer.train()

if __name__ == '__main__':
    main()