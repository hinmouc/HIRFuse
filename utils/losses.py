# -*-coding:utf-8 -*-

# File       : losses.py
# Author     : hingmauc
# Time       : 2024/10/23 14:29
# Description：

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Losses1(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = kornia.losses.SSIMLoss(11, reduction='mean')

    def gradient(self, x):
        return kornia.filters.SpatialGradient()(x)

    def forward(self, vi, ir, vi_hat, ir_hat):
        total_loss = 1 * self.mse(vi,vi_hat) + 5 * self.ssim(vi,vi_hat)+\
                     1 * self.mse(ir,ir_hat) + 5 * self.ssim(ir,ir_hat)

        print("Total Loss:", total_loss.item())
        return total_loss


class Losses2(nn.Module):
    def __init__(self):
        super().__init__()

    def Fusionloss_grad(self, img_F, img_A, img_B):
        def sobel_filter(tensor):
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            sobel_x = sobel_x.repeat(1, tensor.shape[1], 1, 1).to(tensor.device)
            sobel_y = sobel_y.repeat(1, tensor.shape[1], 1, 1).to(tensor.device)

            grad_x = F.conv2d(tensor, sobel_x, padding=1)
            grad_y = F.conv2d(tensor, sobel_y, padding=1)

            return grad_x, grad_y

        sobelx_f, sobely_f = sobel_filter(img_F)
        sobelx_a, sobely_a = sobel_filter(img_A)
        sobelx_b, sobely_b = sobel_filter(img_B)

        sobelx_max = (torch.abs(sobelx_a) >= torch.abs(sobelx_b)) * sobelx_a + (
                torch.abs(sobelx_a) < torch.abs(sobelx_b)) * sobelx_b
        sobely_max = (torch.abs(sobely_a) >= torch.abs(sobely_b)) * sobely_a + (
                torch.abs(sobely_a) < torch.abs(sobely_b)) * sobely_b

        return F.l1_loss(sobelx_f, sobelx_max) + F.l1_loss(sobely_f, sobely_max)

    def l1_loss(self, Y_f, vi, ir):
        loss = F.mse_loss((ir - Y_f), torch.zeros_like(Y_f)) + F.mse_loss((vi - Y_f), torch.zeros_like(Y_f))
        return loss

    def forward(self, Y_f, vi, ir):
        ir = TF.adjust_contrast(ir, 1.5)
        total_loss = 20 * self.Fusionloss_grad(Y_f, vi, ir) + self.l1_loss(Y_f, vi, ir)

        print("Total Loss:", total_loss.item())
        return total_loss


