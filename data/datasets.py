# -*-coding:utf-8 -*-

# File       : datasets.py
# Author     : hingmauc
# Time       : 2024/7/23 16:46
# Description：

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

class VIRDataset(Dataset):
    def __init__(self, vi_path, ir_path, transform=None):
        self.transform = transform
        self.ir_paths = ir_path
        self.vi_paths = vi_path

    def __len__(self):
        return len(self.ir_paths)

    def __getitem__(self, idx):
        ir_image = self.load_image(self.ir_paths[idx], color_mode='L')
        vi_image = self.load_image(self.vi_paths[idx], color_mode='RGB')
        vi_image_ycbcr = self.rgb_to_ycbcr(vi_image)
        vi_image_y = vi_image_ycbcr[:, :, 0]

        ir_image = Image.fromarray((ir_image * 255).astype(np.uint8))
        vi_image_y = Image.fromarray((vi_image_y * 255).astype(np.uint8))

        if self.transform:
            ir_image = self.transform(ir_image)
            vi_image_y = self.transform(vi_image_y)
        return vi_image_y, ir_image

    def load_image(self,file, color_mode='RGB'):
        with Image.open(file) as img:
            if color_mode == 'L':
                img = img.convert('L')
            elif color_mode == 'RGB':
                img = img.convert('RGB')
            img = np.array(img, dtype=np.float32) / 255.0
        return img

    def rgb_to_ycbcr(self,image):
        img_ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        return img_ycbcr
