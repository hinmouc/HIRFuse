# -*-coding:utf-8 -*-

# File       : test.py
# Author     : hingmauc
# Time       : 2026/04/05 14:20
# Description:

import os
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from utils.utils import *
from net.model import Heat2D
from train import HIRFuse


#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(model, path_ir, path_vi, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def enable_heat2d_infer_mode(model):
        for module in model.modules():
            if isinstance(module, Heat2D):
                module.infer_mode = True

    model.eval()  # 设置模型为评估模式
    enable_heat2d_infer_mode(model)
    with torch.no_grad():
        for imgname in tqdm(os.listdir(path_ir)):
            image_ir = image_read_cv2(os.path.join(path_ir, imgname), 'GRAY')[np.newaxis, np.newaxis, ...] / 255
            image_vis = image_read_cv2(os.path.join(path_vi, imgname), 'YCrCb')[:, :, 0][np.newaxis, np.newaxis, ...] / 255

            ir = (torch.FloatTensor(image_ir)).to(device)
            vi = (torch.FloatTensor(image_vis)).to(device)

            data_Fuse = model(vi, ir)

            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())

            color_image = cv2.imread(os.path.join(path_vi, imgname))
            color_image_ycbcr = cv2.cvtColor(color_image, cv2.COLOR_BGR2YCrCb)
            color_image_ycbcr[:, :, 0] = fused_image
            final_image = cv2.cvtColor(color_image_ycbcr, cv2.COLOR_YCrCb2BGR)

            cv2.imwrite(os.path.join(save_path, imgname.split(sep='.')[0] + '.png'), final_image)


def main():
    path_ir = r"test_img\ir"
    path_vi = r"test_img\vi"
    path_save = r"test_result"

    check_path = r"checkpoint/checkpoint.pth"

    model = HIRFuse().to(device)
    checkpoint = torch.load(check_path)
    model.encoder.load_state_dict(checkpoint['Encoder'])
    model.decoder.load_state_dict(checkpoint['Decoder'])
    model.iterative_fusion.load_state_dict(checkpoint['IterativeFuseLayer'])

    model.eval()

    eval(model, path_ir, path_vi, path_save)


if __name__ == '__main__':
    main()