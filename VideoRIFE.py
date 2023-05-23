import os
import torch
from rife.model.RIFE import Model
from torch.nn import functional as F
import numpy as np
import cv2
import gc


class rife_model:
    model_path = 'model_weights/rife'
    fp16 = True

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        self.model = Model(local_rank=-1)
        self.model.load_model(self.model_path, -1)
        self.model.cuda().eval()
        # self.model.device()

        self.scale = 0.5

    # def pad_image(self, img, h, w):
    #     tmp = max(32, int(32 / self.scale))
    #     ph = ((h - 1) // tmp + 1) * tmp
    #     pw = ((w - 1) // tmp + 1) * tmp
    #     padding = (0, pw - w, 0, ph - h)

    #     if(self.fp16):
    #         return F.pad(img.float(), padding).half()
    #     else:
    #         return F.pad(img.float(), padding)
        
    # def pad_image(self, img, h, w):
    #     # h, w, _ = img.shape
    #     ph = 64 - (h % 64)
    #     pw = 64 - (w % 64)
    #     img = np.pad(img, [(0, ph), (0, pw), (0, 0)], 'edge')
    #     img = img.astype(np.float32)  # cast the input tensor to float32
    #     return img

    def inference(self, img0, img1):
        # torch.set_default_tensor_type(torch.cuda.HalfTensor)
        with torch.no_grad():
            h, w, _ = img0.shape
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(self.device, non_blocking=True)).unsqueeze(0).float() / 255.
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(self.device, non_blocking=True)).unsqueeze(0).float() / 255.
            # img0 = self.pad_image(img0, h, w).to(self.device)
            # img1 = self.pad_image(img1, h, w).to(self.device)
            output = self.model.inference(img0=img0, img1=img1, scale=2)
            # print(f"output: {output}")
            for mid in output:
                # print(f'mid: {mid}')
                mid = (mid[0] * 255.).byte().cpu().numpy()
                # print(mid.shape)
                # mid = ((mid.transpose(1, 2, 0)))
                # cv2.imshow('interpolated', mid)
                # cv2.waitKey(1)
            return mid
