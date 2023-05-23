from vfi.LiteFlowNet import FlowNet
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vfi.warp_layer import warp

class VFIModel(nn.Module):
    def __init__(self):
        super(VFIModel, self).__init__()
        
        # Optical flow network
        self.optical_flow = FlowNet()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(16, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Fusion layers
        self.fusion1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fusion2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.fusion3 = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1)
        
        # Upsampling layers
        # self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # self.upsample2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # Split into 2 input frames
        frame1 = x[:, :3, :, :]
        frame2 = x[:, 3:, :, :]

        print(f'frame1.size(): {frame1.size()}, frame2.size(): {frame2.size()}')

        intWidth = frame1.shape[3]
        intHeight = frame2.shape[2]

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tenPreprocessedOne = torch.nn.functional.interpolate(input=frame1, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tenPreprocessedTwo = torch.nn.functional.interpolate(input=frame2, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        
        # Estimate optical flow in both directions
        forward_flow = torch.nn.functional.interpolate(input=self.optical_flow(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)
        backward_flow = torch.nn.functional.interpolate(input=self.optical_flow(tenPreprocessedTwo, tenPreprocessedOne), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        forward_flow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        forward_flow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        backward_flow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        backward_flow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight) 
        
        # Warp frame2 according to forward flow
        # forward_warped_frame = F.grid_sample(frame2[0], forward_flow[0])
        forward_warped_frame = warp(frame2, forward_flow)
        
        # Warp frame1 according to backward flow
        # backward_warped_frame = F.grid_sample(frame1[0], backward_flow[0])
        backward_warped_frame = warp(frame1, backward_flow)
        
        # Concatenate inputs, warped frames and flows
        x = torch.cat([frame1, frame2, forward_warped_frame, backward_warped_frame,
                       forward_flow, backward_flow], dim=1)
        
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x1 = F.relu(self.conv5(x1))
        x2 = self.conv6(torch.cat([x1, x2]))
        
        # Fuse feature maps
        x = F.relu(self.fusion1(x2))
        x = F.relu(self.fusion2(x))
        x = self.fusion3(x)
        
        # # Upsample fused feature maps
        # x = F.relu(self.upsample1(x))
        # x = self.upsample2(x)
        
        return x
