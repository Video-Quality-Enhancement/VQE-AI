from LiteFlowNet import FlowNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class VFIModel(nn.Module):
    def __init__(self):
        super(VFIModel, self).__init__()
        
        # Optical flow network
        self.optical_flow = FlowNet()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Fusion layers
        self.fusion1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fusion2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.fusion3 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        
        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # Split into 2 input frames
        frame1 = x[:, :3, :, :]
        frame2 = x[:, 3:, :, :]
        
        # Estimate optical flow in both directions
        forward_flow = self.optical_flow(frame1, frame2)
        backward_flow = self.optical_flow(frame2, frame1)
        
        # Warp frame2 according to forward flow
        forward_warped_frame = F.grid_sample(frame2, forward_flow)
        
        # Warp frame1 according to backward flow
        backward_warped_frame = F.grid_sample(frame1, backward_flow)
        
        # Concatenate inputs, warped frames and flows
        x = torch.cat([frame1, frame2, forward_warped_frame, backward_warped_frame,
                       forward_flow, backward_flow], dim=1)
        
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Fuse feature maps
        x = F.relu(self.fusion1(x))
        x = F.relu(self.fusion2(x))
        x = self.fusion3(x)
        
        # # Upsample fused feature maps
        # x = F.relu(self.upsample1(x))
        # x = self.upsample2(x)
        
        return x
