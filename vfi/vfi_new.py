from vfi.LiteFlowNet import FlowNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class VFIModel(nn.Module):
    def __init__(self) -> None:
        super(VFIModel, self).__init__()

        # Optical flow network
        self.optical_flow = FlowNet()

        # Convolutional layers
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        # Fusion layers
        self.fusion1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fusion2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.fusion3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.fusion4 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        # self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        # self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # self.upsample4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Split into 2 input frames
        frame1 = x[:, :3, :, :]
        frame2 = x[:, 3:, :, :]

        print(f'frame1: {frame1.shape}, frame2: {frame2.shape}')

        # Resize frame1 to match the size of forward_warped_frame
        frame1_resized = F.interpolate(frame1, size=(128, 224), mode='bilinear', align_corners=False)

        # Resize frame2 to match the size of backward_warped_frame
        frame2_resized = F.interpolate(frame2, size=(128, 224), mode='bilinear', align_corners=False)

        # Estimate optical flow in both directions
        forward_flow = self.optical_flow(frame1, frame2)
        backward_flow = self.optical_flow(frame2, frame1)

        # print(f'forward_flow: {forward_flow.shape}, permute: {forward_flow.permute(0, 2, 3, 1).shape}')
        # print(f'backward_flow: {backward_flow.shape}, permute: {backward_flow.permute(0, 2, 3, 1).shape}')

        # Warp frame2 according to forward flow
        forward_warped_frame = F.grid_sample(frame2, forward_flow.permute(0, 2, 3, 1), align_corners=True)

        # Warp frame1 according to backward flow
        backward_warped_frame = F.grid_sample(frame1, backward_flow.permute(0, 2, 3, 1), align_corners=True)

        # print(f'frame1: {frame1.shape}, frame2: {frame2.shape}, forward_warped_frame: {forward_warped_frame.shape}, backward_warped_frame: {backward_warped_frame.shape}')
        # print(f'frame1_resized: {frame1_resized.shape}, frame2_resized: {frame2_resized.shape}, \
        #       forward_warped_frame: {forward_warped_frame.shape}, backward_warped_frame: {backward_warped_frame.shape}, \
        #       forward_flow: {forward_flow.shape}, backward_flow: {backward_flow.shape}')

        # Concatenate inputs, warped frames and flows
        x = torch.cat([frame1_resized, frame2_resized, forward_warped_frame, backward_warped_frame, forward_flow, backward_flow], dim=1)

        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)

        # Fuse feature maps
        x = F.relu(self.fusion1(x))
        x = F.relu(self.fusion2(x))
        x = F.relu(self.fusion3(x))
        x = self.fusion4(x)

        # # Upsample fused feature maps
        # x = F.relu(self.upsample1(x))
        # x = F.relu(self.upsample2(x))
        # x = F.relu(self.upsample3(x))
        # x = self.upsample4(x)

        return x