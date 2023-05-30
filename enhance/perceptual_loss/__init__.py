import torch
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(torch.nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.device = device
        self.vgg = models.vgg16().features.to(self.device)
        self.vgg.eval()

    def forward(self, predicted_frame, ground_truth_frame):
        # Normalize the input frames
        predicted_frame_norm = F.normalize(predicted_frame, p=2, dim=1)
        ground_truth_frame_norm = F.normalize(ground_truth_frame, p=2, dim=1)

        # Extract high-level features from the VGG-16 network
        predicted_features = self.vgg(predicted_frame_norm)
        ground_truth_features = self.vgg(ground_truth_frame_norm)

        # Compute the mean squared error between the feature maps
        perceptual_loss = F.mse_loss(predicted_features, ground_truth_features)

        return perceptual_loss