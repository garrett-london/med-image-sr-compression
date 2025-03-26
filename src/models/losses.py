"""
Module: losses
Description:
    This module implements custom loss functions used in image super-resolution tasks.
    It includes a PerceptualLoss class that computes a perceptual loss by comparing feature
    maps extracted from a pre-trained VGG19 model. This loss is typically combined with
    pixel-wise losses (e.g., MSE, MAE) and adversarial losses in GAN-based training frameworks.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using a pre-trained VGG19 network.

    This loss measures the similarity between the feature representations of a super-resolved
    image and its corresponding high-resolution ground truth. It leverages a specific layer of
    the VGG19 network to extract features and computes the Mean Squared Error (MSE) between them.

    Attributes:
        feature_extractor (nn.Module): A feature extractor based on VGG19 that outputs features from the specified layer.
        criterion (nn.Module): MSE loss used to compare the extracted feature maps.
    """

    def __init__(self, layer: str = 'features.35'):
        """
        Initialize the PerceptualLoss module.

        Args:
            layer (str, optional): The name of the layer from VGG19 from which to extract features.
                                   Default is 'features.35', which is typically used for high-level features.
        """
        super().__init__()

        # Load the pre-trained VGG19 model's feature layers and set to evaluation mode.
        vgg = vgg19(pretrained=True).features.eval()

        # Create a feature extractor that returns outputs from the specified VGG19 layer.
        self.feature_extractor = create_feature_extractor(vgg, {layer: 'feat'})

        # Freeze the feature extractor parameters to ensure they are not updated during training.
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Define the criterion for comparing the feature maps (Mean Squared Error).
        self.criterion = nn.MSELoss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Compute the perceptual loss between the super-resolved (sr) and high-resolution (hr) images.

        Args:
            sr (torch.Tensor): The super-resolved image tensor.
            hr (torch.Tensor): The high-resolution target image tensor.

        Returns:
            torch.Tensor: The computed perceptual loss value.
        """
        # Extract features from the super-resolved image.
        sr_features = self.feature_extractor(sr)['feat']
        # Extract features from the high-resolution ground truth image.
        hr_features = self.feature_extractor(hr)['feat']

        # Compute and return the MSE loss between the feature maps.
        loss = self.criterion(sr_features, hr_features)
        return loss
