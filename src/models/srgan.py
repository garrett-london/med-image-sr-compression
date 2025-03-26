"""
Module: generator_network
Description:
    This module defines a Generator network for super-resolution tasks along with a ResidualBlock.
    The Generator leverages residual learning and PixelShuffle-based upsampling to reconstruct
    high-resolution images from low-resolution inputs.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block used in the Generator network.

    This block implements two convolutional layers with batch normalization and a PReLU activation.
    A residual connection is added to help mitigate the vanishing gradient problem during training.

    Attributes:
        block (nn.Sequential): A sequential container of layers that constitute the residual block.
    """

    def __init__(self, channels: int):
        """
        Initialize the ResidualBlock.

        Args:
            channels (int): The number of input and output channels for the convolutional layers.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor obtained by adding the block's output to the original input.
        """
        return x + self.block(x)


class Generator(nn.Module):
    """
    Generator network for image super-resolution.

    This network comprises an initial feature extraction block, multiple residual blocks for deep
    feature learning, an intermediate convolution block with a skip connection, and PixelShuffle-based
    upsampling modules to reconstruct a high-resolution image from a low-resolution input.

    Attributes:
        entry (nn.Sequential): Initial convolution block for feature extraction.
        res_blocks (nn.Sequential): Stack of residual blocks for deep feature learning.
        mid (nn.Sequential): Intermediate convolution block to refine features after residual learning.
        upsample (nn.Sequential): Upsampling modules using PixelShuffle for resolution enhancement.
        output (nn.Conv2d): Final convolution layer to produce the high-resolution image.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_res_blocks: int = 5):
        """
        Initialize the Generator network.

        Args:
            in_channels (int): Number of channels in the input image (default: 3 for RGB images).
            out_channels (int): Number of channels in the output image (default: 3 for RGB images).
            num_res_blocks (int): Number of residual blocks to be used (default: 5).
        """
        super().__init__()

        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        # Upsampling block: uses convolution and PixelShuffle layers for efficient upscaling.
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )

        # Output layer: final convolution to produce the high-resolution image.
        self.output = nn.Conv2d(64, out_channels, kernel_size=9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Generator network.

        Args:
            x (torch.Tensor): Input low-resolution image tensor.

        Returns:
            torch.Tensor: Output high-resolution image tensor.
        """
        # Extract initial features from the input.
        x = self.entry(x)

        # Pass through residual blocks for deep feature extraction.
        res = self.res_blocks(x)

        # Process output of residual blocks & add the original features (skip connection).
        x = self.mid(res) + x

        # Upsample features to reconstruct the high-resolution image.
        x = self.upsample(x)

        # Generate the final output image.
        return self.output(x)
