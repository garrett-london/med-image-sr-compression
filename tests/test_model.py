"""
Test file for model functionality.

This test suite validates:
    - The Generator model's forward pass produces the correct output shape.
    - The ResidualBlock maintains input dimensions after processing.
    - The PerceptualLoss module computes a scalar, non-negative loss value.

Usage:
    Run the tests from the command line:
        python test_model_functionality.py
"""

import unittest
import torch
from src.models.srgan import Generator, ResidualBlock
from src.models.losses import PerceptualLoss


class TestGenerator(unittest.TestCase):
    """Unit tests for the Generator model and its components."""

    def setUp(self):
        """
        Set up common parameters and dummy input for testing.
        The Generator is expected to upscale the image by a factor of 4 (due to two PixelShuffle layers with upscale_factor=2).
        """
        self.batch_size = 1
        self.in_channels = 3
        self.out_channels = 3
        self.input_height = 24
        self.input_width = 24
        self.dummy_input = torch.randn(self.batch_size, self.in_channels, self.input_height, self.input_width)
        self.generator = Generator(in_channels=self.in_channels, out_channels=self.out_channels)

    def test_generator_output_shape(self):
        """
        Test that the Generator model produces output of the correct shape.

        Expected output shape:
            (batch_size, out_channels, input_height*4, input_width*4)
        """
        output = self.generator(self.dummy_input)
        expected_height = self.input_height * 4
        expected_width = self.input_width * 4
        expected_shape = (self.batch_size, self.out_channels, expected_height, expected_width)
        self.assertEqual(output.shape, expected_shape,
                         f"Generator output shape {output.shape} does not match expected shape {expected_shape}")

    def test_residual_block_identity(self):
        """
        Test that the ResidualBlock maintains the input shape after processing.
        """
        channels = 64
        block = ResidualBlock(channels)
        dummy_input = torch.randn(self.batch_size, channels, self.input_height, self.input_width)
        output = block(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape,
                         f"ResidualBlock output shape {output.shape} does not match input shape {dummy_input.shape}")


class TestPerceptualLoss(unittest.TestCase):
    """Unit tests for the PerceptualLoss module."""

    def setUp(self):
        """
        Initialize the PerceptualLoss module and create dummy image tensors.
        The dummy images are sized to 224x224, which is common for VGG-based feature extractors.

        Note:
            We pass '35' as the layer name instead of the default 'features.35'
            since the VGG19 features are a Sequential container with keys '0' to '35'.
        """
        self.loss_module = PerceptualLoss(layer="35")
        self.batch_size = 1
        self.channels = 3
        self.height = 224
        self.width = 224
        self.sr = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.hr = torch.randn(self.batch_size, self.channels, self.height, self.width)

    def test_loss_output_scalar(self):
        """
        Test that PerceptualLoss returns a scalar loss value.
        The loss should be a non-negative float.
        """
        loss = self.loss_module(self.sr, self.hr)
        self.assertTrue(isinstance(loss.item(), float),
                        "PerceptualLoss did not return a scalar float value.")
        self.assertGreaterEqual(loss.item(), 0.0,
                                "PerceptualLoss returned a negative loss value, which is unexpected.")


if __name__ == '__main__':
    unittest.main()
