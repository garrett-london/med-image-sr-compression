"""
Module: superres_dataset
Description:
    This module defines the SuperResDataset class, a primitive custom PyTorch Dataset for
    loading paired low-resolution (LR) and high-resolution (HR) images. The dataset
    assumes that both the LR and HR directories contain images with matching filenames.
    Optionally, a transformation can be applied to both images (e.g., for data augmentation).
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class SuperResDataset(Dataset):
    """
    A custom Dataset for loading paired low-resolution (LR) and high-resolution (HR) images.

    Attributes:
        lr_dir (str): Directory path containing low-resolution images.
        hr_dir (str): Directory path containing high-resolution images.
        transform (callable, optional): A function/transform to apply to both LR and HR images.
        filenames (list): A sorted list of image filenames present in the low-resolution directory.
                          It is assumed that the HR directory contains images with identical filenames.
    """

    def __init__(self, lr_dir: str, hr_dir: str, transform=None):
        """
        Initialize the SuperResDataset.

        Args:
            lr_dir (str): Path to the directory with low-resolution images.
            hr_dir (str): Path to the directory with high-resolution images.
            transform (callable, optional): Optional transform to be applied on both LR and HR images.
                                            Defaults to None.
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform

        # Retrieve and sort the list of filenames from the low-resolution directory.
        # This sorting ensures a consistent order and alignment with high-resolution images.
        self.filenames = sorted(os.listdir(lr_dir))

    def __len__(self) -> int:
        """
        Return the number of image pairs available in the dataset.

        Returns:
            int: Total count of image pairs.
        """
        return len(self.filenames)

    def __getitem__(self, idx: int):
        """
        Retrieve the paired low-resolution and high-resolution images at the specified index.

        Args:
            idx (int): Index of the image pair to retrieve.

        Returns:
            tuple: A tuple (lr_img, hr_img) where:
                - lr_img: The transformed low-resolution image (or raw PIL Image if no transform is applied).
                - hr_img: The transformed high-resolution image (or raw PIL Image if no transform is applied).
        """
        # Obtain the filename corresponding to the given index.
        filename = self.filenames[idx]

        # Build the full file paths for both LR and HR images.
        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)

        # Open each image and ensure it is converted to RGB format.
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        # If a transformation is provided, apply it to both images.
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        return lr_img, hr_img
