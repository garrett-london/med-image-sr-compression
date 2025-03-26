"""
Module: train_superres
Description:
    This script trains a super-resolution model using a combination of pixel-wise
    and perceptual losses. It loads configuration settings from a YAML file,
    prepares the dataset and dataloader, initializes the Generator model along with
    its losses and optimizer, and then executes a training loop that logs losses and
    periodically saves generated images.
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from src.models.srgan import Generator
from src.models.losses import PerceptualLoss
from src.data.dataset import SuperResDataset
from src.data.transforms import get_transforms
from tqdm import tqdm


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """
    Load configuration settings from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration settings.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_device(train_cfg: dict) -> str:
    """
    Set up the device for training (GPU if available or CPU).

    Args:
        train_cfg (dict): Training configuration containing device information.

    Returns:
        str: The device identifier ('cuda' or 'cpu').
    """
    if torch.cuda.is_available() or train_cfg["device"] == "cpu":
        device = train_cfg["device"]
    else:
        device = "cpu"
    return device


def prepare_data(data_cfg: dict, train_cfg: dict):
    """
    Prepare the dataset and dataloader for training.

    Args:
        data_cfg (dict): Data configuration with paths to low-resolution and high-resolution images.
        train_cfg (dict): Training configuration containing parameters such as batch size.

    Returns:
        tuple: A tuple containing the training dataset and dataloader.
    """
    train_dataset = SuperResDataset(
        lr_dir=data_cfg["low_res_dir"],
        hr_dir=data_cfg["high_res_dir"],
        transform=get_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True
    )
    return train_dataset, train_loader


def setup_model_and_loss(train_cfg: dict, device: str):
    """
    Initialize the Generator model, loss functions, and optimizer.

    Args:
        train_cfg (dict): Training configuration with learning parameters.
        device (str): The device identifier ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the model, pixel-wise loss criterion, perceptual loss module, and optimizer.
    """
    model = Generator().to(device)

    # Define pixel-wise loss (MSE) and perceptual loss
    criterion = nn.MSELoss()
    perceptual_loss = PerceptualLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    return model, criterion, perceptual_loss, optimizer


def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
          perceptual_loss: nn.Module, optimizer: torch.optim.Optimizer,
          train_cfg: dict, device: str, save_dir: str):
    """
    Execute the training loop for the super-resolution model.

    Args:
        model (nn.Module): The generator model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Pixel-wise loss (e.g., MSELoss).
        perceptual_loss (nn.Module): Perceptual loss module.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        train_cfg (dict): Training configuration parameters.
        device (str): The device to run the training on (e.g., "cuda" or "cpu").
        save_dir (str): Directory to save output images.
    """
    num_epochs = train_cfg["num_epochs"]

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            lr_imgs, hr_imgs = batch
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            sr_imgs = model(lr_imgs)

            pixel_loss = criterion(sr_imgs, hr_imgs)
            percep_loss = perceptual_loss(sr_imgs, hr_imgs)

            loss = pixel_loss + 0.01 * percep_loss

            print(f"Epoch {epoch + 1}: pixel_loss={pixel_loss.item():.4f}, perceptual_loss={percep_loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        save_image(sr_imgs, f"{save_dir}/sr_epoch{epoch + 1}.png")
        save_image(hr_imgs, f"{save_dir}/gt_epoch{epoch + 1}.png")


def main():
    """Main function to run the training pipeline."""
    # ---- Load Configurations ---- #
    config = load_config("configs/default.yaml")
    train_cfg = config["train"]
    data_cfg = config["data"]

    # ---- Setup Device and Output Directory ---- #
    device = setup_device(train_cfg)
    SAVE_DIR = train_cfg["save_dir"]
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- Prepare Data ---- #
    _, train_loader = prepare_data(data_cfg, train_cfg)

    # ---- Initialize Model, Losses, and Optimizer ---- #
    model, criterion, perceptual_loss, optimizer = setup_model_and_loss(train_cfg, device)

    # ---- Training Loop ---- #
    train(model, train_loader, criterion, perceptual_loss, optimizer, train_cfg, device, SAVE_DIR)


if __name__ == "__main__":
    main()
