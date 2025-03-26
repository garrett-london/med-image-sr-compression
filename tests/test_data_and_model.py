import torch
from src.models.srgan import Generator
from src.data.dataset import SuperResDataset
from src.data.transforms import get_transforms
from src.train import load_config


def test_dataset_and_generator():
    lr_dir = "data/processed/low_res"
    hr_dir = "data/processed/high_res"

    dataset = SuperResDataset(lr_dir=lr_dir, hr_dir=hr_dir, transform=get_transforms())

    assert len(dataset) > 0, "Dataset is empty — check your paths."

    lr_img, hr_img = dataset[0]
    assert lr_img.shape == hr_img.shape, "LR and HR shapes mismatch — check your resizing logic."

    print(f"Sample shape: {lr_img.shape}")

    lr_img = lr_img.unsqueeze(0)

    model = Generator()

    with torch.no_grad():
        output = model(lr_img)

    print(f"Output shape: {output.shape}")
    assert output.shape == lr_img.shape, "Output shape mismatch — check your upsampling."


if __name__ == "__main__":
    test_dataset_and_generator()
    print("✅ Dataset and Generator are working as expected.")
