"""
Dataset module
"""

from pathlib import Path
from typing import Tuple, List

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LogoDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx][0]
        label = self.samples[idx][1]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, float(label)


def get_transforms(is_training: bool) -> transforms.Compose:
    """
    Builds torchvision transforms for training and evaluation.

    Parameters:
    - is_training (bool): Whether to build transforms for training (with augmentations) or evaluation (without augmentations).

    Returns:
    - torchvision.transforms.Compose: The composed transforms to apply to the images.
    """

    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def parse_annotations_file(annotations_file_path: str) -> List[Tuple[str, int]]:
    """
    Parses the annotations file and returns a list of (image_path, label) tuples.

    Parameters:
    - annotations_file_path (str): The path to the annotations file.

    Returns:
    - List[Tuple[str, int]]: A list of tuples, where each tuple contains the image path and its corresponding binary label (1 for target logos, 0 for non-target logos).
    """

    samples = []
    with open(annotations_file_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            filename, label = parts[0], parts[1]
            if label in ["Cocacola", "McDonalds", "Starbucks", "Disney"]:
                label = 1
            else:
                label = 0
            samples.append((
                str(Path(config["data"]["raw_images_dir_path"]) / filename), label
            ))

    return samples


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """
    Creates the training and validation dataloaders

    Returns:
    - Tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader and validation DataLoader, respectively.
    """
    
    train_samples = parse_annotations_file(config["data"]["training_set_annotations_path"])
    val_samples = parse_annotations_file(config["data"]["validation_set_annotations_path"])

    train_loader = DataLoader(
        LogoDataset(samples=train_samples, transform=get_transforms(is_training=True)),
        batch_size=config["data"]["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        LogoDataset(samples=val_samples, transform=get_transforms(is_training=False)),
        batch_size=config["data"]["batch_size"],
        shuffle=False
    )

    return train_loader, val_loader