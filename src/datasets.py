from random import shuffle
from unittest import TestLoader
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path

IMG_SIZE = 224  
BATCH_SIZE = 32 
# transform for image
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),

])
def create_dataloader(train_dir, test_dir,dataset:bool=False):
    train_data = datasets.ImageFolder(root=train_dir,transform=train_transform)
    test_data = datasets.ImageFolder(root=test_dir,transform=train_transform)

    if dataset:
        return train_data, test_data
    else:
        # create data Loader
        train_loader = DataLoader(dataset = train_data, shuffle=True, batch_size = BATCH_SIZE)
        test_loader = DataLoader(dataset = test_data, batch_size = BATCH_SIZE)

        return train_loader, test_loader

