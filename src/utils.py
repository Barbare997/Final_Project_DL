import torch
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from collections import Counter
import numpy as np
from PIL import Image

def get_train_transforms():
    """Data augmentation for training - helps prevent overfitting and handles class imbalance"""
    return transforms.Compose([
        transforms.Resize((48, 48)),  # Safety resize (images should already be 48x48)
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale (1 channel) - ImageFolder may convert to RGB
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
        transforms.RandomRotation(degrees=10),   # Small rotations
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small shifts
        transforms.ToTensor(),  # Converts PIL to tensor and scales to [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] range for 1 channel
    ])

def get_val_test_transforms():
    """No augmentation for validation/test - just normalize"""
    return transforms.Compose([
        transforms.Resize((48, 48)),  # Safety resize (images should already be 48x48)
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale (1 channel) - ImageFolder may convert to RGB
        transforms.ToTensor(),  # Converts PIL to tensor and scales to [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] range for 1 channel
    ])

def get_data_loaders(data_dir, batch_size=64, num_workers=2):
    """
    Creates data loaders for train, validation, and test sets.
    
    """
    # Create datasets with appropriate transforms
    train_dataset = ImageFolder(
        root=f"{data_dir}/train",
        transform=get_train_transforms()
    )
    
    val_dataset = ImageFolder(
        root=f"{data_dir}/val",
        transform=get_val_test_transforms()
    )
    
    test_dataset = ImageFolder(
        root=f"{data_dir}/test",
        transform=get_val_test_transforms()
    )
    
    # Handle class imbalance with weighted sampling
    # Count samples per class
    train_labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    
    # Calculate weights: inverse frequency
    weights = torch.DoubleTensor([total_samples / class_counts[label] for label in train_labels])
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes

def preprocess_image_for_inference(image):
    """
    Preprocess a single image for inference (for live webcam demo).
    Image should be a PIL Image or numpy array of shape (H, W) or (H, W, 3).
    """
    transform = get_val_test_transforms()  # No augmentation for inference
    if isinstance(image, np.ndarray):
        # Convert numpy to PIL
        if len(image.shape) == 2:  # Grayscale
            image = Image.fromarray(image, mode='L')
        elif len(image.shape) == 3:  # RGB
            image = Image.fromarray(image, mode='RGB')
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    tensor = transform(image).unsqueeze(0)
    return tensor
