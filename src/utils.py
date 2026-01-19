import torch
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from collections import Counter
import numpy as np
from PIL import Image

def get_train_transforms():
    """Data augmentation for emotion recognition - designed for facial expressions"""
    return transforms.Compose([
        transforms.Resize((48, 48)),  # Safety resize (images should already be 48x48)
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale (1 channel) - ImageFolder may convert to RGB
        transforms.RandomRotation(degrees=10),   # Small rotations (±10°) - real faces can have slight tilt
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small shifts - accounts for imperfect face detection/alignment
        # Note: No horizontal flip - facial expressions are asymmetric (e.g., raised eyebrow on one side conveys different emotion info)
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
    
    # Handle class imbalance: FER-2013 has severe imbalance (e.g., disgust ~436 vs happy ~7215)
    # Weighted sampling ensures model sees rare emotions (disgust, fear) more often during training
    # This prevents model from only learning common emotions (happy, neutral)
    train_labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    
    # Calculate weights: inverse frequency - rare emotions get higher weights
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

def analyze_confusing_pairs(predictions, labels, class_names, confusing_pairs):
    """
    Analyze confusion between emotion pairs that are often confused.
    Useful for understanding model weaknesses and emotion recognition challenges.
    
    Args:
        predictions: Predicted class indices (tensor or numpy array)
        labels: True class indices (tensor or numpy array)
        class_names: List of emotion class names (e.g., ['angry', 'disgust', ...])
        confusing_pairs: List of tuples (emotion1, emotion2) that are often confused
    
    Returns:
        Dictionary with confusion statistics for each confusing pair
        Example: {'fear_vs_surprise': {'fear->surprise': 5, 'surprise->fear': 3, ...}, ...}
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    confusion_stats = {}
    for emotion1, emotion2 in confusing_pairs:
        idx1 = class_to_idx[emotion1]
        idx2 = class_to_idx[emotion2]
        
        # Find instances where predictions/labels involve this pair
        mask1 = (labels == idx1)  # True labels are emotion1
        mask2 = (labels == idx2)  # True labels are emotion2
        
        # Count confusions
        confusions_1_to_2 = ((predictions == idx2) & mask1).sum().item()  # emotion1 misclassified as emotion2
        confusions_2_to_1 = ((predictions == idx1) & mask2).sum().item()  # emotion2 misclassified as emotion1
        
        total_1 = mask1.sum().item()
        total_2 = mask2.sum().item()
        
        confusion_stats[f"{emotion1}_vs_{emotion2}"] = {
            f'{emotion1}->{emotion2}': confusions_1_to_2,
            f'{emotion2}->{emotion1}': confusions_2_to_1,
            f'{emotion1}_total': total_1,
            f'{emotion2}_total': total_2,
            f'{emotion1}_confusion_rate': confusions_1_to_2 / total_1 if total_1 > 0 else 0,
            f'{emotion2}_confusion_rate': confusions_2_to_1 / total_2 if total_2 > 0 else 0,
        }
    
    return confusion_stats
