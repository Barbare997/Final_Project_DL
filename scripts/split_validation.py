import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Define paths
data_dir = Path("data")
train_dir = data_dir / "train"
val_dir = data_dir / "val"

# Create validation directory structure
val_dir.mkdir(parents=True, exist_ok=True)

# Get all emotion classes
emotion_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
print(f"Found {len(emotion_classes)} emotion classes: {emotion_classes}")

# Split 20% from train to val for each class
for emotion in emotion_classes:
    emotion_train_dir = train_dir / emotion
    emotion_val_dir = val_dir / emotion
    emotion_val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files for this emotion
    image_files = list(emotion_train_dir.glob("*.jpg"))
    total_images = len(image_files)
    
    # Calculate 20% split
    num_val = int(total_images * 0.2)
    
    # Randomly select images for validation
    random.shuffle(image_files)
    val_images = image_files[:num_val]
    
    # Move images to validation directory
    moved = 0
    for img_file in val_images:
        dest = emotion_val_dir / img_file.name
        shutil.move(str(img_file), str(dest))
        moved += 1
    
    print(f"{emotion:12s} - Total: {total_images:5d}, Moved to val: {moved:5d}, Remaining in train: {total_images - moved:5d}")

print("\n[SUCCESS] Validation split completed successfully!")
print(f"Training images: {sum(len(list((train_dir / cls).glob('*.jpg'))) for cls in emotion_classes)}")
print(f"Validation images: {sum(len(list((val_dir / cls).glob('*.jpg'))) for cls in emotion_classes)}")
print(f"Test images: {sum(len(list((data_dir / 'test' / cls).glob('*.jpg'))) for cls in emotion_classes)}")
