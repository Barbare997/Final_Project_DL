"""
Training script for emotion recognition CNN.
This script handles the complete training pipeline including training loops,
validation, model saving, and visualization of results.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
from tqdm import tqdm

# Import our custom modules
from model import EmotionCNN
from config import *
from utils import get_data_loaders, analyze_confusing_pairs


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one complete pass through the training data.
    
    This function:
    1. Sets model to training mode (enables dropout, batch norm updates)
    2. Processes batches of images
    3. Computes loss and gradients
    4. Updates model weights
    5. Tracks accuracy and loss statistics
    """
    model.train()  # Enable dropout and batch norm training behavior
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Process each batch of images
    for images, labels in tqdm(train_loader, desc="Training"):
        # Move data to GPU if available
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero out gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: get predictions from model
        predictions = model(images)
        
        # Calculate loss (with label smoothing if configured)
        loss = criterion(predictions, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update model weights using optimizer
        optimizer.step()
        
        # Track statistics for this batch
        running_loss += loss.item()
        
        # Get predicted class (index with highest probability)
        _, predicted_classes = torch.max(predictions.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted_classes == labels).sum().item()
    
    # Calculate average loss and accuracy for this epoch
    average_loss = running_loss / len(train_loader)
    accuracy = 100 * correct_predictions / total_samples
    
    return average_loss, accuracy


def validate_model(model, val_loader, criterion, device):
    """
    Evaluate the model on validation data.
    
    This function:
    1. Sets model to evaluation mode (disables dropout, freezes batch norm)
    2. Processes validation batches without computing gradients (faster)
    3. Collects all predictions and labels for analysis
    4. Returns loss, accuracy, and prediction arrays
    """
    model.eval()  # Disable dropout and batch norm updates
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Store all predictions and labels for confusion matrix analysis
    all_predictions = []
    all_labels = []
    
    # Don't compute gradients during validation (saves memory and time)
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            predictions = model(images)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Track statistics
            running_loss += loss.item()
            _, predicted_classes = torch.max(predictions.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().item()
            
            # Store predictions and labels for later analysis
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average metrics
    average_loss = running_loss / len(val_loader)
    accuracy = 100 * correct_predictions / total_samples
    
    return average_loss, accuracy, all_predictions, all_labels


def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    Create visualization plots showing training progress over epochs.
    
    Creates two plots side by side:
    - Left: Loss curves (training vs validation)
    - Right: Accuracy curves (training vs validation)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy curves
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('experiments/training_curves', exist_ok=True)
    plt.savefig('experiments/training_curves/training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining curves saved to: experiments/training_curves/training_history.png")
    plt.show()


def plot_confusion_matrix_visualization(true_labels, predicted_labels, class_names):
    """
    Create and display a confusion matrix showing model performance per emotion class.
    
    The confusion matrix shows:
    - Rows: True emotion labels
    - Columns: Predicted emotion labels
    - Diagonal: Correct predictions (should be high)
    - Off-diagonal: Misclassifications (shows which emotions are confused)
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Display confusion matrix as heatmap
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Emotion Recognition', fontsize=16, fontweight='bold', pad=20)
    plt.colorbar(label='Number of Samples')
    
    # Set tick marks and labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations showing exact counts
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=10, fontweight='bold')
    
    plt.ylabel('True Emotion Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Emotion Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save the confusion matrix
    os.makedirs('experiments/confusion_matrices', exist_ok=True)
    plt.savefig('experiments/confusion_matrices/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to: experiments/confusion_matrices/confusion_matrix.png")
    plt.show()


def main():
    """
    Main training function that orchestrates the entire training process.
    
    Steps:
    1. Setup device (CPU or GPU)
    2. Load and prepare data
    3. Create model
    4. Setup loss function, optimizer, and learning rate scheduler
    5. Train for multiple epochs
    6. Save best model
    7. Visualize results
    """
    print("=" * 70)
    print("EMOTION RECOGNITION CNN - TRAINING")
    print("=" * 70)
    
    # Determine if we can use GPU
    device = torch.device(DEVICE)
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load data
    print(f"\nLoading data from: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        DATA_DIR, BATCH_SIZE, NUM_WORKERS
    )
    print(f"Found {len(class_names)} emotion classes: {class_names}")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Create model
    print(f"\nCreating CNN model with {NUM_CLASSES} output classes...")
    model = EmotionCNN(num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    print(f"\nLoss function: CrossEntropyLoss with label smoothing = {LABEL_SMOOTHING}")
    
    # Setup optimizer
    if OPTIMIZER.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        print(f"Optimizer: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY
        )
        print(f"Optimizer: SGD (lr={LEARNING_RATE}, momentum=0.9, weight_decay={WEIGHT_DECAY})")
    
    # Setup learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',  # Reduce LR when validation loss stops decreasing
        factor=LR_FACTOR,  # Multiply LR by this factor
        patience=LR_PATIENCE,  # Wait this many epochs before reducing
        min_lr=LR_MIN  # Don't reduce below this
    )
    print(f"Learning rate scheduler: ReduceLROnPlateau (factor={LR_FACTOR}, patience={LR_PATIENCE})")
    
    # Training history tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Track best validation accuracy for model saving
    best_val_accuracy = 0.0
    patience_counter = 0
    
    print(f"\n{'='*70}")
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print(f"{'='*70}\n")
    
    # Main training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 70)
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_predictions, val_labels = validate_model(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"\nResults:")
        print(f"  Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            improvement = val_acc - best_val_accuracy
            best_val_accuracy = val_acc
            patience_counter = 0
            
            # Save model
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
            torch.save(model.state_dict(), model_path)
            print(f"\n✓ New best model saved! (Improvement: +{improvement:.2f}%)")
            print(f"  Saved to: {model_path}")
        else:
            patience_counter += 1
            print(f"\n  No improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
        
        # Early stopping check
        if EARLY_STOPPING and patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n{'='*70}")
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
            print(f"{'='*70}")
            break
    
    # Training complete - visualize results
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation accuracy achieved: {best_val_accuracy:.2f}%")
    
    # Plot training history
    print("\nGenerating training curves...")
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix_visualization(val_labels, val_predictions, class_names)
    
    # Print classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(
        val_labels,
        val_predictions,
        target_names=class_names,
        digits=4
    ))
    
    # Analyze confusing pairs if enabled
    if TRACK_CONFUSING_PAIRS:
        print("\n" + "="*70)
        print("CONFUSING PAIRS ANALYSIS")
        print("="*70)
        confusion_stats = analyze_confusing_pairs(
            np.array(val_predictions),
            np.array(val_labels),
            class_names,
            CONFUSING_PAIRS
        )
        
        for pair_name, stats in confusion_stats.items():
            print(f"\n{pair_name.replace('_', ' ').title()}:")
            print(f"  Confusions: {stats[list(stats.keys())[0]]} vs {stats[list(stats.keys())[1]]}")
            print(f"  Confusion rates: {stats[list(stats.keys())[4]]:.2%} vs {stats[list(stats.keys())[5]]:.2%}")
    
    print(f"\n{'='*70}")
    print("All done! Model saved and results visualized.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
