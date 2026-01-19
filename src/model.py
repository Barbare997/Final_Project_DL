import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    """
    CNN for facial emotion recognition on FER-2013 dataset.
    
    Architecture designed for 48x48 grayscale face images:
    - First conv block: Captures low-level features (edges, textures around eyes/mouth)
    - Second conv block: Detects facial parts (eyes, mouth, eyebrows)
    - Third conv block: Learns emotion-specific patterns (smile, frown, raised eyebrows)
    - FC layers: Maps facial patterns to 7 emotion categories
    
    Key design choices for emotion recognition:
    - 3 conv blocks: Balances feature extraction vs computational cost for 48x48 inputs
      (more blocks would lose spatial info for such small images)
    - Dropout2d after conv: Prevents overfitting on common emotions (happy, neutral)
    - 128 filters in final conv: Sufficient to capture complex emotion patterns
      while keeping model size reasonable (~1.27M parameters)
    """
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        # First conv block: Detects basic facial features (edges, textures)
        # 48x48 -> 24x24 after pooling
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(32)

        # Second conv block: Identifies facial regions (eyes, mouth, eyebrows)
        # 24x24 -> 12x12 after pooling
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.bn2 = nn.BatchNorm2d(64)

        # Third conv block: Learns emotion-specific patterns (smile curvature, eyebrow position)
        # 12x12 -> 6x6 after pooling - maintains spatial resolution for emotion features
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)  # Prevents overfitting on common emotions (happy, neutral)
        self.dropout_fc = nn.Dropout(0.5)  # Dropout for FC layers

        # Final classification layers: Maps 6x6 feature maps to 7 emotion categories
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        self.fc2 = nn.Linear(256, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # First conv block: 48x48 -> 24x24
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)

        # Second conv block: 24x24 -> 12x12
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)

        # Third conv block: 12x12 -> 6x6
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Classification
        x = self.dropout_fc(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
