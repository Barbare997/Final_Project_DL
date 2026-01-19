import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        # Three conv layers with increasing filters
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Final classification layers
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # First conv block: 48x48 -> 24x24
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Second conv block: 24x24 -> 12x12
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Third conv block: 12x12 -> 6x6
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Classification
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
