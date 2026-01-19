# Hyperparameters for emotion recognition model

# Data
DATA_DIR = "data"
NUM_CLASSES = 7
IMG_SIZE = 48

# FER-2013 Dataset Characteristics:
# - 35,887 total images (training: 28,709, testing: 7,178)
# - Class imbalance: happy (7215) >> disgust (436)
# - 48x48 grayscale pre-cropped faces
# - Label noise: some emotions are subjective (neutral vs sad)
# - Challenge: distinguishing similar emotions (fear vs surprise)

# Emotion class mapping (FER-2013 order)
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Emotion recognition specific settings
# Rare emotions need special attention due to class imbalance
# Handling strategy: WeightedRandomSampler in utils.py (inverse frequency weighting)
# - Rare emotions (disgust, fear) get higher sampling weights
# - Ensures model sees all emotions during training, not just common ones
RARE_EMOTIONS = ['disgust', 'fear']  # These have fewest samples in FER-2013 (~436 and ~4097 respectively)
COMMON_EMOTIONS = ['happy', 'neutral']  # Most frequent emotions (~7215 and ~4965 respectively)

# Emotion pairs that are often confused (FER-2013 specific)
# Strategies to handle:
# 1. Track confusion matrix for these pairs during evaluation (in train.py)
# 2. Use per-emotion accuracy metrics (not just overall accuracy)
# 3. Could add custom loss penalty for confusing pair errors 
CONFUSING_PAIRS = [
    ('fear', 'surprise'),  # Both have wide eyes - similar facial features
    ('angry', 'disgust'),   # Both involve frowning - subtle differences
    ('neutral', 'sad')      # Subtle differences in mouth/eyebrow position
]

# Evaluation: Track confusing pair accuracy (implement in training/evaluation)
TRACK_CONFUSING_PAIRS = True  # Set to True to analyze confusion between these pairs

# Training
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_WORKERS = 2

# Optimizer
OPTIMIZER = "Adam"  # Options: "Adam" or "SGD"
WEIGHT_DECAY = 0.0001  # L2 regularization

# Loss Function
LABEL_SMOOTHING = 0.1  # Prevents overconfidence, helps with class imbalance (rare emotions like disgust)
# Label smoothing helps model handle ambiguous cases and prevents overfitting to common emotions

# Learning Rate Scheduling
LR_SCHEDULER = "ReduceLROnPlateau"  # Options: "ReduceLROnPlateau", "StepLR", "CosineAnnealingLR"
LR_FACTOR = 0.5  # Factor to reduce LR by
LR_PATIENCE = 5  # Epochs to wait before reducing LR
LR_MIN = 1e-6  # Minimum learning rate

# Early Stopping
EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 10  # Epochs to wait before stopping
EARLY_STOP_MIN_DELTA = 0.001  # Minimum change to qualify as improvement

# Model
DROPOUT_RATE = 0.5

# Paths
MODEL_SAVE_DIR = "models"
MODEL_NAME = "cnn_model.pth"

# Device
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
