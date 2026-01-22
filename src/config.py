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

RARE_EMOTIONS = ['disgust', 'fear']
COMMON_EMOTIONS = ['happy', 'neutral']

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
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
NUM_WORKERS = 2

# Optimizer
OPTIMIZER = "Adam"
WEIGHT_DECAY = 0.0

# Loss Function
LABEL_SMOOTHING = 0.0

# Learning Rate Scheduling
LR_SCHEDULER = "ReduceLROnPlateau"
LR_FACTOR = 0.1
LR_PATIENCE = 1
LR_MIN = 1e-7

# Early Stopping
EARLY_STOPPING = False
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.001

# Focal Loss
USE_FOCAL_LOSS = False
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Model
DROPOUT_RATE = 0.5

# Paths
MODEL_SAVE_DIR = "models"
MODEL_NAME = "cnn_model.pth"

# Device
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
