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
NUM_EPOCHS = 100  # Increased from 50 - model was still improving at epoch 50
LEARNING_RATE = 0.001  # Increased to 0.001 - model needs higher LR to learn (was stuck at 0.0005)
NUM_WORKERS = 2
WARMUP_EPOCHS = 5  # Learning rate warmup for better training stability

# Optimizer
OPTIMIZER = "Adam"  # Options: "Adam" or "SGD"
WEIGHT_DECAY = 0.0001  # L2 regularization

# Loss Function
LABEL_SMOOTHING = 0.0  # Temporarily disabled - was interfering with learning when model collapses to one class
# Can re-enable later (0.05-0.1) once model starts learning properly

# Learning Rate Scheduling
LR_SCHEDULER = "CosineAnnealingLR"  # Changed to CosineAnnealingLR - better for longer training
LR_FACTOR = 0.5  # Factor to reduce LR by (for ReduceLROnPlateau)
LR_PATIENCE = 5  # Epochs to wait before reducing LR (for ReduceLROnPlateau)
LR_MIN = 1e-6  # Minimum learning rate
LR_T_MAX = 50  # Period for cosine annealing (restart every 50 epochs)

# Early Stopping
EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 15  # Increased from 10 - allow more time for deeper model to converge
EARLY_STOP_MIN_DELTA = 0.001  # Minimum change to qualify as improvement

# Focal Loss (helps with hard examples like fear, angry)
USE_FOCAL_LOSS = False  # Temporarily disabled - was causing model collapse. Use weighted CrossEntropy instead
FOCAL_ALPHA = 0.25  # Balancing factor for rare classes
FOCAL_GAMMA = 2.0  # Focusing parameter (higher = more focus on hard examples)

# Model
DROPOUT_RATE = 0.5

# Paths
MODEL_SAVE_DIR = "models"
MODEL_NAME = "cnn_model.pth"

# Device
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
