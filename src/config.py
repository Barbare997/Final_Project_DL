# Hyperparameters for emotion recognition model

# Data
DATA_DIR = "data"
NUM_CLASSES = 7  # FER-2013 has 7 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise
IMG_SIZE = 48  # FER-2013 images are pre-cropped to 48x48. This is optimal for the dataset - larger sizes don't improve accuracy but increase computation. 48x48 maintains facial features while keeping model efficient.

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
BATCH_SIZE = 64  # Balance between training stability and memory usage. 64 provides good gradient estimates without excessive memory. Smaller batches (32) were slower, larger (128) caused memory issues on GPU.
NUM_EPOCHS = 40  # Sufficient for convergence. Model reached best validation accuracy (~59%) around epoch 28-34, so 40 epochs provides margin without overfitting. Early experiments showed model still improving up to epoch 30+.
LEARNING_RATE = 0.001  # Found through experimentation. 0.0005 was too conservative (model stuck at ~14% accuracy), 0.001 allows faster learning. 0.002+ caused instability. This LR works well with Adam optimizer.
NUM_WORKERS = 2  # Data loading parallelism. 2 workers provide good speedup without overwhelming system. More workers (4+) didn't improve significantly and sometimes caused issues on Windows.

# Optimizer
OPTIMIZER = "Adam"  # Adam chosen over SGD for adaptive learning rate. Adam converges faster and handles sparse gradients better, which helps with class imbalance. SGD with momentum (0.9) was tested but Adam performed better.
WEIGHT_DECAY = 0.0  # L2 regularization disabled. With dropout already providing regularization, weight decay wasn't necessary. Experiments with 0.0001 showed no improvement and slightly slower convergence.

# Loss Function
LABEL_SMOOTHING = 0.0  # Disabled to allow model to learn clearly. Initially tried 0.1 but it interfered with learning when model was struggling. Can be re-enabled (0.05-0.1) if overfitting becomes an issue, but current model doesn't show significant overfitting.

# Learning Rate Scheduling
LR_SCHEDULER = "ReduceLROnPlateau"  # Responds to validation loss plateaus. Better than CosineAnnealingLR for this task as it adapts to actual performance rather than fixed schedule. CosineAnnealingLR was tested but didn't perform as well.
LR_FACTOR = 0.1  # Aggressive reduction (10x) when stuck. This helps escape local minima. Less aggressive (0.5) was too slow. Model benefits from larger LR drops when validation loss plateaus.
LR_PATIENCE = 1  # Reduce LR after 1 epoch without improvement. Low patience allows quick adaptation. Higher patience (3-5) was tested but model benefited from faster LR reduction when stuck. This prevents wasting epochs at suboptimal LR.
LR_MIN = 1e-7  # Very low minimum to allow fine-tuning. Model reached best performance with LR around 0.0001, so 1e-7 provides room for further refinement if needed.

# Early Stopping
EARLY_STOPPING = False  # Disabled because model showed gradual improvement throughout training. Best model was found at epoch 34, so early stopping would have stopped too early.
EARLY_STOP_PATIENCE = 10  # If enabled, wait 10 epochs without improvement. This gives model time to recover from temporary plateaus. Lower values (5-7) were too aggressive given the gradual improvement pattern.
EARLY_STOP_MIN_DELTA = 0.001  # Minimum improvement threshold (0.1% accuracy). Prevents stopping on tiny fluctuations. This value balances sensitivity to real improvements vs noise in validation metrics.

# Focal Loss
USE_FOCAL_LOSS = False  # Disabled - was causing model collapse to single class (disgust). Focal loss focuses on hard examples but combined with class imbalance, it destabilized training. Standard CrossEntropyLoss works better for this dataset.
FOCAL_ALPHA = 0.25  # Standard value for focal loss (if enabled). Balances rare vs common classes. 0.25 means rare classes get 4x weight. This is a common default from focal loss paper.
FOCAL_GAMMA = 2.0  # Focusing parameter (if enabled). Higher gamma = more focus on hard examples. 2.0 is standard default. Higher values (3.0+) were too aggressive and caused instability.

# Model
DROPOUT_RATE = 0.5

# Paths
MODEL_SAVE_DIR = "models"
MODEL_NAME = "cnn_model.pth"

# Device
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
