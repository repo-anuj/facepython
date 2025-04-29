import os
import sys
import time
from emotion_training import EmotionTrainer

# Print system info
print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Check for TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow is available, version: {tf.__version__}")
    print(f"TensorFlow is using GPU: {tf.config.list_physical_devices('GPU')}")
except ImportError as e:
    print(f"TensorFlow is not available: {e}")
    sys.exit(1)

# Check for emotion dataset
if not os.path.exists('emotion_dataset'):
    print("emotion_dataset directory does not exist")
    sys.exit(1)

# Create emotion trainer
print("Creating emotion trainer...")
emotion_trainer = EmotionTrainer(data_dir='emotion_dataset', model_dir='models')

# Train the model
print("Starting emotion model training...")
success, message = emotion_trainer.train_model(epochs=5, batch_size=32)
print(f"Training started: {success}, {message}")

# Wait for training to complete
if success:
    while emotion_trainer.is_training:
        status = emotion_trainer.get_training_status()
        print(f"Training progress: {status['progress']}%, Status: {status['status']}")
        time.sleep(5)
    
    print("Training complete!")
    print("Model info:", emotion_trainer.get_model_info())
else:
    print(f"Failed to start training: {message}")
