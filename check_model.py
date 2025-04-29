import os
import sys

# Print system info
print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Check for TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow is available, version: {tf.__version__}")
    tf.config.set_visible_devices([], 'GPU')  # Disable GPU
    print("GPU disabled for TensorFlow")
except ImportError as e:
    print(f"TensorFlow is not available: {e}")
    sys.exit(1)

# Check for models directory
if os.path.exists('models'):
    print("models directory exists")
    files = os.listdir('models')
    if files:
        print(f"Found files in models directory: {', '.join(files)}")
        
        # Try to load the emotion model if it exists
        if 'emotion_model.h5' in files:
            try:
                print("Loading emotion model...")
                model = tf.keras.models.load_model(os.path.join('models', 'emotion_model.h5'))
                print("Model loaded successfully!")
                print(f"Model summary: {model.summary()}")
            except Exception as e:
                print(f"Error loading model: {e}")
    else:
        print("No files found in models directory")
else:
    print("models directory does not exist")
