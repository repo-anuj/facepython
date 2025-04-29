try:
    import tensorflow as tf
    print(f"TensorFlow is available, version: {tf.__version__}")
    print(f"TensorFlow is using GPU: {tf.config.list_physical_devices('GPU')}")
except ImportError as e:
    print(f"TensorFlow is not available: {e}")

try:
    import mediapipe as mp
    print(f"MediaPipe is available")
except ImportError as e:
    print(f"MediaPipe is not available: {e}")
