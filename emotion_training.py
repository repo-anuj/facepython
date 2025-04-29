import os
import cv2
import numpy as np
import pickle
from datetime import datetime
import threading

# Try to import TensorFlow, but continue if not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow successfully imported for emotion training")
except ImportError as e:
    print(f"Warning: {e}")
    print("Emotion model training will not be available without TensorFlow")
    TENSORFLOW_AVAILABLE = False

# Only import these if TensorFlow is available
if TENSORFLOW_AVAILABLE:
    try:
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.models import Model, load_model
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.utils import to_categorical
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        print("All required TensorFlow modules imported for emotion training")
    except ImportError as e:
        print(f"Warning: {e}")
        print("Some TensorFlow modules could not be imported")
        TENSORFLOW_AVAILABLE = False

class EmotionTrainer:
    def __init__(self, data_dir='emotion_dataset', model_dir='models'):
        """Initialize the emotion trainer with directories for data and models."""
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = None
        self.label_encoder = None
        self.is_training = False
        self.training_thread = None
        self.training_progress = 0
        self.training_status = "Not started"
        self.training_log = []
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _log(self, message):
        """Add a message to the training log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.training_log.append(log_entry)

    def load_data(self):
        """Load emotion images from the data directory."""
        if not TENSORFLOW_AVAILABLE:
            self._log("TensorFlow not available, cannot load data for training")
            return None, None

        if not os.path.exists(self.data_dir):
            self._log(f"Data directory {self.data_dir} does not exist")
            return None, None

        self._log("Loading emotion data from directory...")
        images = []
        labels = []

        # Check if emotion folders exist - be more flexible with folder names
        emotion_folders = []
        self._log(f"Looking for emotion folders in {self.data_dir}")

        try:
            for d in os.listdir(self.data_dir):
                folder_path = os.path.join(self.data_dir, d)
                if os.path.isdir(folder_path):
                    # Accept any folder that exists
                    emotion_folders.append(d)
                    self._log(f"Found folder: {d}")

                    # Count images in the folder
                    try:
                        image_files = [f for f in os.listdir(folder_path)
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        self._log(f"  - Contains {len(image_files)} image files")
                    except Exception as e:
                        self._log(f"  - Error counting images: {e}")
        except Exception as e:
            self._log(f"Error listing directory {self.data_dir}: {e}")

        if not emotion_folders:
            self._log("No emotion folders found in data directory")
            return None, None

        self._log(f"Found {len(emotion_folders)} emotion folders: {', '.join(emotion_folders)}")

        # Load images for each emotion
        for emotion in emotion_folders:
            emotion_dir = os.path.join(self.data_dir, emotion)
            self._log(f"Loading images for {emotion}...")

            # Get all image files
            image_files = [f for f in os.listdir(emotion_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                self._log(f"No images found for {emotion}, skipping")
                continue

            self._log(f"Found {len(image_files)} images for {emotion}")

            # Load a subset of images if there are too many (for faster training)
            max_images_per_emotion = 1000
            if len(image_files) > max_images_per_emotion:
                self._log(f"Using only {max_images_per_emotion} images for {emotion}")
                image_files = image_files[:max_images_per_emotion]

            # Load each image
            loaded_count = 0
            error_count = 0
            for img_file in image_files:
                img_path = os.path.join(emotion_dir, img_file)
                try:
                    # Check if file exists
                    if not os.path.exists(img_path):
                        self._log(f"Image file does not exist: {img_path}, skipping")
                        error_count += 1
                        continue

                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is None:
                        self._log(f"Failed to load image {img_path}, skipping")
                        error_count += 1
                        continue

                    # Convert BGR to grayscale (emotion recognition typically uses grayscale)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Resize to 48x48 (standard size for emotion recognition)
                    img = cv2.resize(img, (48, 48))

                    # Add to dataset
                    images.append(img)
                    labels.append(emotion)
                    loaded_count += 1

                    # Print progress every 100 images
                    if loaded_count % 100 == 0:
                        self._log(f"Loaded {loaded_count} images for {emotion} so far...")

                except Exception as e:
                    self._log(f"Error processing {img_path}: {str(e)}")
                    error_count += 1

            self._log(f"Successfully loaded {loaded_count} images for {emotion}, with {error_count} errors")

        if not images:
            self._log("No valid images found")
            return None, None

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        # Reshape images to include channel dimension (48x48x1)
        X = X.reshape(X.shape[0], 48, 48, 1)

        self._log(f"Loaded {len(X)} images with {len(set(y))} unique emotions")

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Convert to one-hot encoding
        y_categorical = to_categorical(y_encoded)

        return X, y_categorical

    def build_model(self, num_classes):
        """Build a CNN model for emotion recognition."""
        if not TENSORFLOW_AVAILABLE:
            self._log("TensorFlow not available, cannot build model")
            return None

        self._log("Building CNN model for emotion recognition...")

        model = tf.keras.Sequential([
            # First convolutional block
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Second convolutional block
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Third convolutional block
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Flatten and dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self._log(f"Model built with {num_classes} output classes")
        return model

    def train_model_thread(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Training function to run in a separate thread."""
        try:
            self.is_training = True
            self.training_progress = 0
            self.training_status = "Preparing data"

            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )

            # Normalize pixel values
            X_train = X_train.astype('float32') / 255.0
            X_val = X_val.astype('float32') / 255.0

            # Build the model
            num_classes = y.shape[1]
            self.model = self.build_model(num_classes)

            if self.model is None:
                self.training_status = "Failed to build model"
                self.is_training = False
                return

            # Data augmentation
            self.training_status = "Training model"
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            # Train the model with a custom callback to update progress
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, trainer):
                    self.trainer = trainer

                def on_epoch_end(self, epoch, logs=None):
                    # Update progress (0-100%)
                    self.trainer.training_progress = int(((epoch + 1) / epochs) * 100)
                    self.trainer._log(f"Epoch {epoch+1}/{epochs} - "
                                     f"loss: {logs['loss']:.4f} - "
                                     f"accuracy: {logs['accuracy']:.4f} - "
                                     f"val_loss: {logs['val_loss']:.4f} - "
                                     f"val_accuracy: {logs['val_accuracy']:.4f}")

            # Train the model
            self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=[ProgressCallback(self)]
            )

            # Save the model and label encoder
            self.save_model()

            self.training_status = "Training complete"
            self._log("Emotion model training completed successfully")

        except Exception as e:
            self.training_status = f"Error: {str(e)}"
            self._log(f"Error during training: {str(e)}")

        finally:
            self.is_training = False

    def train_model(self, epochs=50, batch_size=32, validation_split=0.2):
        """Start training the model in a separate thread."""
        if not TENSORFLOW_AVAILABLE:
            return False, "TensorFlow not available, cannot train model"

        if self.is_training:
            return False, "Training already in progress"

        # Load data
        X, y = self.load_data()
        if X is None or y is None:
            return False, "Failed to load training data"

        # Check if we have enough data - be more lenient with the minimum
        if len(X) < 10:
            return False, f"Not enough training data. Found only {len(X)} images, need at least 10."

        self._log(f"Starting training with {len(X)} images across {y.shape[1]} emotion classes")

        # Start training in a separate thread
        self.training_thread = threading.Thread(
            target=self.train_model_thread,
            args=(X, y, epochs, batch_size, validation_split)
        )
        self.training_thread.daemon = True
        self.training_thread.start()

        return True, "Emotion model training started in background"

    def save_model(self):
        """Save the trained model and label encoder."""
        if self.model is None or self.label_encoder is None:
            self._log("No model or label encoder to save")
            return False

        # Create timestamp for the model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the Keras model
        model_path = os.path.join(self.model_dir, 'emotion_model.h5')
        self.model.save(model_path)
        self._log(f"Emotion model saved to {model_path}")

        # Save the label encoder
        encoder_path = os.path.join(self.model_dir, 'emotion_label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        self._log(f"Emotion label encoder saved to {encoder_path}")

        return True

    def load_model(self):
        """Load the trained emotion model and label encoder."""
        if not TENSORFLOW_AVAILABLE:
            self._log("TensorFlow not available, cannot load model")
            return False

        model_path = os.path.join(self.model_dir, 'emotion_model.h5')
        encoder_path = os.path.join(self.model_dir, 'emotion_label_encoder.pkl')

        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            self._log("Emotion model or label encoder not found")
            return False

        try:
            # Load the model
            self.model = load_model(model_path)

            # Load the label encoder
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)

            self._log(f"Loaded emotion model from {model_path}")
            self._log(f"Loaded emotion label encoder with {len(self.label_encoder.classes_)} classes")
            return True

        except Exception as e:
            self._log(f"Error loading emotion model: {str(e)}")
            return False

    def predict_emotion(self, face_img):
        """
        Predict the emotion of a face.

        Args:
            face_img: Face image (BGR format from OpenCV)

        Returns:
            (emotion, confidence): Predicted emotion and confidence score
        """
        if not TENSORFLOW_AVAILABLE or self.model is None or self.label_encoder is None:
            return "Unknown", 0.0

        try:
            # Preprocess the image
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))

            # Normalize pixel values
            gray = gray.astype('float32') / 255.0

            # Reshape for model input (add batch and channel dimensions)
            gray = gray.reshape(1, 48, 48, 1)

            # Make prediction
            predictions = self.model.predict(gray)[0]

            # Get the emotion with highest probability
            emotion_idx = np.argmax(predictions)
            confidence = float(predictions[emotion_idx])

            # Convert class index to emotion name
            emotion = self.label_encoder.inverse_transform([emotion_idx])[0]

            return emotion, confidence

        except Exception as e:
            print(f"Error predicting emotion: {str(e)}")
            return "Error", 0.0

    def get_training_status(self):
        """Get the current status of model training."""
        return {
            'is_training': self.is_training,
            'progress': self.training_progress,
            'status': self.training_status,
            'log': self.training_log[-10:] if self.training_log else []  # Last 10 log entries
        }

    def get_model_info(self):
        """Get information about the loaded model."""
        if self.model is None:
            return {
                'loaded': False,
                'classes': [],
                'summary': "No emotion model loaded"
            }

        classes = list(self.label_encoder.classes_) if self.label_encoder else []

        return {
            'loaded': True,
            'classes': classes,
            'summary': f"Emotion model with {len(classes)} classes: {', '.join(classes)}"
        }
