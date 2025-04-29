import os
import cv2
import numpy as np
import pickle
from datetime import datetime
import threading

# Try to import TensorFlow and Keras, but continue if not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow successfully imported for model training")
except ImportError as e:
    print(f"Warning: {e}")
    print("Model training will not be available without TensorFlow")
    TENSORFLOW_AVAILABLE = False

# Only import these if TensorFlow is available
if TENSORFLOW_AVAILABLE:
    try:
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.utils import to_categorical
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        print("All required TensorFlow modules imported for model training")
    except ImportError as e:
        print(f"Warning: {e}")
        print("Some TensorFlow modules could not be imported")
        TENSORFLOW_AVAILABLE = False

class ModelTrainer:
    def __init__(self, data_dir='collected_data', model_dir='models'):
        """Initialize the model trainer with directories for data and models."""
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = None
        self.label_encoder = None
        self.is_training = False
        self.training_thread = None
        self.training_progress = 0
        self.training_status = "Not started"
        self.training_log = []

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
        """Load face images and labels from the data directory."""
        if not TENSORFLOW_AVAILABLE:
            self._log("TensorFlow not available, cannot load data for training")
            return None, None

        if not os.path.exists(self.data_dir):
            self._log(f"Data directory {self.data_dir} does not exist")
            return None, None

        self._log("Loading data from directory...")
        images = []
        labels = []

        # Get list of people (subdirectories)
        people = [d for d in os.listdir(self.data_dir)
                 if os.path.isdir(os.path.join(self.data_dir, d))]

        if not people:
            self._log("No people found in data directory")
            return None, None

        self._log(f"Found {len(people)} people: {', '.join(people)}")

        # Load images for each person
        for person in people:
            person_dir = os.path.join(self.data_dir, person)
            self._log(f"Loading images for {person}...")

            # Get all image files
            image_files = [f for f in os.listdir(person_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                self._log(f"No images found for {person}, skipping")
                continue

            self._log(f"Found {len(image_files)} images for {person}")

            # Load each image
            for img_file in image_files:
                img_path = os.path.join(person_dir, img_file)
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is None:
                        self._log(f"Failed to load image {img_path}, skipping")
                        continue

                    # Convert BGR to RGB (Keras uses RGB)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Resize to 224x224 (VGG16 input size)
                    img = cv2.resize(img, (224, 224))

                    # Add to dataset
                    images.append(img)
                    labels.append(person)
                except Exception as e:
                    self._log(f"Error processing {img_path}: {str(e)}")

        if not images:
            self._log("No valid images found")
            return None, None

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        self._log(f"Loaded {len(X)} images with {len(set(y))} unique labels")

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Convert to one-hot encoding for categorical_crossentropy loss
        y_encoded = to_categorical(y_encoded)

        return X, y_encoded

    def build_model(self, num_classes):
        """Build a VGG16-based model for face recognition."""
        if not TENSORFLOW_AVAILABLE:
            self._log("TensorFlow not available, cannot build model")
            return None

        self._log("Building VGG16-based model...")

        # Load VGG16 without top layers, pre-trained on ImageNet
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',  # Changed from sparse_categorical_crossentropy
            metrics=['accuracy']
        )

        self._log(f"Model built with {num_classes} output classes")
        return model

    def train_model_thread(self, X, y, epochs=20, batch_size=32, validation_split=0.2):
        """Training function to run in a separate thread."""
        try:
            self.is_training = True
            self.training_progress = 0
            self.training_status = "Preparing data"

            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, stratify=y, random_state=42
            )

            # Normalize pixel values
            X_train = X_train.astype('float32') / 255.0
            X_val = X_val.astype('float32') / 255.0

            # Build the model
            num_classes = len(np.unique(y))
            self.model = self.build_model(num_classes)

            if self.model is None:
                self.training_status = "Failed to build model"
                self.is_training = False
                return

            # Data augmentation
            self.training_status = "Training model"
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
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
            # Handle small datasets by ensuring steps_per_epoch is at least 1
            steps = max(1, len(X_train) // batch_size)

            # Adjust batch size if needed
            if len(X_train) < batch_size:
                actual_batch_size = len(X_train)
                self._log(f"Adjusting batch size to {actual_batch_size} due to small dataset")
            else:
                actual_batch_size = batch_size

            self.model.fit(
                datagen.flow(X_train, y_train, batch_size=actual_batch_size),
                steps_per_epoch=steps,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=[ProgressCallback(self)]
            )

            # Save the model and label encoder
            self.save_model()

            self.training_status = "Training complete"
            self._log("Model training completed successfully")

        except Exception as e:
            self.training_status = f"Error: {str(e)}"
            self._log(f"Error during training: {str(e)}")

        finally:
            self.is_training = False

    def train_model(self, epochs=20, batch_size=32, validation_split=0.2):
        """Start training the model in a separate thread."""
        if not TENSORFLOW_AVAILABLE:
            return False, "TensorFlow not available, cannot train model"

        if self.is_training:
            return False, "Training already in progress"

        # Load data
        X, y = self.load_data()
        if X is None or y is None:
            return False, "Failed to load training data"

        # Check if we have enough data
        if len(X) < 10:
            return False, f"Not enough training data. Found only {len(X)} images, need at least 10."

        # Start training in a separate thread
        self.training_thread = threading.Thread(
            target=self.train_model_thread,
            args=(X, y, epochs, batch_size, validation_split)
        )
        self.training_thread.daemon = True
        self.training_thread.start()

        return True, "Training started in background"

    def save_model(self):
        """Save the trained model and label encoder."""
        if self.model is None or self.label_encoder is None:
            self._log("No model or label encoder to save")
            return False

        # Create timestamp for the model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"face_recognition_model_{timestamp}")

        # Save the Keras model
        self.model.save(model_path)
        self._log(f"Model saved to {model_path}")

        # Save the label encoder
        encoder_path = os.path.join(self.model_dir, f"label_encoder_{timestamp}.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        self._log(f"Label encoder saved to {encoder_path}")

        return True

    def load_latest_model(self):
        """Load the latest trained model and label encoder."""
        if not TENSORFLOW_AVAILABLE:
            self._log("TensorFlow not available, cannot load model")
            return False

        if not os.path.exists(self.model_dir):
            self._log(f"Model directory {self.model_dir} does not exist")
            return False

        # Find the latest model
        model_dirs = [d for d in os.listdir(self.model_dir)
                     if os.path.isdir(os.path.join(self.model_dir, d))
                     and d.startswith("face_recognition_model_")]

        if not model_dirs:
            self._log("No trained models found")
            return False

        # Sort by timestamp (newest first)
        latest_model_dir = sorted(model_dirs)[-1]
        model_path = os.path.join(self.model_dir, latest_model_dir)

        # Find the corresponding label encoder
        timestamp = latest_model_dir.split("_")[-1]
        encoder_path = os.path.join(self.model_dir, f"label_encoder_{timestamp}.pkl")

        if not os.path.exists(encoder_path):
            self._log(f"Label encoder not found for model {latest_model_dir}")
            return False

        try:
            # Load the model
            self.model = tf.keras.models.load_model(model_path)

            # Load the label encoder
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)

            self._log(f"Loaded model from {model_path}")
            self._log(f"Loaded label encoder with {len(self.label_encoder.classes_)} classes")
            return True

        except Exception as e:
            self._log(f"Error loading model: {str(e)}")
            return False

    def predict(self, face_img):
        """
        Predict the identity of a face.

        Args:
            face_img: Face image (BGR format from OpenCV)

        Returns:
            (name, confidence): Predicted name and confidence score
        """
        if not TENSORFLOW_AVAILABLE or self.model is None or self.label_encoder is None:
            return "Unknown", 0.0

        try:
            # Preprocess the image
            img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, (224, 224))  # Resize to model input size
            img = img.astype('float32') / 255.0  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Make prediction
            predictions = self.model.predict(img)[0]

            # Get the predicted class and confidence
            predicted_class_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_class_idx])

            # Convert class index to name
            predicted_name = self.label_encoder.inverse_transform([predicted_class_idx])[0]

            return predicted_name, confidence

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
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
                'summary': "No model loaded"
            }

        classes = list(self.label_encoder.classes_) if self.label_encoder else []

        return {
            'loaded': True,
            'classes': classes,
            'summary': f"Model with {len(classes)} classes: {', '.join(classes)}"
        }
