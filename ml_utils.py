import os
import numpy as np
import cv2
import random

# Try to import TensorFlow and MediaPipe, but continue if not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow successfully imported")
except ImportError as e:
    print(f"Warning: {e}")
    print("Running in fallback mode without TensorFlow")
    TENSORFLOW_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe successfully imported")
except ImportError as e:
    print(f"Warning: {e}")
    print("Running in fallback mode without MediaPipe")
    MEDIAPIPE_AVAILABLE = False

if TENSORFLOW_AVAILABLE:
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.image import img_to_array
        print("Keras modules successfully imported")
    except ImportError as e:
        print(f"Warning: {e}")
        print("Some TensorFlow/Keras modules could not be imported")
        TENSORFLOW_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available")

# Initialize MediaPipe Face Detection if available
if MEDIAPIPE_AVAILABLE:
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
else:
    mp_face_detection = None
    mp_drawing = None
    mp_face_mesh = None

# Constants
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
AGE_RANGES = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
GENDERS = ['Male', 'Female']

# Model paths
EMOTION_MODEL_PATH = os.path.join('models', 'emotion_model.h5')
AGE_MODEL_PATH = os.path.join('models', 'age_model.h5')
GENDER_MODEL_PATH = os.path.join('models', 'gender_model.h5')

# Face recognition database
face_db = {}

class MLProcessor:
    def __init__(self):
        # Initialize variables
        self.face_detection = None
        self.face_mesh = None
        self.emotion_model = None
        self.age_model = None
        self.gender_model = None

        if MEDIAPIPE_AVAILABLE:
            # Initialize MediaPipe face detection
            self.face_detection = mp_face_detection.FaceDetection(
                min_detection_confidence=0.5)

            # Initialize MediaPipe face mesh (for landmarks)
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)

        if TENSORFLOW_AVAILABLE:
            # Load models if they exist
            self.emotion_model = self._load_model(EMOTION_MODEL_PATH)
            self.age_model = self._load_model(AGE_MODEL_PATH)
            self.gender_model = self._load_model(GENDER_MODEL_PATH)

        print("ML Processor initialized")
        print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")
        print(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")
        print(f"Emotion model loaded: {self.emotion_model is not None}")
        print(f"Age model loaded: {self.age_model is not None}")
        print(f"Gender model loaded: {self.gender_model is not None}")

    def _load_model(self, model_path):
        """Load a TensorFlow model if it exists."""
        if os.path.exists(model_path):
            try:
                return load_model(model_path)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                return None
        else:
            print(f"Model not found: {model_path}")
            return None

    def detect_faces(self, frame):
        """Detect faces using MediaPipe or fallback to Haar cascade."""
        faces = []

        if not MEDIAPIPE_AVAILABLE or self.face_detection is None:
            # Fallback to Haar cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Convert to our format with confidence score
            for (x, y, w, h) in haar_faces:
                faces.append((x, y, w, h, 0.9))  # Arbitrary confidence score

            return faces

        # MediaPipe is available, use it
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape

                # Convert relative coordinates to absolute
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Ensure coordinates are within frame boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)

                # Add face to list
                faces.append((x, y, w, h, detection.score[0]))

        return faces

    def detect_landmarks(self, frame):
        """Detect facial landmarks using MediaPipe Face Mesh."""
        landmarks_list = []

        if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
            # If MediaPipe is not available, return empty list
            return landmarks_list

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert landmarks to pixel coordinates
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    landmarks.append((x, y))

                landmarks_list.append(landmarks)

                # Draw landmarks on the frame (optional)
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        return landmarks_list

    def predict_emotion(self, frame, face):
        """Predict emotion for a face."""
        if not TENSORFLOW_AVAILABLE or self.emotion_model is None:
            # If TensorFlow is not available or model is not loaded, return random emotion
            return random.choice(EMOTIONS)

        try:
            x, y, w, h, _ = face

            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Resize to model input size
            face_roi = cv2.resize(face_roi, (48, 48))

            # Convert to grayscale (if your model was trained on grayscale)
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Normalize pixel values
            roi = gray_roi.astype('float') / 255.0

            # Reshape for model input
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Make prediction
            prediction = self.emotion_model.predict(roi)[0]

            # Get the emotion with highest probability
            emotion_idx = np.argmax(prediction)
            emotion = EMOTIONS[emotion_idx]

            return emotion
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return random.choice(EMOTIONS)

    def predict_age_gender(self, frame, face):
        """Predict age and gender for a face."""
        if not TENSORFLOW_AVAILABLE or (self.age_model is None and self.gender_model is None):
            # If TensorFlow is not available or models are not loaded, return random values
            age = random.choice(AGE_RANGES)
            gender = random.choice(GENDERS)
            return age, gender

        try:
            x, y, w, h, _ = face

            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Resize to model input size (224x224 is common for many models)
            face_roi = cv2.resize(face_roi, (224, 224))

            # Preprocess for model
            roi = face_roi.astype('float') / 255.0

            # Default values
            age = random.choice(AGE_RANGES)
            gender = random.choice(GENDERS)

            if TENSORFLOW_AVAILABLE:
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Predict age if model is available
                if self.age_model is not None:
                    age_pred = self.age_model.predict(roi)[0]
                    age_idx = np.argmax(age_pred)
                    age = AGE_RANGES[age_idx]

                # Predict gender if model is available
                if self.gender_model is not None:
                    gender_pred = self.gender_model.predict(roi)[0]
                    gender_idx = np.argmax(gender_pred)
                    gender = GENDERS[gender_idx]

            return age, gender
        except Exception as e:
            print(f"Error predicting age/gender: {e}")
            return random.choice(AGE_RANGES), random.choice(GENDERS)

    def recognize_face(self, frame, face, name=None):
        """
        Recognize a face or register a new face.
        If name is provided, register the face with that name.
        Otherwise, try to recognize the face.
        """
        x, y, w, h, _ = face

        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Convert to RGB (face_recognition uses RGB)
        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        # Resize for consistency
        rgb_roi = cv2.resize(rgb_roi, (128, 128))

        # Flatten the image to create a simple face encoding
        # Note: This is a very basic approach. In a real system, you'd use a proper face embedding model
        simple_encoding = rgb_roi.flatten().mean(axis=0)

        # If name is provided, register the face
        if name:
            face_db[name] = simple_encoding
            return name

        # Otherwise, try to recognize the face
        if not face_db:
            return "Unknown"

        # Find the closest match
        min_dist = float('inf')
        match_name = "Unknown"

        for db_name, db_encoding in face_db.items():
            # Calculate Euclidean distance
            dist = np.linalg.norm(simple_encoding - db_encoding)

            # If distance is below threshold, consider it a match
            if dist < min_dist and dist < 50:  # Threshold value needs tuning
                min_dist = dist
                match_name = db_name

        return match_name

    def draw_results(self, frame, face, emotion=None, age=None, gender=None, name=None):
        """Draw detection results on the frame."""
        x, y, w, h, confidence = face

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw confidence
        conf_text = f"Conf: {confidence:.2f}"
        cv2.putText(frame, conf_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw emotion if available
        if emotion:
            cv2.putText(frame, f"Emotion: {emotion}", (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw age and gender if available
        if age and gender:
            cv2.putText(frame, f"{gender}, {age}", (x, y+h+45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw name if available
        if name:
            cv2.putText(frame, name, (x, y-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        return frame

# Function to download models (placeholder)
def download_models():
    """
    Download pre-trained models if they don't exist.
    This is a placeholder function - in a real implementation,
    you would download actual models from a server or repository.
    """
    # Check if models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Check for emotion model
    if not os.path.exists(EMOTION_MODEL_PATH):
        print(f"Emotion model not found at {EMOTION_MODEL_PATH}")
        print("In a real implementation, this would download the model.")

    # Check for age model
    if not os.path.exists(AGE_MODEL_PATH):
        print(f"Age model not found at {AGE_MODEL_PATH}")
        print("In a real implementation, this would download the model.")

    # Check for gender model
    if not os.path.exists(GENDER_MODEL_PATH):
        print(f"Gender model not found at {GENDER_MODEL_PATH}")
        print("In a real implementation, this would download the model.")

# Create a simple emotion recognition model (for demonstration)
def create_demo_emotion_model():
    """Create a simple CNN model for emotion recognition."""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available, skipping model creation")
        return

    if os.path.exists(EMOTION_MODEL_PATH):
        print(f"Emotion model already exists at {EMOTION_MODEL_PATH}")
        return

    try:
        # Create a simple CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(EMOTIONS), activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # Save the model
        model.save(EMOTION_MODEL_PATH)
        print(f"Demo emotion model created and saved to {EMOTION_MODEL_PATH}")
    except Exception as e:
        print(f"Error creating demo emotion model: {e}")

# Initialize
if __name__ == "__main__":
    download_models()
    create_demo_emotion_model()
