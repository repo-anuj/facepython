import cv2
import numpy as np
import os
import random  # For simulating emotion and age/gender detection
from ml_utils import MLProcessor, create_demo_emotion_model
from data_collection import DataCollector
from model_training import ModelTrainer

class Camera:
    def __init__(self, emotion_trainer=None):
        self.video = cv2.VideoCapture(0)

        # Initialize ML processor
        print("Initializing ML processor...")
        create_demo_emotion_model()  # Create a demo emotion model
        self.ml_processor = MLProcessor()

        # Store the emotion trainer
        self.emotion_trainer = emotion_trainer

        # Initialize data collector and model trainer
        self.data_collector = DataCollector(base_dir='collected_data')
        self.model_trainer = ModelTrainer(data_dir='collected_data', model_dir='models')

        # Try to load a pre-trained model if available
        self.model_trainer.load_latest_model()

        # Keep the Haar cascade as fallback
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Load filters
        self.filters = []
        self.load_filters()
        self.active_filter = 0  # 0 means no filter

        # Feature toggles
        self.emotion_detection = False
        self.age_gender_detection = False
        self.use_ml_detection = True  # Use ML-based detection by default
        self.show_landmarks = False  # Toggle for facial landmarks
        self.face_recognition = False  # Toggle for face recognition
        self.data_collection_mode = False  # Toggle for data collection mode

    def load_filters(self):
        """Load filter images for face overlays."""
        filter_dir = 'static/filters'
        if not os.path.exists(filter_dir):
            os.makedirs(filter_dir)
            # You would add actual filter images to this directory

        # Default empty filter (no filter)
        self.filters.append(None)

        # Example: Add more filters here when you have the images
        # filter_files = os.listdir(filter_dir)
        # for filter_file in filter_files:
        #     filter_path = os.path.join(filter_dir, filter_file)
        #     filter_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)
        #     self.filters.append(filter_img)

    def load_models(self):
        """Load models for emotion recognition and age/gender estimation."""
        # These would be actual model paths in a real implementation
        # For now, we'll just set placeholders
        self.emotion_model = None
        self.age_gender_model = None

        # In a real implementation, you would load models like:
        # self.emotion_model = cv2.dnn.readNetFromCaffe('path_to_model.prototxt', 'path_to_weights.caffemodel')

    def set_filter(self, filter_id):
        """Set the active filter."""
        if 0 <= filter_id < len(self.filters):
            self.active_filter = filter_id

    def toggle_emotion_detection(self):
        """Toggle emotion detection on/off."""
        self.emotion_detection = not self.emotion_detection

    def toggle_age_gender(self):
        """Toggle age and gender estimation on/off."""
        self.age_gender_detection = not self.age_gender_detection

    def detect_faces(self, frame):
        """Detect faces in the frame using ML or fallback to Haar cascade."""
        if self.use_ml_detection:
            # Use ML-based face detection
            ml_faces = self.ml_processor.detect_faces(frame)

            # Detect facial landmarks if enabled
            if self.show_landmarks:
                self.ml_processor.detect_landmarks(frame)

            # Process each detected face
            for face in ml_faces:
                x, y, w, h, _ = face

                # Extract face ROI for recognition and data collection
                face_roi = frame[y:y+h, x:x+w].copy()

                # Apply filter if active
                if self.active_filter > 0 and self.filters[self.active_filter] is not None:
                    self.apply_filter(frame, x, y, w, h)

                # Collect face data if in collection mode
                if self.data_collection_mode:
                    collection_status = self.data_collector.get_collection_status()
                    if collection_status['active']:
                        success, message = self.data_collector.collect_face(frame, (x, y, w, h))
                        if success:
                            # Draw collection progress
                            progress = collection_status['progress']
                            cv2.rectangle(frame, (x, y-30), (x + int(w * progress/100), y-20), (0, 255, 0), -1)
                            cv2.putText(frame, f"Collecting: {collection_status['count']}/{collection_status['max']}",
                                       (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Recognize face if enabled
                if self.face_recognition:
                    name, confidence = self.model_trainer.predict(face_roi)
                    if confidence > 0.5:  # Only show high-confidence predictions
                        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Detect emotion if enabled
                if self.emotion_detection:
                    emotion = self.ml_processor.predict_emotion(frame, face)
                    self.ml_processor.draw_results(frame, face, emotion=emotion)
                else:
                    self.ml_processor.draw_results(frame, face)

                # Estimate age and gender if enabled
                if self.age_gender_detection:
                    age, gender = self.ml_processor.predict_age_gender(frame, face)
                    self.ml_processor.draw_results(frame, face, age=age, gender=gender)

            # Add face count
            cv2.putText(frame, f'Faces: {len(ml_faces)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add ML mode indicator
            cv2.putText(frame, "ML Mode", (frame.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            # Fallback to Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Extract face ROI for recognition and data collection
                face_roi = frame[y:y+h, x:x+w].copy()

                # Detect eyes
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                try:
                    # Check if ROI is valid
                    if roi_gray.size > 0 and roi_gray.shape[0] > 0 and roi_gray.shape[1] > 0:
                        eyes = self.eye_cascade.detectMultiScale(roi_gray)
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                except cv2.error as e:
                    print(f"Eye detection error: {e}")
                    # Continue without eye detection

                # Apply filter if active
                if self.active_filter > 0 and self.filters[self.active_filter] is not None:
                    self.apply_filter(frame, x, y, w, h)

                # Collect face data if in collection mode
                if self.data_collection_mode:
                    collection_status = self.data_collector.get_collection_status()
                    if collection_status['active']:
                        success, message = self.data_collector.collect_face(frame, (x, y, w, h))
                        if success:
                            # Draw collection progress
                            progress = collection_status['progress']
                            cv2.rectangle(frame, (x, y-30), (x + int(w * progress/100), y-20), (0, 255, 0), -1)
                            cv2.putText(frame, f"Collecting: {collection_status['count']}/{collection_status['max']}",
                                       (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Recognize face if enabled
                if self.face_recognition:
                    name, confidence = self.model_trainer.predict(face_roi)
                    if confidence > 0.5:  # Only show high-confidence predictions
                        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Detect emotion if enabled
                if self.emotion_detection:
                    self.detect_emotion(frame, x, y, w, h)

                # Estimate age and gender if enabled
                if self.age_gender_detection:
                    self.estimate_age_gender(frame, x, y, w, h)

            # Add face count
            cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add classic mode indicator
            cv2.putText(frame, "Classic Mode", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def apply_filter(self, frame, x, y, w, h):
        """Apply filter overlay to face."""
        # This is a placeholder. In a real implementation, you would:
        # 1. Resize the filter to match the face size
        # 2. Blend the filter with the face region using alpha compositing
        # For now, we'll just add a text label
        cv2.putText(frame, f'Filter {self.active_filter}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def detect_emotion(self, frame, x, y, w, h):
        """Detect emotion on face."""
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w].copy()

        # Use the emotion trainer if available
        if self.emotion_trainer and self.emotion_trainer.model is not None:
            try:
                emotion, confidence = self.emotion_trainer.predict_emotion(face_roi)
                # Display the detected emotion with confidence
                cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return
            except Exception as e:
                print(f"Error using emotion trainer: {e}")
                # Fall back to random if there's an error

        # Fallback to random emotion if no trainer or error occurred
        emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']
        emotion = random.choice(emotions)  # Random for demo
        cv2.putText(frame, emotion, (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def estimate_age_gender(self, frame, x, y, w, h):
        """Estimate age and gender."""
        # This is a placeholder. In a real implementation, you would:
        # 1. Extract the face region
        # 2. Preprocess it for the age/gender model
        # 3. Run inference with the model
        # 4. Display the estimated age and gender
        genders = ['Male', 'Female']
        gender = random.choice(genders)  # Random for demo
        age = random.randint(20, 40)  # Random for demo
        cv2.putText(frame, f'{gender}, {age}', (x, y+h+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def get_frame(self):
        """Get frame from camera with face detection applied."""
        success, frame = self.video.read()
        if success:
            # Flip the frame for a more natural view
            frame = cv2.flip(frame, 1)

            # Apply face detection
            frame = self.detect_faces(frame)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
        return None

    def save_frame(self, filename):
        """Save current frame to file."""
        success, frame = self.video.read()
        if success:
            frame = cv2.flip(frame, 1)
            frame = self.detect_faces(frame)
            cv2.imwrite(filename, frame)
            return True
        return False

    # Data collection methods
    def toggle_data_collection(self):
        """Toggle data collection mode on/off."""
        self.data_collection_mode = not self.data_collection_mode
        if not self.data_collection_mode:
            # Stop any active collection when turning off collection mode
            self.data_collector.stop_collection()
        return self.data_collection_mode

    def start_collection(self, person_name, max_images=50):
        """Start collecting face images for a person."""
        success, message = self.data_collector.start_collection(person_name, max_images)
        if success:
            self.data_collection_mode = True
        return success, message

    def stop_collection(self):
        """Stop the current collection process."""
        return self.data_collector.stop_collection()

    def get_collection_status(self):
        """Get the current status of data collection."""
        return self.data_collector.get_collection_status()

    def get_collected_people(self):
        """Get a list of people for whom data has been collected."""
        return self.data_collector.get_collected_people()

    # Face recognition methods
    def toggle_face_recognition(self):
        """Toggle face recognition on/off."""
        self.face_recognition = not self.face_recognition
        return self.face_recognition

    def train_model(self, epochs=20, batch_size=32):
        """Start training the face recognition model."""
        return self.model_trainer.train_model(epochs=epochs, batch_size=batch_size)

    def get_training_status(self):
        """Get the current status of model training."""
        return self.model_trainer.get_training_status()

    def get_model_info(self):
        """Get information about the loaded model."""
        return self.model_trainer.get_model_info()

    def __del__(self):
        """Release video capture when object is deleted."""
        self.video.release()
