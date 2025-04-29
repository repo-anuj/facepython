from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import time
import sys

# Try to import TensorFlow and MediaPipe first
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

# Now import our modules
from camera import Camera
from emotion_training import EmotionTrainer

app = Flask(__name__)
camera = None

# Ensure directories exist
if not os.path.exists('emotion_dataset'):
    os.makedirs('emotion_dataset')
    print("Created emotion_dataset directory")

if not os.path.exists('models'):
    os.makedirs('models')
    print("Created models directory")

# Check for emotion dataset folders
if os.path.exists('emotion_dataset'):
    folders = [d for d in os.listdir('emotion_dataset') if os.path.isdir(os.path.join('emotion_dataset', d))]
    if folders:
        print(f"Found emotion dataset folders: {', '.join(folders)}")
    else:
        print("No emotion dataset folders found")

emotion_trainer = EmotionTrainer(data_dir='emotion_dataset', model_dir='models')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen_frames():
    """Video streaming generator function."""
    global camera
    if camera is None:
        camera = Camera(emotion_trainer=emotion_trainer)

    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    """Capture and save a screenshot."""
    global camera
    if camera:
        timestamp = int(time.time())
        if not os.path.exists('static/captures'):
            os.makedirs('static/captures')
        filename = f'static/captures/capture_{timestamp}.jpg'
        success = camera.save_frame(filename)
        return jsonify({'success': success, 'filename': filename if success else None})
    return jsonify({'success': False})

@app.route('/toggle_filter', methods=['POST'])
def toggle_filter():
    """Toggle face filter."""
    filter_id = request.json.get('filter_id', 0)
    if camera:
        camera.set_filter(filter_id)
        return jsonify({'success': True, 'active_filter': filter_id})
    return jsonify({'success': False})

@app.route('/toggle_emotion', methods=['POST'])
def toggle_emotion():
    """Toggle emotion detection."""
    if camera:
        camera.toggle_emotion_detection()
        return jsonify({'success': True, 'emotion_detection': camera.emotion_detection})
    return jsonify({'success': False})

@app.route('/toggle_age_gender', methods=['POST'])
def toggle_age_gender():
    """Toggle age and gender estimation."""
    if camera:
        camera.toggle_age_gender()
        return jsonify({'success': True, 'age_gender': camera.age_gender_detection})
    return jsonify({'success': False})

@app.route('/toggle_ml_mode', methods=['POST'])
def toggle_ml_mode():
    """Toggle between ML-based and classic detection."""
    if camera:
        camera.use_ml_detection = not camera.use_ml_detection
        return jsonify({'success': True, 'ml_mode': camera.use_ml_detection})
    return jsonify({'success': False})

@app.route('/toggle_landmarks', methods=['POST'])
def toggle_landmarks():
    """Toggle facial landmarks display."""
    if camera:
        camera.show_landmarks = not camera.show_landmarks
        return jsonify({'success': True, 'landmarks': camera.show_landmarks})
    return jsonify({'success': False})

# Data collection routes
@app.route('/toggle_data_collection', methods=['POST'])
def toggle_data_collection():
    """Toggle data collection mode."""
    if camera:
        is_active = camera.toggle_data_collection()
        return jsonify({'success': True, 'data_collection': is_active})
    return jsonify({'success': False})

@app.route('/start_collection', methods=['POST'])
def start_collection():
    """Start collecting face images for a person."""
    if not camera:
        return jsonify({'success': False, 'message': 'Camera not initialized'})

    person_name = request.json.get('person_name', '')
    max_images = request.json.get('max_images', 50)

    success, message = camera.start_collection(person_name, max_images)
    return jsonify({'success': success, 'message': message})

@app.route('/stop_collection', methods=['POST'])
def stop_collection():
    """Stop the current collection process."""
    if not camera:
        return jsonify({'success': False, 'message': 'Camera not initialized'})

    success, message = camera.stop_collection()
    return jsonify({'success': success, 'message': message})

@app.route('/collection_status', methods=['GET'])
def collection_status():
    """Get the current status of data collection."""
    if not camera:
        return jsonify({'active': False})

    status = camera.get_collection_status()
    return jsonify(status)

@app.route('/collected_people', methods=['GET'])
def collected_people():
    """Get a list of people for whom data has been collected."""
    if not camera:
        return jsonify([])

    people = camera.get_collected_people()
    return jsonify(people)

# Face recognition routes
@app.route('/toggle_face_recognition', methods=['POST'])
def toggle_face_recognition():
    """Toggle face recognition on/off."""
    if camera:
        is_active = camera.toggle_face_recognition()
        return jsonify({'success': True, 'face_recognition': is_active})
    return jsonify({'success': False})

@app.route('/train_model', methods=['POST'])
def train_model():
    """Start training the face recognition model."""
    if not camera:
        return jsonify({'success': False, 'message': 'Camera not initialized'})

    epochs = request.json.get('epochs', 20)
    batch_size = request.json.get('batch_size', 32)

    success, message = camera.train_model(epochs=epochs, batch_size=batch_size)
    return jsonify({'success': success, 'message': message})

@app.route('/training_status', methods=['GET'])
def training_status():
    """Get the current status of model training."""
    if not camera:
        return jsonify({'is_training': False})

    status = camera.get_training_status()
    return jsonify(status)

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    if not camera:
        return jsonify({'loaded': False})

    info = camera.get_model_info()
    return jsonify(info)

# Emotion model training routes
@app.route('/train_emotion_model', methods=['POST'])
def train_emotion_model():
    """Start training the emotion recognition model."""
    epochs = request.json.get('epochs', 50)
    batch_size = request.json.get('batch_size', 32)

    # Check if the emotion_dataset directory exists and has data
    if os.path.exists('emotion_dataset'):
        folders = [d for d in os.listdir('emotion_dataset') if os.path.isdir(os.path.join('emotion_dataset', d))]
        if folders:
            print(f"Found emotion dataset folders: {', '.join(folders)}")
        else:
            print("No emotion dataset folders found")
    else:
        print("emotion_dataset directory not found")

    success, message = emotion_trainer.train_model(epochs=epochs, batch_size=batch_size)
    return jsonify({'success': success, 'message': message})

@app.route('/emotion_training_status', methods=['GET'])
def emotion_training_status():
    """Get the current status of emotion model training."""
    status = emotion_trainer.get_training_status()
    return jsonify(status)

@app.route('/emotion_model_info', methods=['GET'])
def emotion_model_info():
    """Get information about the loaded emotion model."""
    info = emotion_trainer.get_model_info()
    return jsonify(info)

@app.route('/load_emotion_model', methods=['POST'])
def load_emotion_model():
    """Load the trained emotion model."""
    success = emotion_trainer.load_model()
    return jsonify({'success': success})

if __name__ == '__main__':
    app.run(debug=True)
