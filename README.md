# FacePython - Advanced Face Detection Web Application with Machine Learning

FacePython is a feature-rich face detection web application built with Python, Flask, OpenCV, and TensorFlow. It offers real-time face detection with various cool features through a user-friendly web interface, enhanced with machine learning capabilities.

## Features

### Basic Features
- Real-time face detection using webcam
- Face filters
- Screenshot capture functionality
- Face counting (detect multiple faces)

### Machine Learning Features
- Advanced face detection using MediaPipe
- Facial landmarks detection (68 points)
- ML-based emotion recognition
- Age and gender estimation using neural networks
- Toggle between classic and ML-based detection

### Data Collection & Model Training
- Collect face images via webcam for training
- Organize collected images by person
- Train a facial recognition model using VGG16
- Real-time face recognition of trained individuals

## Requirements

- Python 3.6+ (tested with Python 3.12)
- Webcam (laptop built-in or external)
- Required Python packages (see requirements.txt):
  - Flask
  - OpenCV
  - NumPy
  - TensorFlow (CPU version)
  - MediaPipe
  - scikit-learn

## Installation

1. Clone this repository or download the source code.

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   Note: The installation might take some time as TensorFlow and MediaPipe are large packages.

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Grant permission to access your webcam when prompted.

4. Enjoy the face detection features:
   - Click "Take Screenshot" to capture the current frame
   - Select different filters to apply them to detected faces
   - Toggle emotion detection and age/gender estimation
   - Toggle between ML-based and classic detection modes
   - Enable facial landmarks visualization
   - View and download captured images in the gallery

5. Collect face data for training:
   - Toggle "Collection Mode" to enable data collection
   - Enter a person's name and optionally set the number of images to collect
   - Click "Start Collection" to begin capturing face images
   - The application will automatically capture images when a face is detected
   - Click "Stop Collection" to end the collection process early

6. Train the face recognition model:
   - After collecting data for one or more people, go to the "Model Training" section
   - Optionally adjust the number of epochs and batch size
   - Click "Train Model" to start the training process
   - The training progress will be displayed in real-time
   - Once training is complete, toggle "Face Recognition" to start recognizing faces

## Customization

### Adding Custom Filters

To add custom filters, place PNG images with transparent backgrounds in the `static/filters` directory. The application will automatically detect and load them.

### Extending Functionality

The modular design of FacePython makes it easy to extend with additional features:

1. Add new detection methods in the `Camera` class in `camera.py`
2. Create corresponding UI controls in `templates/index.html`
3. Add event handlers in `static/js/main.js`

## Troubleshooting

- **Camera not working**: Make sure your webcam is properly connected and not being used by another application.
- **Slow performance**: Disable ML mode and use classic mode instead, or disable facial landmarks and other computationally intensive features.
- **Installation issues**: If you encounter issues with TensorFlow or MediaPipe, try installing them separately with `pip install tensorflow-cpu` and `pip install mediapipe`.
- **Model loading errors**: The application will create a simple demo emotion model on first run. For better accuracy, you can replace it with a pre-trained model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for the computer vision capabilities
- Flask for the web framework
- TensorFlow for the machine learning capabilities
- MediaPipe for the advanced face detection and landmark detection
- Google for developing and maintaining MediaPipe
- The open-source community for providing resources and examples
