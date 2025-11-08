from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
print("Loading the trained model...")
try:
    model = load_model('image_classifier_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure to train the model first by running: python train_model.py")
    model = None

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Preprocess the uploaded image for prediction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image array ready for model prediction
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert BGR to RGB (OpenCV reads in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 32x32 (CIFAR-10 input size)
    img = cv2.resize(img, (32, 32))
    
    # Normalize pixel values to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Reshape for model input: (1, 32, 32, 3)
    img = np.expand_dims(img, axis=0)
    
    return img


@app.route('/')
def index():
    """Render the main page with image upload form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    # Check if file is present in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image (png, jpg, jpeg, gif, bmp)'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image
        processed_image = preprocess_image(filepath)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get the predicted class name
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 2),
            'all_predictions': {
                CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            }
        })
    
    except Exception as e:
        # Clean up file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

