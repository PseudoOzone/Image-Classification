# CIFAR-10 Image Classifier with Flask Deployment

A full-stack deep learning application that trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset and deploys it as a web application using Flask. This project demonstrates end-to-end machine learning workflow from model training to production deployment.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.1-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

- **Deep Learning Model**: CNN architecture optimized for CIFAR-10 image classification
- **Web Interface**: Modern, responsive UI with drag-and-drop image upload
- **Real-time Prediction**: Instant image classification with confidence scores
- **Complete Predictions**: View predictions for all 10 CIFAR-10 classes
- **Image Preprocessing**: Automatic image resizing and normalization
- **Production Ready**: Flask-based deployment with error handling

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [API Endpoints](#api-endpoints)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a complete machine learning pipeline:

1. **Data Preprocessing**: Loads and normalizes CIFAR-10 dataset
2. **Model Training**: Trains a CNN with 3 convolutional blocks
3. **Model Persistence**: Saves trained model to disk
4. **Web Deployment**: Flask application for real-time predictions

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## 📁 Project Structure

```
Image Classification/
│
├── requirements.txt              # Python dependencies
├── cnn_model.py                  # CNN model architecture
├── dataset_preprocessing.py      # Data loading and preprocessing
├── train_model.py                # Model training script
├── flask_app.py                  # Flask web application
├── image_classifier_model.h5      # Trained model (generated after training)
│
└── templates/
    └── index.html                # Web interface
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/image-classification.git
cd image-classification
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `tensorflow` - Deep learning framework
- `keras` - High-level neural networks API
- `numpy` - Numerical computing
- `flask` - Web framework
- `opencv-python` - Image processing

## 💻 Usage

### Step 1: Train the Model

Train the CNN model on the CIFAR-10 dataset:

```bash
python train_model.py
```

This will:
- Download the CIFAR-10 dataset (if not already present)
- Preprocess the data (normalize pixel values to [0, 1])
- Create and compile the CNN model
- Train for 5 epochs
- Save the model as `image_classifier_model.h5`
- Display training progress and final test accuracy

**Expected Output:**
- Training accuracy: ~71%
- Test accuracy: ~69%

### Step 2: Start the Flask Application

```bash
python flask_app.py
```

The application will start on `http://localhost:5001` (or port 5000 if available).

**Note:** On macOS, port 5000 may be occupied by AirPlay Receiver. The app automatically uses port 5001 in such cases.

### Step 3: Use the Web Interface

1. Open your browser and navigate to `http://localhost:5001`
2. Click "Choose Image" or drag and drop an image file
3. Click "Classify Image" to get predictions
4. View the predicted class and confidence scores for all 10 categories

## 🧠 Model Architecture

The CNN model consists of:

```
Input Layer: (32, 32, 3)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
Flatten
    ↓
Dense (64 units) + ReLU
    ↓
Dense (10 units) + Softmax
```

**Model Summary:**
- **Total Parameters**: ~120,000+
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

## 📊 Dataset

The CIFAR-10 dataset is automatically downloaded from Keras datasets on first run.

- **Training Set**: 50,000 images
- **Test Set**: 10,000 images
- **Image Size**: 32x32 pixels
- **Channels**: RGB (3 channels)
- **Classes**: 10

## 🛠 Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Flask**: Web application framework
- **OpenCV**: Image processing and preprocessing
- **NumPy**: Numerical operations
- **HTML/CSS/JavaScript**: Frontend interface

## 🔌 API Endpoints

### GET `/`
Renders the main web interface.

### POST `/predict`
Accepts an image file and returns classification results.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file (png, jpg, jpeg, gif, bmp)

**Response:**
```json
{
  "predicted_class": "cat",
  "confidence": 85.23,
  "all_predictions": {
    "airplane": 2.15,
    "automobile": 1.23,
    "bird": 3.45,
    "cat": 85.23,
    "deer": 1.12,
    "dog": 4.56,
    "frog": 0.89,
    "horse": 0.67,
    "ship": 0.34,
    "truck": 0.36
  }
}
```

## 📸 Screenshots

*Add screenshots of your application here*

## 🔧 Configuration

### Changing Training Parameters

Edit `train_model.py` to modify:
- Number of epochs (default: 5)
- Batch size (default: 32)
- Validation split

### Changing Model Architecture

Edit `cnn_model.py` to modify:
- Number of convolutional layers
- Filter sizes
- Dense layer units
- Activation functions

### Changing Flask Port

Edit `flask_app.py` line 129:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

## 🐛 Troubleshooting

### Port Already in Use
If port 5000/5001 is already in use:
- On macOS: Disable AirPlay Receiver in System Preferences
- Or change the port in `flask_app.py`

### Model Not Found Error
Make sure to run `python train_model.py` before starting the Flask app to generate `image_classifier_model.h5`.

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- CIFAR-10 dataset creators
- TensorFlow/Keras team
- Flask development community

---

⭐ If you found this project helpful, please consider giving it a star!

