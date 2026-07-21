# CIFAR-10 CNN with Flask Demo

An educational end-to-end image-classification project that trains a small convolutional neural network on CIFAR-10 and serves predictions through a Flask web interface.

> **Status:** learning project and local demo. It is not production-ready and is not intended for safety-critical or real-world image-recognition decisions.

## What it demonstrates

- Loading and normalizing CIFAR-10 with Keras
- Building a three-block convolutional neural network
- Training and saving a TensorFlow/Keras model
- Accepting an image through a Flask endpoint
- Resizing and normalizing the uploaded image
- Returning the top class and class probabilities
- Rendering a simple browser interface

## Model architecture

```text
32 x 32 RGB image
    |
Conv2D(32, 3x3) + ReLU
    |
MaxPooling2D
    |
Conv2D(64, 3x3) + ReLU
    |
MaxPooling2D
    |
Conv2D(64, 3x3) + ReLU
    |
Flatten
    |
Dense(64) + ReLU
    |
Dense(10) + Softmax
```

The ten CIFAR-10 labels are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Repository structure

```text
Image-Classification/
├── cnn_model.py
├── dataset_preprocessing.py
├── train_model.py
├── flask_app.py
├── requirements.txt
├── templates/
└── image_classifier_model.h5   # generated after training, if retained locally
```

## Setup

```bash
git clone https://github.com/PseudoOzone/Image-Classification.git
cd Image-Classification

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

Training downloads CIFAR-10 if necessary and writes the model file expected by the Flask application.

Do not cite an accuracy value unless it comes from the current run output or a committed evaluation artifact. Accuracy changes with TensorFlow versions, random initialization, epochs, and training configuration.

## Run the web application

```bash
python flask_app.py
```

Open the local URL shown in the terminal, upload an image, and inspect the returned class probabilities.

CIFAR-10 contains low-resolution 32 x 32 training images. Predictions on arbitrary photographs can be poor even when the model performs reasonably on the benchmark test set.

## API

### `GET /`

Renders the upload page.

### `POST /predict`

Accepts a multipart image upload and returns a JSON prediction when the model is loaded successfully.

## Known limitations

- The application checks file extensions but does not verify the true file type before decoding.
- Invalid or corrupted images can cause OpenCV preprocessing errors.
- Uploaded files are stored on disk and are not automatically removed.
- Reusing the same filename can overwrite an earlier upload.
- The model is loaded when the module imports, making startup dependent on the local model file.
- Running Flask with debug mode on `0.0.0.0` is unsafe outside a trusted local environment.
- There is no authentication, rate limiting, CSRF protection, malware scanning, or production storage policy.
- The network is a small tutorial architecture without augmentation, regularization, calibration, or modern benchmark comparison.
- There is no automated test suite or continuous-integration workflow.

## Recommended improvements

- decode uploads in memory and reject invalid images
- generate unique temporary filenames and delete them after inference
- add pixel-count and file-size safeguards
- use an application factory and explicit model lifecycle
- disable debug mode by default
- add tests for preprocessing and API error cases
- record deterministic seeds and evaluation results
- add data augmentation, early stopping, calibration, and confusion-matrix reporting
- package the model with a model card describing intended use and limitations

## License

Educational and portfolio use only unless a separate license file states otherwise.
