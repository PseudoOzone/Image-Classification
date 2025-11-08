from tensorflow.keras.datasets import cifar10
import numpy as np


def load_and_preprocess_cifar10():
    """
    Load and preprocess the CIFAR-10 dataset.
    
    Returns:
        Tuple of (x_train, y_train), (x_test, y_test) with normalized pixel values
    """
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    return (x_train, y_train), (x_test, y_test)

