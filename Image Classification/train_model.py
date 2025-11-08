from cnn_model import create_cnn_model
from dataset_preprocessing import load_and_preprocess_cifar10


def main():
    """
    Main function to train the CNN model on CIFAR-10 dataset.
    """
    print("Loading and preprocessing CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    print("\nCreating CNN model...")
    model = create_cnn_model()
    
    print("\nModel architecture:")
    model.summary()
    
    print("\nTraining the model...")
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    print("\nEvaluating the model on test data...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    print("\nSaving the model to image_classifier_model.h5...")
    model.save('image_classifier_model.h5')
    print("Model saved successfully!")


if __name__ == '__main__':
    main()

