import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Function to save samples as images
def save_samples(images, labels, num_samples=5, save_dir='mnist_samples'):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_samples):
        img = images[i]
        label = labels[i]
        
        img = img[:, :, np.newaxis]  # This gives shape (28, 28, 1)
        img = img.astype(np.uint8)        
        pil_image = Image.fromarray(img.squeeze(), mode='L') 
        
        # Create a file path using the label
        file_path = os.path.join(save_dir, f'label_{label}_img_{i + 1}.png')
        
        # Save the image
        pil_image.save(file_path, format='PNG')
        print(img.shape)
        print(f'Saved {file_path}')

# Save a subset of training images to files
save_samples(train_images, train_labels, num_samples=5)