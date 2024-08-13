import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model/rafter_detection_model.keras')

def predict(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)

    # Make a prediction
    predictions = model.predict(image_resized)
    print(predictions)  # This will give you the model's prediction

# Example usage:
predict('dataset/images/test/17.jpg')
