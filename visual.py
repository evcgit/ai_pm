import tensorflow as tf
import cv2
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('model/pmt_detection_model.keras')

def visualize_predictions(image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    original_image = image.copy()
    original_height, original_width = image.shape[:2]

    # The image is already 224x224, so no need to resize or scale
    image_resized = image / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension

    # Make predictions
    bbox_pred, class_pred = model.predict(image_resized)
    bbox_pred = bbox_pred[0]  # Remove batch dimension
    class_pred = class_pred[0][0]  # Remove batch dimension and get class prediction

    # Print predictions for debugging
    print(f"Bounding box prediction: {bbox_pred}")
    print(f"Class prediction: {class_pred}")

    # Convert predicted bounding box coordinates to pixel values
    bbox_pred[0] *= original_width  # x_center
    bbox_pred[1] *= original_height  # y_center
    bbox_pred[2] *= original_width  # width
    bbox_pred[3] *= original_height  # height

    # Calculate the top-left and bottom-right corners of the bounding box
    x_min = int(bbox_pred[0] - bbox_pred[2] / 2)
    y_min = int(bbox_pred[1] - bbox_pred[3] / 2)
    x_max = int(bbox_pred[0] + bbox_pred[2] / 2)
    y_max = int(bbox_pred[1] + bbox_pred[3] / 2)

    # Ensure the coordinates are within the image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(original_width, x_max)
    y_max = min(original_height, y_max)

    # Print calculated coordinates for debugging
    print(f"Top-left corner: ({x_min}, {y_min}), Bottom-right corner: ({x_max}, {y_max})")

    # Draw the bounding box on the original image
    if x_min < x_max and y_min < y_max:
        color = (0, 255, 0) if class_pred > 0.5 else (0, 0, 255)  # Green for positive, red for negative
        cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.imshow('Predicted Image', original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Invalid bounding box coordinates, skipping drawing.")

# Example usage
visualize_predictions('dataset/images/test/21.jpeg', model)
