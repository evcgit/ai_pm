import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np

def load_data(image_dir, label_dir):
    images = []
    labels_bbox = []
    labels_class = []

    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        label_file = img_file.replace('.jpeg', '.txt').replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"Warning: Label file {label_path} not found for image {img_file}")
            continue
        
        # Load the image
        image = cv2.imread(img_path)
        images.append(image)

        # Load and preprocess the label
        with open(label_path, 'r') as f:
            line = f.readline()
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = float(parts[0])  # Class ID
                bbox = [float(x) for x in parts[1:]]
                
                labels_bbox.append(bbox)
                labels_class.append(class_id)

    images = np.array(images) / 255.0
    labels_bbox = np.array(labels_bbox)
    labels_class = np.array(labels_class).reshape(-1, 1)  # Reshape to (None, 1)

    return images, {'bbox_output': labels_bbox, 'class_output': labels_class}


def create_model(input_shape, num_classes=1):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Output for bounding box predictions (only one box, so 4 coordinates)
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox_output')(x)

    # Output for class predictions (binary classification)
    class_output = layers.Dense(1, activation='sigmoid', name='class_output')(x)

    model = models.Model(inputs=inputs, outputs=[bbox_output, class_output])

    model.compile(optimizer='adam',
                  loss={'bbox_output': 'mean_squared_error',
                        'class_output': 'binary_crossentropy'},
                  metrics={'bbox_output': 'mse',
                           'class_output': 'accuracy'})

    return model

train_images, train_labels = load_data('dataset/images/train', 'dataset/labels/train')
test_images, test_labels = load_data('dataset/images/test', 'dataset/labels/test')

input_shape = (224, 224, 3)  # Adjusted input shape to match the new image size

model = create_model(input_shape)
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
losses = model.evaluate(test_images, test_labels)
print(f"Total Loss: {losses[0]}")
print(f"BBox MSE: {losses[1]}")
print(f"Classification Accuracy: {losses[2]}")

# Save the model
model.save("model/pmt_detection_model.keras")
