import cv2
import os

def resize_images(input_dir, output_dir, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        if image is not None:
            resized_image = cv2.resize(image, target_size)
            cv2.imwrite(os.path.join(output_dir, img_name), resized_image)
        else:
            print(f"Could not open {img_name}")

# Example usage:
resize_images('./input_resize_images', './output', (224, 224))
