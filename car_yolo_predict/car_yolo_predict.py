import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import random
import easyocr

def display_images(post_training_files_path, image_files):

    for image_file in image_files:
        image_path = os.path.join(post_training_files_path, image_file)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 10), dpi=120)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

image_files = [
    'F1_curve.png',
    'P_curve.png',
    'R_curve.png',
    'PR_curve.png',
    'results.png'
]


post_training_files_path = 'runs1/detect/train'
#display_images(post_training_files_path, image_files)


model = YOLO('runs1/detect/train/weights/best.pt')

classes = ["licence_Plate"]

test_images_dir = 'datasets/cars_license_plate/test/images'

test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if f.endswith('.png') or f.endswith('.jpg')]

num_images = min(len(test_images), 16)

sample_images = random.sample(test_images, num_images)


fig, axs = plt.subplots(4, 4, figsize=(20, 20))

# Initialize the OCR reader
reader = easyocr.Reader(['en'])


for i, image_path in enumerate(sample_images):
    row = i // 4
    col = i % 4

    if image_path is None:
        axs[row, col].axis('off')
        continue

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model.predict(image, imgsz=640)  

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        class_id = int(result.cls[0])
        label = classes[class_id]

        # Crop the image within the bounding box
        cropped_image = image[y1:y2, x1:x2]
        cv2.imshow("image",cropped_image)
        cv2.waitKey(0)
        # Read text from the cropped image
        result = reader.readtext(cropped_image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', decoder='wordbeamsearch', beamWidth=100)
        text_to_write = ""  # Initialize an empty string
        if len(result) > 0:
            
            # Print the OCR result
            for text in result:
                text_to_write += text[1] + " "  # Concatenate the detected texts with a space

            text_to_write = text_to_write.strip()  # Remove leading/trailing whitespace
        label = label + ":" + text_to_write

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("image",image)
        cv2.waitKey(0)
        cv2.putText(image, text_to_write, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (152, 215, 235), 2)

    axs[row, col].imshow(image)
    axs[row, col].axis('off')

plt.tight_layout()
plt.savefig('test.png')
plt.show()
