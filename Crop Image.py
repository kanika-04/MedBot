import cv2
import pytesseract
import re
import glob
import os
import matplotlib.pyplot as plt
from google.colab import files

# Function to upload an image
def upload_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        return filename

# Upload the image
image_path = upload_image()
print(f"Uploaded Image: {image_path}")

# Ensure the model path is correct
model_path = '/content/runs/detect/train/weights/best.pt'
results_path = 'runs/detect/predict3'

# Run the detection model on the uploaded image
!yolo task=detect mode=predict model={model_path} source={image_path} save_txt save_conf save_crop

# Function to extract dates from cropped image regions
def extract_date_from_image(cropped_region):
    # Convert the cropped region to grayscale
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Apply dilation and erosion to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Adding custom options for Tesseract
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(eroded, config=custom_config)

    # Use regular expressions to find dates in the format YYYY.MM.DD
    date_pattern = r'\d{4}\.\d{2}\.\d{2}'
    matches = re.findall(date_pattern, text)

    # Return the first matched date or None if no date is found
    return matches[0] if matches else None

# Load the image with bounding boxes
result_image_path = f'{results_path}/{os.path.basename(image_path)}'
image_with_boxes = cv2.imread(result_image_path)

# Process the saved cropped regions from the YOLOv8 predictions
cropped_images_path = f'{results_path}/crops/date'
cropped_images = glob.glob(f'{cropped_images_path}/*.jpg')

extracted_dates = []
for cropped_image_path in cropped_images:
    cropped_image = cv2.imread(cropped_image_path)
    extracted_date = extract_date_from_image(cropped_image)

    # Store the extracted date
    extracted_dates.append((os.path.basename(cropped_image_path), extracted_date))

# Draw bounding boxes and display the image with bounding boxes
for label_path in glob.glob(f'{results_path}/labels/*.txt'):
    with open(label_path, 'r') as f:
        labels = f.readlines()

    for label in labels:
        parts = label.strip().split()
        class_id, x_center, y_center, width, height, confidence = map(float, parts)

        img_height, img_width, _ = image_with_boxes.shape
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw the bounding box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Print the extracted dates
for image_name, date in extracted_dates:
    if date:
        print(f"Extracted Date from {image_name}: {date}")
    else:
        print(f"No date found in cropped region of {image_name}")
