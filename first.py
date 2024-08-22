pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="x6B8kDRYvZTLUmKi6dXy")
project = rf.workspace("choi-t72ze").project("expiry-date")
version = project.version(6)
dataset = version.download("yolov8")

import yaml

dataset_location = "/content/Expiry-Date-6"  # Dataset location

# Reading the data.yaml file
yaml_file_path = f'{dataset_location}/data.yaml'
with open(yaml_file_path, 'r') as file:
    data = yaml.safe_load(file)

# Updating the 'path' key in the YAML data
data['path'] = dataset_location

# Writing the updated data back to the data.yaml file
with open(yaml_file_path, 'w') as file:
    yaml.dump(data, file, sort_keys=False)

!yolo task=detect mode=train model=yolov8m.pt data={dataset_location}/data.yaml epochs=100 imgsz=640

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the file path and the width
filename = '/content/runs/detect/train/confusion_matrix.png'
width = 600

# Load and display the image
img = mpimg.imread(filename)
height = img.shape[0] * width / img.shape[1]

plt.figure(figsize=(width / 100, height / 100))
plt.imshow(img)
plt.axis('off')  # Hide the axes
plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the file path and the width
filename = '/content/runs/detect/train/results.png'

# Load and display the image
img = mpimg.imread(filename)
height = img.shape[0] * width / img.shape[1]

plt.figure(figsize=(width / 100, height / 100))
plt.imshow(img)
plt.axis('off')  # Hide the axes
plt.show()


