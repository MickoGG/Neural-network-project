import glob
import os

import numpy as np

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import load_model

import matplotlib
matplotlib.use('TkAgg')     # Set backend
import matplotlib.pyplot as plt


# Initialization
data_path = './predict'
load_model_name = 'model.keras'

img_size = (224, 224)
batch_size = 64

# Class names
classes = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

# Getting data to predict
Xdata = image_dataset_from_directory(data_path,
                                     image_size=img_size,
                                     batch_size=batch_size,
                                     shuffle=False)

# Load model
model = load_model(load_model_name)

# Predicting data
data_predictions = np.array([])
data_images = []
for img, lab in Xdata:
    data_predictions = np.append(data_predictions, np.argmax(model.predict(img, verbose=0), axis=1))
    for elem in img:
        data_images.append(elem)

# Getting file names
file_names = glob.glob(f'{data_path}/**/*')
files_len = len(file_names)
file_names = [os.path.basename(file_names[i]) for i in range(files_len)]

# Printing results
print('==========RESULTS==========')
for i in range(files_len):
    print(file_names[i] + '  =>  ' + classes[int(data_predictions[i])])

# Showing results
plt.figure()
for i in range(files_len):
    pred = int(data_predictions[i])
    img = data_images[i]
    name = file_names[i]

    rows = files_len // 5
    if files_len % 5 != 0:
        rows += 1

    plt.subplot(rows, 5, i + 1)
    plt.imshow(img.numpy().astype('uint8'))
    plt.title(name + '\n' + classes[pred])
    plt.axis('off')

plt.show()
