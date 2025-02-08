# Neural network project

### Vegetable image classification using deep learning.

This is a college project and the text of the project can be found in **`Project.pdf`** (Serbian language).  
Additionally, there is required **`Project report.pdf`** (also in Serbian language, but it contains useful screenshots).

The necessary Python modules/packages are listed in **`requirements.txt`**.

## Dataset

The dataset used in this project, along with detailed information, can be found on [Kaggle](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset). It is already split into **training, validation and test** sets.  
**Expected folder structure after downloading the dataset:**
```plaintext
VegetableImages/
  ├── train/
  ├── validation/
  └── test/
```

## Create, train and test the model
The **`main.py`** script is responsible for **creating, training and testing** the model using a neural network. At the end of execution, the user is prompted to save the model if desired.

### Key Features of `main.py`: 
- Getting and printing class names
- Showing a graph for the distribution of class samples
- Showing one example of data from each class
- Creating a model using a neural network
- Printing network summary
- Predictions on the training set
- Predictions on the test set
- Printing the model accuracy on the training set
- Printing the model accuracy on the test set
- Showing well-classified examples
- Showing misclassified examples
- Showing the confusion matrix for the training set
- Showing the confusion matrix for the test set
- Saving the model if desired

## Predict new data with the model

The **`predict_data.py`** script allows users to **predict custom images** using a saved model. It predicts images from all subfolders inside the **`predict/`** folder (in this example **`predict/data_to_predict/`**).

### Key Features of `predict_data.py`:
- Loading a model (e.g., model saved from **`main.py`**)
- Predicting all images in subfolders inside the **`predict/`** folder
- Printing results in the console
- Displaying results with a GUI
