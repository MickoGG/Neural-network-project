import os

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np

import matplotlib
matplotlib.use('TkAgg')     # Set backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Initialization
train_path = './VegetableImages/train'
val_path = './VegetableImages/validation'
test_path = './VegetableImages/test'

img_size = (224, 224)
batch_size = 64
ep = 50
patience = 7

load_existing_model = True
do_tuning_before_making_model = True
load_model_name = 'model.keras'
save_model_name = 'model.keras'


from tensorflow.keras.utils import image_dataset_from_directory

# Get training, validation and test data
Xtrain = image_dataset_from_directory(train_path,
                                      image_size=img_size,
                                      batch_size=batch_size)

Xval = image_dataset_from_directory(val_path,
                                    image_size=img_size,
                                    batch_size=batch_size)

Xtest = image_dataset_from_directory(test_path,
                                     image_size=img_size,
                                     batch_size=batch_size)


# Getting and printing class names
classes = Xtrain.class_names
print(classes)


# Showing a graph for the number of class samples
sorted_classes = sorted(classes)
plt.bar(sorted_classes,
        [len(os.listdir(f'{train_path}/{c}')) + len(os.listdir(f'{val_path}/{c}')) + len(os.listdir(f'{test_path}/{c}')) for c in sorted_classes])
plt.show()


# Showing one example of data from each class
num_classes = len(classes)
plt.figure()
for i in range(len(os.listdir('VegetableImages/examples'))):
    img = mpimg.imread(f'VegetableImages/examples/{sorted_classes[i]}.jpg')

    plt.subplot(2, int(num_classes / 2) + 1, i + 1)
    plt.imshow(img)
    plt.title(sorted_classes[i])
    plt.axis('off')

plt.show()


from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import keras_tuner as kt

data_augmentation = Sequential([
    layers.Input(shape=(img_size[0], img_size[1], 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.15)
])


# Creating a model
def make_model(hp):
    # Creating neural network
    new_model = Sequential([
        layers.Input(shape=(img_size[0], img_size[1], 3)),
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation="softmax")
    ])

    # Setting the hyperparameter
    if hp is not None:
        lr = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
    else:
        lr = 0.001
    opt = Adam(learning_rate=lr)

    # Model configuration for training
    new_model.compile(optimizer=opt,
                      loss=SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

    return new_model


def make_model_with_tuner():
    # Early stopping for the tuner
    es = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

    # Creating the tuner
    tuner = kt.RandomSearch(make_model,
                            objective='val_accuracy',
                            overwrite=True,
                            max_trials=3,
                            project_name="keras_tuner_randomsearch")

    # Tuning the model
    tuner.search(Xtrain,
                 epochs=5,      # or epochs=ep
                 batch_size=batch_size,
                 validation_data=Xval,
                 callbacks=[es],
                 verbose=2)

    # Retrieving the best hyperparameter
    best_hyperparam = tuner.get_best_hyperparameters(num_trials=1)[0]
    print('Optimal learning rate: ', best_hyperparam['learning_rate'])

    # Building the model with the best hyperparameter
    return tuner.hypermodel.build(best_hyperparam)


if load_existing_model:
    # Load existing model
    model = load_model(load_model_name)
elif do_tuning_before_making_model:
    # Make model with searching for the hyperparameter
    model = make_model_with_tuner()
else:
    # Make model without searching for the hyperparameter
    model = make_model(None)

# Prints network summary
model.summary()

if not load_existing_model:
    # Early stopping for the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, verbose=1, restore_best_weights=True)

    # Training the model
    history = model.fit(Xtrain,
                        epochs=ep,
                        batch_size=batch_size,
                        validation_data=Xval,
                        callbacks=[es],
                        verbose=2)

    # Showing graphs for (accuracy, validation_accuracy) and (loss, validation_loss)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure()
    plt.subplot(121)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Accuracy')
    plt.subplot(122)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Loss')
    plt.show()


print('\nCalculating performance...')


# Predictions on the training set
labels_train = np.array([])
predictions_train = np.array([])
for img, lab in Xtrain:
    labels_train = np.append(labels_train, lab)
    predictions_train = np.append(predictions_train, np.argmax(model.predict(img, verbose=0), axis=1))


# Lists for examples of good and bad predictions
well_pred_imgs = []     # Well-predicted images
well_labs = []          # Labels of well-predicted images

poorly_pred_imgs = []   # Poorly predicted images
actual_labs = []        # Actual labels of poorly predicted images
bad_labs = []           # Bad labels of poorly predicted images


# Retrieving well and poorly classified examples from the dataset
def find_good_and_bad_preds(img, lab, p):
    if len(actual_labs) < num_classes or len(well_labs) < num_classes:
        for i in range(len(lab)):
            if p[i] != lab[i] and lab[i] not in actual_labs:
                poorly_pred_imgs.append(img[i])
                actual_labs.append(lab[i])
                bad_labs.append(p[i])
            elif p[i] == lab[i] and lab[i] not in well_labs:
                well_pred_imgs.append(img[i])
                well_labs.append(p[i])


# Predictions on the test set
labels_test = np.array([])
predictions_test = np.array([])
for img, lab in Xtest:
    labels_test = np.append(labels_test, lab)
    p = np.argmax(model.predict(img, verbose=0), axis=1)
    predictions_test = np.append(predictions_test, p)

    find_good_and_bad_preds(img, lab, p)


from sklearn.metrics import accuracy_score

# Printing the model accuracy on the training set
print('The accuracy of the model on the training set is: ' + str(100 * accuracy_score(labels_train, predictions_train)) + '%')

# Printing the model accuracy on the test set
print('The accuracy of the model on the test set is: ' + str(100 * accuracy_score(labels_test, predictions_test)) + '%')


# Showing well-classified examples
plt.figure()
for i in range(len(well_labs)):
    img = well_pred_imgs[i]
    lab = well_labs[i]

    plt.subplot(2, int(num_classes / 2) + 1, i + 1)
    plt.imshow(img.numpy().astype('uint8'))
    plt.title(classes[lab])
    plt.axis('off')

plt.show()

# Showing misclassified examples
plt.figure()
for i in range(len(actual_labs)):
    img = poorly_pred_imgs[i]
    lab = actual_labs[i]
    bad_lab = bad_labs[i]

    plt.subplot(2, int(num_classes / 2) + 1, i + 1)
    plt.imshow(img.numpy().astype('uint8'))
    plt.title("P: " + str(classes[bad_lab]) + "\nR: " + str(classes[lab]))
    plt.axis('off')

plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Showing the confusion matrix for the training set
cm_train = confusion_matrix(labels_train, predictions_train, normalize='true')
cm_display_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=classes)
cm_display_train.plot()
plt.show()

# Showing the confusion matrix for the test set
cm_test = confusion_matrix(labels_test, predictions_test, normalize='true')
cm_display_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=classes)
cm_display_test.plot()
plt.show()


# Save model
save = input("Save the model ? (y/n): ").strip().lower()
if save in ('y', 'yes'):
    model.save(save_model_name)
    print('Model is saved.')
else:
    print('Model is not saved.')
