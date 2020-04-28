"""
!/usr/bin/env python
coding: utf-8

Created on Fri Apr  3 18:10:04 2020

@author: Tuomas Jalonen

This is my submission for Coursera's Deep Learning in Computer Vision week 2
peer-graded assigment 'Facial keypoints detection'.

"""

# ## Facial keypoints detection

# In this task you will create facial keypoint detector based on CNN regressor.

# ## Load and preprocess data

# Script `get_data.py` unpacks data — images and labelled points. 6000 images
# are located in `images` folder and keypoint coordinates are in `gt.csv` file.

# Now you have to read `gt.csv` file and images from `images` dir.
# File `gt.csv` contains header and ground truth points for every image in
# `images` folder. It has 29 columns. First column is a filename and next 28
# columns are `x` and `y` coordinates for 14 facepoints. We will make
# following preprocessing:

# 1. Scale all images to resolution $100 \times 100$ pixels.
# 2. Scale all coordinates to range $[-0.5; 0.5]$. To obtain that, divide
#   all x's by width (or number of columns) of image, and divide all y's by
# height (or number of rows) of image and subtract 0.5 from all values.

# Function `load_imgs_and_keypoint` should return a tuple of two numpy arrays:
# `imgs` of shape `(N, 100, 100, 3)`, where `N` is the number of images and
# `points` of shape `(N, 28)`.

# Let's load some packages:

import os
import pandas as pd
from skimage import io
from skimage import transform
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def load_imgs_and_keypoints(dir_name='data'):
    """
    This function loads the images and keypoints from the data

    Parameters
    ----------
    dir_name : The default is 'data'. This is the location of your
        'data' folder

    Returns
    -------
    imgs : Numpy array of shape (N, 100, 100, 3). N = 6000 by default.
        This is a tensor of RGB images that are read and resized to (100, 100).
    points : Numpy array of shape (N, 28). The 28 float values are facial
        keypoints [x1, y1, ..., x14, y14].

    """

    # Write your code for loading images and points here

    # 1. Load csv-file into pandas dataframe
    dataframe = pd.read_csv(os.path.join(dir_name, 'gt.csv'))

    # 2. Transform the dataframe into python list points
    points = dataframe.values.tolist()

    # 3. Let's create an array for images with shape (N, 100, 100, 3)
    imgs = np.ndarray(shape=(len(points), 100, 100, 3))

    # 4. Loop through rows:
    for row in range(len(points)):

        # 5. Get file_name from 1st column
        file_name = points[row][0]

        # 6. Get full path of the image
        path = os.path.join(dir_name, 'images', file_name)

        # 7. Read image
        image = io.imread(path)

        # 8. Get image width and height
        width = image.shape[0]
        height = image.shape[1]

        # 9. Loop through columns with step=2. The row is formed like
        # x1 y1 x2 y2 ..., so we are looping through them as pairs

        for column in range(1, 29, 2):

            # 10. X-coordinate is in the first column.
            # Let's scale it to [-0.5, 0.5] as instructed
            points[row][column] = points[row][column] / width - 0.5

            # 11. Y-coordinate is in the second column i.e. [column + 1]
            points[row][column + 1] = points[row][column + 1] / height -0.5

        # 12. The image is already read saved so let's resize it now and
        # add it to imgs
        image = transform.resize(image, (100, 100, 3))
        imgs[row] = image

        # 13. Print progress
        print(row + 1, '/', len(points))

    # 14. Our points list still has the file name as the first element in
    # the second axis so let's delete it and convert the list to float

    points = np.delete(points, 0, 1)
    points = np.array(points, dtype=float)

    return imgs, points

IMGS, POINTS = load_imgs_and_keypoints()

# ## Visualize data

# Let's prepare a function to visualize points on image. Such function obtains
# two arguments: an image and a vector of points' coordinates and draws points
# on image (just like first image in this notebook).

# Circle may be useful for drawing points on face
# See matplotlib documentation for more info

def visualize_points(img, points):
    """
    This function plots the points on the image.

    Parameters
    ----------
    img : A single image
    points : Facial keypoints of the image

    Returns
    -------
    None.

    """

    # Write here function which obtains image and normalized
    # coordinates and visualizes points on image

    # 1. Create image
    fig, axis = plt.subplots()

    # 2. Show the image

    axis.imshow(img)

    # 3. Loop through points list, and create a circle at each x,y pair. The
    # coordinates are scaled to the same coordinate system as the image
    # [0, 100].
    for index in range(0, 28, 2):
        x_coordinate = (points[index] + 0.5) * 100
        y_coordinate = (points[index + 1] + 0.5) * 100
        new_circle = Circle((x_coordinate, y_coordinate), radius=1)
        axis.add_patch(new_circle)

    # 4. Show the image
    plt.show()

visualize_points(IMGS[1], POINTS[1])

# ## Train/val split

# Run the following code to obtain train/validation split for training neural
# network.

IMGS_TRAIN, IMGS_VAL, POINTS_TRAIN, POINTS_VAL = train_test_split(IMGS, POINTS, test_size=0.1)

# ## Simple data augmentation

# For better training we will use simple data augmentation — flipping an image
# and points. Implement function flip_img which flips an image and its' points.
# Make sure that points are flipped correctly! For instance, points on right
# eye now should be points on left eye (i.e. you have to mirror coordinates
# and swap corresponding points on the left and right sides of the face).
# Visualize an example of original and flipped image.

def flip_img(img, points):

    """

    Parameters
    ----------
    img : Input image
    points : Facial keypoints of the input image

    Returns
    -------
    flipped_img : Horizontally flipped image
    swapped_points : Horizontally flipped points, that have been swapped
        according the position, e.g. Left eye (center) becomes right eye
        (center)

    """

    # Write your code for flipping here

    # 1. Flip the image horizontally:
    flipped_img = img[:, ::-1]

    # 2. Create an empty list for the flipped points
    flipped_points = []

    # 3. Loop through the points list
    for index in range(0, 28, 2):

        # 4. Flip the x-coordinates horizontally. Y-coordinates stay the same.
        new_x = - (points[index])
        new_y = points[index + 1]

        # 5. Append the coordinates to the list flipped_points
        flipped_points.append(new_x)
        flipped_points.append(new_y)

    # 6. Swap corresponding points:

    # Create an empty list of shape (28) for the swapped points
    swapped_points = np.empty(28)

    # Naming follows the rule: part (point location in the part). The comments
    # indicate which parts have to be swappped

    # Left eyebrow (left) <-> right eyebrow (right)
    swapped_points[6:8] = flipped_points[0:2]
    swapped_points[0:2] = flipped_points[6:8]

    # Left eyebrow (right) <-> right eyebrow(left)
    swapped_points[4:6] = flipped_points[2:4]
    swapped_points[2:4] = flipped_points[4:6]

    # Left eye (left) <-> right eye (right)
    swapped_points[18:20] = flipped_points[8:10]
    swapped_points[8:10] = flipped_points[18:20]

    # Left eye (center) <-> right eye (center)
    swapped_points[16:18] = flipped_points[10:12]
    swapped_points[10:12] = flipped_points[16:18]

    # Left eye (right) <-> right eye (left)
    swapped_points[14:16] = flipped_points[12:14]
    swapped_points[12:14] = flipped_points[14:16]

    # Nose and mouth (center) stay the same
    swapped_points[20:22] = flipped_points[20:22]
    swapped_points[24:26] = flipped_points[24:26]

    # Mouth (left) <-> mouth(right)
    swapped_points[26:28] = flipped_points[22:24]
    swapped_points[22:24] = flipped_points[26:28]

    return flipped_img, swapped_points

# Visualize a sample

F_IMG, F_POINTS = flip_img(IMGS[1], POINTS[1])

visualize_points(F_IMG, F_POINTS)

# Time to augment our training sample. Apply flip to every image in training
# sample. As a result you should obtain two arrays: `aug_imgs_train` and
# `aug_points_train` which contain original images and points along with
# flipped ones.

# Write your code here:

def create_aug_datasets(imgs, points):
    """
    This function creates the augmented datasets.

    Parameters
    ----------
    imgs : input images
    points : Facial keypoints of the input images

    Returns
    -------
    aug_imgs: Original + augmented (flipped) images
    aug_points: Original + augmented (flipped) points of the aug_images

    """

    # 1. Let's create two empty lists for our augmented datasets
    aug_imgs = []
    aug_points = []

    # 2. Loop through datasets
    for index in range(len(imgs)):

        # 3. Get flipped images and points from function flip_img
        flipped_img, flipped_points = flip_img(imgs[index], points[index])

        # 4. Append original and flipped images to the list
        aug_imgs.append(imgs[index])
        aug_imgs.append(flipped_img)

        # 5. Append original and flipped points to the list
        aug_points.append(points[index])
        aug_points.append(flipped_points)

    # 6. Convert to numpy arrays
    aug_imgs = np.array(aug_imgs)
    aug_points = np.array(aug_points)

    return aug_imgs, aug_points

# Create datasets with function create_aug_datasets
AUG_IMGS_TRAIN, AUG_POINTS_TRAIN = create_aug_datasets(IMGS_TRAIN,
                                                       POINTS_TRAIN
                                                       )

# Visualize some samples
visualize_points(AUG_IMGS_TRAIN[2], AUG_POINTS_TRAIN[2])
visualize_points(AUG_IMGS_TRAIN[4], AUG_POINTS_TRAIN[4])

"""

## Network architecture and training

Now let's define neural network regressor. It will have 28 outputs,
2 numbers per point. The precise architecture is up to you. We recommend to
add 2-3 (`Conv2D` + `MaxPooling2D`) pairs, then `Flatten` and 2-3 `Dense`
layers. Don't forget about ReLU activations. We also recommend to add
`Dropout` to every `Dense` layer (with p from 0.2 to 0.5) to prevent
overfitting.

Define here your model

"""

# Let's create a VGG-like convnet. Dropouts are not used because the model
# wasn't overfitting that much.

MODEL = Sequential()

MODEL.add(Conv2D(32, (3, 3),
                 activation='relu',
                 data_format='channels_last',
                 input_shape=(100, 100, 3)))
MODEL.add(MaxPooling2D(pool_size=(2, 2)))

MODEL.add(Conv2D(32, (3, 3), activation='relu'))
MODEL.add(Conv2D(32, (3, 3), activation='relu'))
MODEL.add(MaxPooling2D(pool_size=(2, 2)))

MODEL.add(Conv2D(64, (3, 3), activation='relu'))
MODEL.add(Conv2D(64, (3, 3), activation='relu'))
MODEL.add(MaxPooling2D(pool_size=(2, 2)))

MODEL.add(Flatten())

MODEL.add(Dense(1000, activation='relu'))
MODEL.add(Dense(256, activation='relu'))
MODEL.add(Dense(28))

# ## Time to train!

# Since we are training a regressor, make sure that you use
# mean squared error (mse) as loss. Feel free to experiment with optimization
# method (SGD, Adam, etc.) and its' parameters.

# ModelCheckpoint can be used for saving model during training.
# Saved models are useful for finetuning your model
# See keras documentation for more info

# Choose optimizer, compile model and run training

ADAM = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
MODEL.compile(loss='mean_squared_error', optimizer=ADAM)

MODEL.summary()

MC = ModelCheckpoint('best_model.h5',
                     monitor='val_loss',
                     mode='min',
                     verbose=1)

MODEL.fit(AUG_IMGS_TRAIN,
          AUG_POINTS_TRAIN,
          batch_size=32,
          epochs=10,
          callbacks=[MC],
          validation_data=(IMGS_VAL, POINTS_VAL),
          use_multiprocessing=True
          )
SCORE = MODEL.evaluate(IMGS_VAL, POINTS_VAL, batch_size=32)

# ## Visualize results

# Now visualize neural network results on several images from validation
# sample. Make sure that your network outputs different points for images
# (i.e. it doesn't output some constant).

for index in range(5):
    visualize_points(IMGS_VAL[index],
                     np.squeeze(MODEL.predict(IMGS_VAL[np.newaxis, index])))
